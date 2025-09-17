"""
RNN Building Blocks for Time Series and HAR Tasks

This module contains various RNN architectures including LSTM variants,
GRU variants, and other recurrent architectures not available in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IndRNN(nn.Module):
    """
    Independently Recurrent Neural Network (IndRNN)
    
    Each neuron is independent and only connects to neurons in the next layer.
    This allows for much deeper RNN architectures.
    
    Reference: "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN"
    """
    
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='relu', 
                 recurrent_init_std=0.01):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        # Input-to-hidden weights
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        
        # Hidden-to-hidden weights (diagonal matrix - each neuron only connects to itself)
        self.weight_hh = nn.Parameter(torch.randn(hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            
        self.reset_parameters(recurrent_init_std)
        
    def reset_parameters(self, recurrent_init_std):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
            
        # Initialize recurrent weights with smaller std
        self.weight_hh.data.normal_(0, recurrent_init_std)
        
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = input.new_zeros(input.size(0), self.hidden_size)
            
        outputs = []
        
        for i in range(input.size(1)):  # For each time step
            x = input[:, i, :]
            
            # Linear transformation
            gi = F.linear(x, self.weight_ih, self.bias_ih)
            
            # Recurrent connection (element-wise multiplication)
            gh = hidden * self.weight_hh
            
            # Combine and apply activation
            hidden = gi + gh
            
            if self.nonlinearity == 'relu':
                hidden = F.relu(hidden)
            elif self.nonlinearity == 'tanh':
                hidden = torch.tanh(hidden)
            else:
                raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
                
            outputs.append(hidden.unsqueeze(1))
            
        return torch.cat(outputs, dim=1), hidden


class ConvLSTM1D(nn.Module):
    """
    1D Convolutional LSTM Cell
    
    Applies convolution operations instead of linear operations in LSTM.
    Useful for time series with spatial structure.
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Convolution for input-to-hidden
        self.conv_ih = nn.Conv1d(
            input_channels, 4 * hidden_channels, kernel_size, 
            padding=self.padding, bias=bias
        )
        
        # Convolution for hidden-to-hidden
        self.conv_hh = nn.Conv1d(
            hidden_channels, 4 * hidden_channels, kernel_size,
            padding=self.padding, bias=bias
        )
        
    def forward(self, input, hidden=None):
        if hidden is None:
            h = input.new_zeros(input.size(0), self.hidden_channels, input.size(2))
            c = input.new_zeros(input.size(0), self.hidden_channels, input.size(2))
        else:
            h, c = hidden
            
        outputs = []
        
        for i in range(input.size(1)):  # For each time step
            x = input[:, i, :, :]  # (batch, channels, length)
            
            # Compute gates
            gi = self.conv_ih(x)
            gh = self.conv_hh(h)
            
            i_gate, f_gate, g_gate, o_gate = torch.split(gi + gh, self.hidden_channels, dim=1)
            
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            g_gate = torch.tanh(g_gate)
            o_gate = torch.sigmoid(o_gate)
            
            # Update cell state
            c = f_gate * c + i_gate * g_gate
            
            # Update hidden state
            h = o_gate * torch.tanh(c)
            
            outputs.append(h.unsqueeze(1))
            
        return torch.cat(outputs, dim=1), (h, c)


class LayerNormLSTM(nn.Module):
    """
    LSTM with Layer Normalization
    
    Applies layer normalization to improve training stability and convergence.
    """
    
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input-to-hidden weights
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        # Hidden-to-hidden weights
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        # Layer normalization
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                weight.data.uniform_(-std, std)
            else:
                weight.data.zero_()
                
    def forward(self, input, hidden=None):
        if hidden is None:
            h = input.new_zeros(input.size(0), self.hidden_size)
            c = input.new_zeros(input.size(0), self.hidden_size)
        else:
            h, c = hidden
            
        outputs = []
        
        for i in range(input.size(1)):
            x = input[:, i, :]
            
            # Compute pre-activations
            gi = F.linear(x, self.weight_ih, self.bias_ih)
            gh = F.linear(h, self.weight_hh, self.bias_hh)
            
            # Apply layer normalization
            gi = self.ln_ih(gi)
            gh = self.ln_hh(gh)
            
            # Split into gates
            i_gate, f_gate, g_gate, o_gate = torch.split(gi + gh, self.hidden_size, dim=1)
            
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            g_gate = torch.tanh(g_gate)
            o_gate = torch.sigmoid(o_gate)
            
            # Update cell state with layer norm
            c = f_gate * c + i_gate * g_gate
            c_norm = self.ln_c(c)
            
            # Update hidden state
            h = o_gate * torch.tanh(c_norm)
            
            outputs.append(h.unsqueeze(1))
            
        return torch.cat(outputs, dim=1), (h, c)


class PeepholeLSTM(nn.Module):
    """
    LSTM with Peephole Connections
    
    Allows gate layers to look at the cell state for better long-term dependencies.
    """
    
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Standard LSTM weights
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        
        # Peephole connections
        self.weight_ci = nn.Parameter(torch.randn(hidden_size))  # cell to input gate
        self.weight_cf = nn.Parameter(torch.randn(hidden_size))  # cell to forget gate
        self.weight_co = nn.Parameter(torch.randn(hidden_size))  # cell to output gate
        
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
            
    def forward(self, input, hidden=None):
        if hidden is None:
            h = input.new_zeros(input.size(0), self.hidden_size)
            c = input.new_zeros(input.size(0), self.hidden_size)
        else:
            h, c = hidden
            
        outputs = []
        
        for i in range(input.size(1)):
            x = input[:, i, :]
            
            gi = F.linear(x, self.weight_ih, self.bias_ih)
            gh = F.linear(h, self.weight_hh, self.bias_hh)
            
            i_i, i_f, i_g, i_o = torch.split(gi, self.hidden_size, dim=1)
            h_i, h_f, h_g, h_o = torch.split(gh, self.hidden_size, dim=1)
            
            # Input gate with peephole
            i_gate = torch.sigmoid(i_i + h_i + self.weight_ci * c)
            
            # Forget gate with peephole
            f_gate = torch.sigmoid(i_f + h_f + self.weight_cf * c)
            
            # Cell gate
            g_gate = torch.tanh(i_g + h_g)
            
            # Update cell state
            c = f_gate * c + i_gate * g_gate
            
            # Output gate with peephole
            o_gate = torch.sigmoid(i_o + h_o + self.weight_co * c)
            
            # Update hidden state
            h = o_gate * torch.tanh(c)
            
            outputs.append(h.unsqueeze(1))
            
        return torch.cat(outputs, dim=1), (h, c)


class BiDirectionalLSTM(nn.Module):
    """Enhanced Bidirectional LSTM with various options"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=True, dropout=0.0, lstm_type='standard'):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        if lstm_type == 'standard':
            self.forward_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                      bias=bias, batch_first=batch_first, dropout=dropout)
            self.backward_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                       bias=bias, batch_first=batch_first, dropout=dropout)
        elif lstm_type == 'layer_norm':
            self.forward_lstm = LayerNormLSTM(input_size, hidden_size, bias)
            self.backward_lstm = LayerNormLSTM(input_size, hidden_size, bias)
        elif lstm_type == 'peephole':
            self.forward_lstm = PeepholeLSTM(input_size, hidden_size, bias)
            self.backward_lstm = PeepholeLSTM(input_size, hidden_size, bias)
        else:
            raise ValueError(f"Unknown LSTM type: {lstm_type}")
            
        self.lstm_type = lstm_type
        
    def forward(self, input, hidden=None):
        if self.lstm_type == 'standard':
            # Standard PyTorch LSTM
            forward_out, forward_hidden = self.forward_lstm(input, hidden)
            
            # Reverse input for backward pass
            if self.batch_first:
                backward_input = torch.flip(input, [1])
            else:
                backward_input = torch.flip(input, [0])
                
            backward_out, backward_hidden = self.backward_lstm(backward_input, hidden)
            
            # Reverse backward output
            if self.batch_first:
                backward_out = torch.flip(backward_out, [1])
            else:
                backward_out = torch.flip(backward_out, [0])
        else:
            # Custom LSTM implementations
            forward_out, forward_hidden = self.forward_lstm(input, hidden)
            
            # Reverse input
            backward_input = torch.flip(input, [1])
            backward_out, backward_hidden = self.backward_lstm(backward_input, hidden)
            
            # Reverse backward output
            backward_out = torch.flip(backward_out, [1])
            
        # Concatenate forward and backward outputs
        output = torch.cat([forward_out, backward_out], dim=-1)
        
        return output, (forward_hidden, backward_hidden)


class StackedLSTM(nn.Module):
    """Stacked LSTM with different configurations per layer"""
    
    def __init__(self, input_size, hidden_sizes, lstm_types=None, dropouts=None):
        super().__init__()
        
        if lstm_types is None:
            lstm_types = ['standard'] * len(hidden_sizes)
        if dropouts is None:
            dropouts = [0.0] * len(hidden_sizes)
            
        self.layers = nn.ModuleList()
        
        for i, (hidden_size, lstm_type, dropout) in enumerate(zip(hidden_sizes, lstm_types, dropouts)):
            layer_input_size = input_size if i == 0 else hidden_sizes[i-1]
            
            if lstm_type == 'standard':
                layer = nn.LSTM(layer_input_size, hidden_size, batch_first=True)
            elif lstm_type == 'layer_norm':
                layer = LayerNormLSTM(layer_input_size, hidden_size)
            elif lstm_type == 'peephole':
                layer = PeepholeLSTM(layer_input_size, hidden_size)
            elif lstm_type == 'indRNN':
                layer = IndRNN(layer_input_size, hidden_size)
            else:
                raise ValueError(f"Unknown LSTM type: {lstm_type}")
                
            self.layers.append(layer)
            
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
                
    def forward(self, input, hidden=None):
        output = input
        hidden_states = []
        
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                output = layer(output)
            else:
                if isinstance(layer, nn.LSTM):
                    output, h = layer(output, hidden)
                else:
                    output, h = layer(output, hidden)
                hidden_states.append(h)
                
        return output, hidden_states


class GRUCell(nn.Module):
    """Enhanced GRU Cell with additional features"""
    
    def __init__(self, input_size, hidden_size, bias=True, layer_norm=False):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        
        # Input-to-hidden weights
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        # Hidden-to-hidden weights
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        if layer_norm:
            self.ln_ih = nn.LayerNorm(3 * hidden_size)
            self.ln_hh = nn.LayerNorm(3 * hidden_size)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                weight.data.uniform_(-std, std)
            else:
                weight.data.zero_()
                
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = input.new_zeros(input.size(0), self.hidden_size)
            
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden, self.weight_hh, self.bias_hh)
        
        if self.layer_norm:
            gi = self.ln_ih(gi)
            gh = self.ln_hh(gh)
            
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        
        hy = new_gate + update_gate * (hidden - new_gate)
        
        return hy


class QRNN(nn.Module):
    """
    Quasi-Recurrent Neural Network (QRNN)
    
    Combines convolution and recurrence for efficient sequence modeling.
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, kernel_size=2, dropout=0.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(QRNNLayer(layer_input_size, hidden_size, kernel_size, dropout))
            
    def forward(self, input, hidden=None):
        output = input
        hidden_states = []
        
        for i, layer in enumerate(self.layers):
            h = None if hidden is None else hidden[i]
            output, h = layer(output, h)
            hidden_states.append(h)
            
        return output, hidden_states


class QRNNLayer(nn.Module):
    """Single QRNN Layer"""
    
    def __init__(self, input_size, hidden_size, kernel_size=2, dropout=0.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        # Convolution for Z, F, O gates
        self.conv = nn.Conv1d(input_size, 3 * hidden_size, kernel_size, padding=kernel_size-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden=None):
        # input: (batch, seq_len, input_size)
        # Transpose for conv1d: (batch, input_size, seq_len)
        x = input.transpose(1, 2)
        
        # Apply convolution
        conv_out = self.conv(x)
        
        # Remove extra padding
        if self.kernel_size > 1:
            conv_out = conv_out[:, :, :-(self.kernel_size-1)]
            
        # Transpose back: (batch, seq_len, 3*hidden_size)
        conv_out = conv_out.transpose(1, 2)
        
        # Split into gates
        Z, F, O = conv_out.chunk(3, dim=-1)
        
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)
        
        # Apply recurrence (fo-pooling)
        outputs = []
        h = hidden if hidden is not None else Z.new_zeros(Z.size(0), self.hidden_size)
        
        for t in range(Z.size(1)):
            z_t = Z[:, t]
            f_t = F[:, t]
            o_t = O[:, t]
            
            h = f_t * h + (1 - f_t) * z_t
            output = o_t * h
            outputs.append(output.unsqueeze(1))
            
        output = torch.cat(outputs, dim=1)
        output = self.dropout(output)
        
        return output, h
