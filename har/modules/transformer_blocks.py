"""
Transformer Building Blocks for Time Series and HAR Tasks

This module contains transformer architectures that build upon existing
PyTorch transformer components, optimized for time series data and HAR tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models
    
    Supports both sinusoidal and learnable positional encodings.
    """
    
    def __init__(self, d_model, max_len=5000, encoding_type='sinusoidal', dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout)
        
        if encoding_type == 'sinusoidal':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
            
        elif encoding_type == 'learnable':
            self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))
            
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
            
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
        """
        if x.dim() == 3 and x.size(1) == self.d_model:
            # Assume (batch_size, d_model, seq_len) - transpose
            x = x.transpose(1, 2)
            
        if x.size(0) != x.size(1):  # (batch_size, seq_len, d_model)
            x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, d_model)
            
        seq_len = x.size(0)
        
        if self.encoding_type == 'sinusoidal':
            x = x + self.pe[:seq_len, :]
        else:
            x = x + self.pe[:seq_len, :]
            
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Wrapper around PyTorch's MultiheadAttention with additional features
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1, batch_first=True):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: Input tensors (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len, seq_len) or (seq_len, seq_len)
        """
        # Convert mask format if needed
        if mask is not None and mask.dim() == 3:
            # PyTorch expects (seq_len, seq_len) or (batch_size * num_heads, seq_len, seq_len)
            if mask.size(0) == query.size(0):  # (batch_size, seq_len, seq_len)
                mask = mask[0]  # Use first batch's mask for all
        
        output, attention_weights = self.attention(query, key, value, attn_mask=mask)
        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Standard Transformer Encoder Block using PyTorch's TransformerEncoderLayer"""
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, 
                 activation='relu', layer_norm_eps=1e-6, batch_first=True):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.transformer_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first
        )
        
    def forward(self, x, mask=None):
        # PyTorch's TransformerEncoderLayer handles everything internally
        output = self.transformer_layer(x, src_key_padding_mask=mask)
        return output, None  # PyTorch doesn't return attention weights by default


class ConvTransformerBlock(nn.Module):
    """
    Convolutional Transformer Block
    
    Combines convolution with transformer attention for local and global modeling.
    """
    
    def __init__(self, d_model, num_heads, kernel_size=3, dropout=0.1):
        super().__init__()
        
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size, 
                               padding=kernel_size//2, groups=d_model)
        self.transformer_block = TransformerBlock(d_model, num_heads, dropout=dropout)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Apply convolution (local modeling)
        residual = x
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_model)
        x = self.norm(residual + self.dropout(x_conv))
        
        # Apply transformer (global modeling)
        x, attn_weights = self.transformer_block(x, mask)
        
        return x, attn_weights


class TimeSeriesTransformer(nn.Module):
    """
    Complete Transformer model for time series data using PyTorch's TransformerEncoder
    
    Includes input projection, positional encoding, and stacked transformer blocks.
    """
    
    def __init__(self, input_dim, d_model, num_heads, num_layers, 
                 max_seq_len=1000, dropout=0.1, output_dim=None,
                 pos_encoding='sinusoidal', activation='relu'):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, pos_encoding, dropout)
        
        # Use PyTorch's TransformerEncoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        if output_dim:
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            self.output_projection = None
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Attention mask (batch_size, seq_len) - padding mask
        """
        # Input projection
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        # Output projection
        if self.output_projection:
            x = self.output_projection(x)
            
        return x


class LinearAttention(nn.Module):
    """
    Linear Attention mechanism for efficient computation
    
    Reduces complexity from O(n²) to O(n) for long sequences.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Apply feature map (ELU + 1 for positivity)
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Linear attention computation
        KV = torch.matmul(K.transpose(-2, -1), V)  # (batch, heads, d_k, d_k)
        Z = torch.matmul(Q, KV)  # (batch, heads, seq_len, d_k)
        
        # Normalization
        D = torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1))
        Z = Z / (D + 1e-6)
        
        # Concatenate heads
        Z = Z.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear transformation
        output = self.w_o(Z)
        
        return output


class ReversibleTransformerBlock(nn.Module):
    """
    Reversible Transformer Block for memory-efficient training
    
    Uses reversible residual connections to reduce memory usage.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model // 2, num_heads, dropout)
        self.feed_forward = FeedForward(d_model // 2, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.norm2 = nn.LayerNorm(d_model // 2)
        
    def forward(self, x):
        # Split input
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        # Attention block
        y1 = x1 + self.attention(self.norm1(x2), self.norm1(x2), self.norm1(x2))[0]
        
        # Feed-forward block
        y2 = x2 + self.feed_forward(self.norm2(y1))
        
        return torch.cat([y1, y2], dim=-1)
    
    def backward_pass(self, y):
        # Reverse the forward pass
        y1, y2 = torch.chunk(y, 2, dim=-1)
        
        # Reverse feed-forward
        x2 = y2 - self.feed_forward(self.norm2(y1))
        
        # Reverse attention
        x1 = y1 - self.attention(self.norm1(x2), self.norm1(x2), self.norm1(x2))[0]
        
        return torch.cat([x1, x2], dim=-1)


class LocalGlobalTransformer(nn.Module):
    """
    Transformer with separate local and global attention
    
    Processes local patterns with convolution and global patterns with attention.
    """
    
    def __init__(self, d_model, num_heads, local_window=32, dropout=0.1):
        super().__init__()
        
        self.local_window = local_window
        
        # Local processing
        self.local_conv = nn.Conv1d(d_model, d_model, local_window, 
                                   padding=local_window//2, groups=d_model)
        self.local_norm = nn.LayerNorm(d_model)
        
        # Global processing
        self.global_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.global_norm = nn.LayerNorm(d_model)
        
        # Fusion
        self.fusion = nn.Linear(2 * d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        
        # Local processing
        x_local = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_local = self.local_conv(x_local)
        x_local = x_local.transpose(1, 2)  # (batch, seq_len, d_model)
        x_local = self.local_norm(residual + self.dropout(x_local))
        
        # Global processing
        x_global, _ = self.global_attention(x, x, x, mask)
        x_global = self.global_norm(residual + self.dropout(x_global))
        
        # Fusion
        x_fused = torch.cat([x_local, x_global], dim=-1)
        output = self.fusion(x_fused)
        
        return output


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for sequence-to-sequence tasks using PyTorch's TransformerDecoder
    """
    
    def __init__(self, input_dim, d_model, num_heads, num_layers, 
                 max_seq_len=1000, dropout=0.1, output_dim=None):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout=dropout)
        
        # Use PyTorch's TransformerDecoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        if output_dim:
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            self.output_projection = None
            
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: Target sequence (batch_size, tgt_len, input_dim)
            memory: Encoder output (batch_size, src_len, d_model)
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
        """
        # Input projection
        tgt = self.input_projection(tgt) * math.sqrt(self.d_model)
        
        # Positional encoding
        tgt = self.pos_encoding(tgt)
        
        # Apply transformer decoder
        output = self.transformer(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        output = self.norm(output)
        
        # Output projection
        if self.output_projection:
            output = self.output_projection(output)
            
        return output


class TransformerEncoderDecoder(nn.Module):
    """
    Complete Encoder-Decoder Transformer using PyTorch's components
    """
    
    def __init__(self, src_input_dim, tgt_input_dim, d_model, num_heads, 
                 num_encoder_layers, num_decoder_layers, max_seq_len=1000, 
                 dropout=0.1, output_dim=None):
        super().__init__()
        
        self.encoder = TimeSeriesTransformer(
            input_dim=src_input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            output_dim=None  # Keep d_model for decoder
        )
        
        self.decoder = TransformerDecoder(
            input_dim=tgt_input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            output_dim=output_dim
        )
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            src: Source sequence (batch_size, src_len, src_input_dim)
            tgt: Target sequence (batch_size, tgt_len, tgt_input_dim)
        """
        # Encode source
        memory = self.encoder(src, mask=src_key_padding_mask)
        
        # Decode target
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return output


class TransformerWithConv(nn.Module):
    """
    Hybrid Conv-Transformer model for time series
    
    Uses convolution for local patterns and transformer for global patterns.
    """
    
    def __init__(self, input_dim, d_model, num_heads, num_layers,
                 conv_layers=2, conv_channels=None, kernel_size=3,
                 max_seq_len=1000, dropout=0.1, output_dim=None):
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [d_model // 2, d_model]
            
        # Convolutional layers for local patterns
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
            
        # Transformer for global patterns
        self.transformer = TimeSeriesTransformer(
            input_dim=conv_channels[-1],
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            output_dim=output_dim
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        """
        # Apply convolutional layers
        x_conv = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        for conv_layer in self.conv_layers:
            x_conv = conv_layer(x_conv)
            
        x_conv = x_conv.transpose(1, 2)  # (batch_size, seq_len, channels)
        
        # Apply transformer
        output = self.transformer(x_conv, mask=mask)
        
        return output


class TransformerWithPooling(nn.Module):
    """
    Transformer with adaptive pooling for variable-length sequences
    """
    
    def __init__(self, input_dim, d_model, num_heads, num_layers,
                 max_seq_len=1000, dropout=0.1, output_dim=None,
                 pooling_type='mean'):
        super().__init__()
        
        self.transformer = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            output_dim=None  # Keep d_model for pooling
        )
        
        if pooling_type == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling_type == 'attention':
            self.pooling = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softmax(dim=1)
            )
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
            
        self.pooling_type = pooling_type
        
        if output_dim:
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            self.output_projection = None
            
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        """
        # Apply transformer
        x = self.transformer(x, mask=mask)
        
        # Apply pooling
        if self.pooling_type == 'attention':
            # Attention pooling
            attention_weights = self.pooling(x)  # (batch_size, seq_len, 1)
            output = torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)
        else:
            # Mean/Max pooling
            x_pool = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
            output = self.pooling(x_pool).squeeze(-1)  # (batch_size, d_model)
            
        # Output projection
        if self.output_projection:
            output = self.output_projection(output)
            
        return output
