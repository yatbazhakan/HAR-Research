"""
Attention Modules for HAR and Time Series Tasks

This module contains various attention mechanisms commonly used in
deep learning for human activity recognition and time series analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism for sequence modeling
    
    Computes attention weights for each position in the sequence.
    """
    
    def __init__(self, input_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Attention mask (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute queries, keys, values
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.bmm(attention_weights, V)
        
        return output, attention_weights


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism between two sequences
    
    Allows one sequence to attend to another sequence.
    """
    
    def __init__(self, query_dim, key_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = min(query_dim, key_dim)
            
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(query_dim, hidden_dim)
        self.key = nn.Linear(key_dim, hidden_dim)
        self.value = nn.Linear(key_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)
        
    def forward(self, query_seq, key_seq, mask=None):
        """
        Args:
            query_seq: Query sequence (batch_size, query_len, query_dim)
            key_seq: Key/Value sequence (batch_size, key_len, key_dim)
            mask: Attention mask (batch_size, query_len, key_len)
        """
        # Compute queries, keys, values
        Q = self.query(query_seq)
        K = self.key(key_seq)
        V = self.value(key_seq)
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.bmm(attention_weights, V)
        
        return output, attention_weights


class SpatialAttention(nn.Module):
    """
    Spatial Attention for multi-channel signals
    
    Computes attention weights across spatial dimensions (channels/sensors).
    """
    
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction),
            nn.ReLU(),
            nn.Linear(num_channels // reduction, num_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, num_channels, seq_len)
        """
        # Global average and max pooling across time dimension
        avg_out = self.avg_pool(x).squeeze(-1)  # (batch_size, num_channels)
        max_out = self.max_pool(x).squeeze(-1)
        
        # Compute attention weights
        avg_weights = self.fc(avg_out).unsqueeze(-1)  # (batch_size, num_channels, 1)
        max_weights = self.fc(max_out).unsqueeze(-1)
        
        # Combine and apply weights
        attention_weights = avg_weights + max_weights
        output = x * attention_weights
        
        return output, attention_weights.squeeze(-1)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM)
    
    Learns to emphasize important channels in multi-channel time series.
    """
    
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction),
            nn.ReLU(),
            nn.Linear(num_channels // reduction, num_channels)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, num_channels, seq_len)
        """
        # Global pooling
        avg_out = self.avg_pool(x).squeeze(-1)  # (batch_size, num_channels)
        max_out = self.max_pool(x).squeeze(-1)
        
        # Shared MLP
        avg_out = self.shared_mlp(avg_out)
        max_out = self.shared_mlp(max_out)
        
        # Combine and apply sigmoid
        attention_weights = torch.sigmoid(avg_out + max_out)
        attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, num_channels, 1)
        
        output = x * attention_weights
        
        return output, attention_weights.squeeze(-1)


class TemporalAttention(nn.Module):
    """
    Temporal Attention Module (TAM)
    
    Learns to emphasize important time steps in sequences.
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = attention_weights.unsqueeze(-1) * x  # (batch_size, seq_len, hidden_dim)
        output = torch.sum(attended, dim=1)  # (batch_size, hidden_dim)
        
        return output, attention_weights


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    
    Combines channel and spatial attention sequentially.
    """
    
    def __init__(self, num_channels, reduction=16, kernel_size=7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(num_channels, reduction)
        self.spatial_attention = SpatialAttentionCBAM(kernel_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, num_channels, seq_len)
        """
        # Apply channel attention
        x, channel_weights = self.channel_attention(x)
        
        # Apply spatial attention
        x, spatial_weights = self.spatial_attention(x)
        
        return x, (channel_weights, spatial_weights)


class SpatialAttentionCBAM(nn.Module):
    """Spatial attention component of CBAM for 1D signals"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, num_channels, seq_len)
        """
        # Compute channel-wise average and max
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, seq_len)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        combined = torch.cat([avg_out, max_out], dim=1)  # (batch_size, 2, seq_len)
        attention_weights = torch.sigmoid(self.conv(combined))  # (batch_size, 1, seq_len)
        
        output = x * attention_weights
        
        return output, attention_weights.squeeze(1)


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA)
    
    Lightweight channel attention with 1D convolution.
    """
    
    def __init__(self, num_channels, k_size=3):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, num_channels, seq_len)
        """
        # Global average pooling
        y = self.avg_pool(x)  # (batch_size, num_channels, 1)
        
        # 1D convolution across channels
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Apply sigmoid and broadcast
        attention_weights = torch.sigmoid(y)  # (batch_size, num_channels, 1)
        
        output = x * attention_weights
        
        return output, attention_weights.squeeze(-1)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for capturing positional information
    
    Decomposes channel attention into two 1D feature encoding processes.
    """
    
    def __init__(self, num_channels, reduction=32):
        super().__init__()
        
        self.pool_h = nn.AdaptiveAvgPool1d(1)
        
        mip = max(8, num_channels // reduction)
        
        self.conv1 = nn.Conv1d(num_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip)
        self.act = nn.ReLU()
        
        self.conv_h = nn.Conv1d(mip, num_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, num_channels, seq_len)
        """
        identity = x
        
        n, c, h = x.size()
        
        # Global average pooling
        x_h = self.pool_h(x)  # (batch_size, num_channels, 1)
        
        # Combine and reduce dimensions
        y = self.conv1(x_h)
        y = self.bn1(y)
        y = self.act(y)
        
        # Generate attention weights
        a_h = self.conv_h(y).sigmoid()
        
        output = identity * a_h.expand_as(identity)
        
        return output, a_h.squeeze(-1)


class NonLocalAttention(nn.Module):
    """
    Non-Local Attention for capturing long-range dependencies
    
    Computes attention between all pairs of positions in the sequence.
    """
    
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction
        
        self.theta = nn.Conv1d(in_channels, self.inter_channels, 1, 1, 0, bias=False)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, 1, 1, 0, bias=False)
        self.g = nn.Conv1d(in_channels, self.inter_channels, 1, 1, 0, bias=False)
        self.W = nn.Conv1d(self.inter_channels, in_channels, 1, 1, 0, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, in_channels, seq_len)
        """
        batch_size, channels, seq_len = x.size()
        
        # Compute theta, phi, g
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # (batch_size, seq_len, inter_channels)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # (batch_size, seq_len, inter_channels)
        
        # Compute attention
        f = torch.matmul(theta_x, phi_x)  # (batch_size, seq_len, seq_len)
        f = F.softmax(f, dim=-1)
        
        # Apply attention
        y = torch.matmul(f, g_x)  # (batch_size, seq_len, inter_channels)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, seq_len)
        
        # Final transformation
        W_y = self.W(y)
        z = W_y + x
        
        return z, f


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block
    
    Classic channel attention mechanism from SENet.
    """
    
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_channels, num_channels // reduction)
        self.fc2 = nn.Linear(num_channels // reduction, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, num_channels, seq_len)
        """
        # Squeeze
        b, c, _ = x.size()
        y = self.global_pool(x).view(b, c)
        
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Scale
        y = y.view(b, c, 1)
        output = x * y.expand_as(x)
        
        return output, y.squeeze(-1)


class GatedAttention(nn.Module):
    """
    Gated Attention mechanism
    
    Uses gating to control information flow in attention.
    """
    
    def __init__(self, input_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention components
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Gating components
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        """
        # Compute attention scores
        attention_scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute gates
        gates = self.gate(x)  # (batch_size, seq_len, hidden_dim)
        
        # Apply gating and attention
        gated_x = x * gates  # Element-wise gating
        attended = attention_weights.unsqueeze(-1) * gated_x
        output = torch.sum(attended, dim=1)  # (batch_size, input_dim)
        
        output = self.dropout(output)
        
        return output, attention_weights


class MultiScaleAttention(nn.Module):
    """
    Multi-Scale Attention for capturing patterns at different time scales
    
    Uses multiple attention heads with different receptive fields.
    """
    
    def __init__(self, input_dim, num_scales=3, hidden_dim=None, dropout=0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.num_scales = num_scales
        self.attentions = nn.ModuleList()
        
        # Create attention modules for different scales
        for i in range(num_scales):
            kernel_size = 2 ** (i + 1) + 1  # 3, 5, 9, ...
            attention = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, 1, 1),
                nn.Sigmoid()
            )
            self.attentions.append(attention)
            
        self.fusion = nn.Linear(num_scales * input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        """
        # Transpose for convolution
        x_conv = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        attended_features = []
        attention_maps = []
        
        for attention in self.attentions:
            # Compute attention weights
            att_weights = attention(x_conv)  # (batch_size, 1, seq_len)
            attention_maps.append(att_weights.squeeze(1))
            
            # Apply attention
            attended = x_conv * att_weights  # (batch_size, input_dim, seq_len)
            attended = attended.transpose(1, 2)  # (batch_size, seq_len, input_dim)
            attended_features.append(attended)
            
        # Fuse multi-scale features
        fused = torch.cat(attended_features, dim=-1)  # (batch_size, seq_len, num_scales * input_dim)
        output = self.fusion(fused)  # (batch_size, seq_len, input_dim)
        output = self.dropout(output)
        
        return output, attention_maps
