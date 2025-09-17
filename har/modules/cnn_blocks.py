"""
CNN Building Blocks for Time Series and HAR Tasks

This module contains various CNN architectures and blocks commonly used
in human activity recognition and time series analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock1D(nn.Module):
    """Basic 1D Convolutional Block with BatchNorm and Activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=None, dilation=1, groups=1, bias=False, 
                 norm_layer=nn.BatchNorm1d, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
            
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, groups, bias)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.activation = activation() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock1D(nn.Module):
    """1D Residual Block with optional bottleneck"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 bottleneck=False, expansion=4, dropout=0.1):
        super().__init__()
        
        self.stride = stride
        self.bottleneck = bottleneck
        
        if bottleneck:
            # Bottleneck design: 1x1 -> 3x3 -> 1x1
            mid_channels = out_channels // expansion
            self.conv1 = ConvBlock1D(in_channels, mid_channels, 1, 1, 0, dropout=dropout)
            self.conv2 = ConvBlock1D(mid_channels, mid_channels, kernel_size, stride, dropout=dropout)
            self.conv3 = nn.Conv1d(mid_channels, out_channels, 1, 1, 0, bias=False)
            self.norm3 = nn.BatchNorm1d(out_channels)
        else:
            # Basic design: 3x3 -> 3x3
            self.conv1 = ConvBlock1D(in_channels, out_channels, kernel_size, stride, dropout=dropout)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, 
                                  (kernel_size - 1) // 2, bias=False)
            self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        if self.bottleneck:
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.norm3(out)
        else:
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.norm2(out)
            
        out += residual
        out = self.activation(out)
        return out


class DepthwiseSeparableConv1D(nn.Module):
    """Depthwise Separable Convolution for efficient computation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=None, dilation=1, bias=False, dropout=0.0):
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
            
        # Depthwise convolution
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride,
                                  padding, dilation, groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation Block for 1D signals"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class InceptionBlock1D(nn.Module):
    """1D Inception Block with multiple kernel sizes"""
    
    def __init__(self, in_channels, out_channels_list, reduce_channels=None):
        super().__init__()
        
        if reduce_channels is None:
            reduce_channels = in_channels // 4
            
        # 1x1 conv branch
        self.branch1 = ConvBlock1D(in_channels, out_channels_list[0], 1)
        
        # 1x1 -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            ConvBlock1D(in_channels, reduce_channels, 1),
            ConvBlock1D(reduce_channels, out_channels_list[1], 3)
        )
        
        # 1x1 -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            ConvBlock1D(in_channels, reduce_channels, 1),
            ConvBlock1D(reduce_channels, out_channels_list[2], 5)
        )
        
        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(3, 1, 1),
            ConvBlock1D(in_channels, out_channels_list[3], 1)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class DilatedConvBlock(nn.Module):
    """Dilated Convolution Block for capturing long-range dependencies"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rates=None):
        super().__init__()
        
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]
            
        self.dilated_convs = nn.ModuleList()
        channels_per_branch = out_channels // len(dilation_rates)
        
        for dilation in dilation_rates:
            self.dilated_convs.append(
                ConvBlock1D(in_channels, channels_per_branch, kernel_size, 
                           dilation=dilation)
            )
            
        # Adjust channels if not evenly divisible
        remaining_channels = out_channels - channels_per_branch * len(dilation_rates)
        if remaining_channels > 0:
            self.dilated_convs.append(
                ConvBlock1D(in_channels, remaining_channels, kernel_size)
            )
            
    def forward(self, x):
        outputs = []
        for conv in self.dilated_convs:
            outputs.append(conv(x))
        return torch.cat(outputs, 1)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network (TCN) with dilated convolutions"""
    
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size, 
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
                                   
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    """Building block for TCN"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove extra padding from TCN"""
    
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# Import weight_norm for TCN
try:
    from torch.nn.utils import weight_norm
except ImportError:
    def weight_norm(module, name='weight', dim=0):
        return module


class MobileNetBlock1D(nn.Module):
    """MobileNet-style block for 1D signals"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6):
        super().__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Pointwise expansion
        if expand_ratio != 1:
            layers.append(ConvBlock1D(in_channels, hidden_dim, 1))
            
        # Depthwise convolution
        layers.append(DepthwiseSeparableConv1D(hidden_dim, hidden_dim, kernel_size, stride))
        
        # Pointwise linear
        layers.append(nn.Conv1d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientBlock1D(nn.Module):
    """EfficientNet-style block for 1D signals"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 expand_ratio=1, se_ratio=0.25, dropout=0.2):
        super().__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = ConvBlock1D(in_channels, hidden_dim, 1, dropout=dropout)
        else:
            self.expand_conv = nn.Identity()
            
        # Depthwise convolution
        self.dw_conv = ConvBlock1D(hidden_dim, hidden_dim, kernel_size, stride, 
                                  groups=hidden_dim, dropout=dropout)
        
        # Squeeze and Excitation
        if se_ratio > 0:
            self.se_block = SEBlock1D(hidden_dim, int(in_channels * se_ratio))
        else:
            self.se_block = nn.Identity()
            
        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 and self.use_residual else nn.Identity()
        
    def forward(self, x):
        residual = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise
        x = self.dw_conv(x)
        
        # Squeeze and Excitation
        x = self.se_block(x)
        
        # Projection
        x = self.project_conv(x)
        
        # Residual connection
        if self.use_residual:
            x = self.dropout(x)
            x = x + residual
            
        return x
