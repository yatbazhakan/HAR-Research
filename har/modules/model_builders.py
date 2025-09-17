"""
Model Builder Utilities for HAR Tasks

This module provides high-level builders for creating complete HAR models
using the modular components from the modules package.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Any

from .cnn_blocks import *
from .rnn_blocks import *
from .gnn_blocks import *
from .transformer_blocks import *
from .attention_modules import *


class HARModelBuilder:
    """
    High-level builder for creating HAR models with different architectures
    """
    
    def __init__(self, input_dim: int, num_classes: int, sequence_length: int):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
    def build_cnn_model(self, 
                       conv_channels: List[int] = [64, 128, 256],
                       kernel_sizes: List[int] = [3, 3, 3],
                       dropout: float = 0.2,
                       use_residual: bool = True,
                       use_attention: bool = True) -> nn.Module:
        """
        Build a CNN-based HAR model
        
        Args:
            conv_channels: List of output channels for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout: Dropout rate
            use_residual: Whether to use residual connections
            use_attention: Whether to add attention mechanisms
        """
        layers = []
        in_channels = self.input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            if use_residual and i > 0:
                layers.append(ResidualBlock1D(in_channels, out_channels, kernel_size, dropout=dropout))
            else:
                layers.append(ConvBlock1D(in_channels, out_channels, kernel_size, dropout=dropout))
            
            if use_attention:
                layers.append(SEBlock1D(out_channels))
                
            in_channels = out_channels
            
        # Global pooling and classification
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels[-1], self.num_classes)
        ])
        
        return nn.Sequential(*layers)
    
    def build_lstm_model(self,
                        hidden_sizes: List[int] = [128, 64],
                        lstm_types: List[str] = ['standard', 'layer_norm'],
                        dropout: float = 0.2,
                        bidirectional: bool = True,
                        use_attention: bool = True) -> nn.Module:
        """
        Build an LSTM-based HAR model
        
        Args:
            hidden_sizes: List of hidden sizes for each LSTM layer
            lstm_types: List of LSTM types ('standard', 'layer_norm', 'peephole', 'indRNN')
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to add temporal attention
        """
        layers = []
        
        # Input projection
        layers.append(nn.Linear(self.input_dim, hidden_sizes[0]))
        
        # LSTM layers
        if bidirectional:
            for i, (hidden_size, lstm_type) in enumerate(zip(hidden_sizes, lstm_types)):
                if lstm_type == 'standard':
                    lstm = nn.LSTM(hidden_sizes[i-1] if i > 0 else hidden_sizes[0], 
                                 hidden_size, batch_first=True, bidirectional=True)
                elif lstm_type == 'layer_norm':
                    lstm = LayerNormLSTM(hidden_sizes[i-1] if i > 0 else hidden_sizes[0], 
                                       hidden_size)
                elif lstm_type == 'peephole':
                    lstm = PeepholeLSTM(hidden_sizes[i-1] if i > 0 else hidden_sizes[0], 
                                      hidden_size)
                elif lstm_type == 'indRNN':
                    lstm = IndRNN(hidden_sizes[i-1] if i > 0 else hidden_sizes[0], 
                                hidden_size)
                else:
                    raise ValueError(f"Unknown LSTM type: {lstm_type}")
                    
                layers.append(lstm)
                layers.append(nn.Dropout(dropout))
        else:
            stacked_lstm = StackedLSTM(
                self.input_dim, hidden_sizes, lstm_types, [dropout] * len(hidden_sizes)
            )
            layers.append(stacked_lstm)
        
        # Attention mechanism
        if use_attention:
            layers.append(TemporalAttention(hidden_sizes[-1] * (2 if bidirectional else 1)))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_sizes[-1] * (2 if bidirectional else 1), self.num_classes))
        else:
            layers.extend([
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(hidden_sizes[-1] * (2 if bidirectional else 1), self.num_classes)
            ])
        
        return nn.Sequential(*layers)
    
    def build_transformer_model(self,
                               d_model: int = 256,
                               num_heads: int = 8,
                               num_layers: int = 6,
                               dropout: float = 0.1,
                               use_conv: bool = False,
                               pooling_type: str = 'mean') -> nn.Module:
        """
        Build a Transformer-based HAR model
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            use_conv: Whether to use Conv-Transformer hybrid
            pooling_type: Type of pooling ('mean', 'max', 'attention')
        """
        if use_conv:
            return TransformerWithConv(
                input_dim=self.input_dim,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=self.sequence_length,
                dropout=dropout,
                output_dim=self.num_classes
            )
        else:
            return TransformerWithPooling(
                input_dim=self.input_dim,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=self.sequence_length,
                dropout=dropout,
                output_dim=self.num_classes,
                pooling_type=pooling_type
            )
    
    def build_hybrid_model(self,
                          conv_channels: List[int] = [64, 128],
                          lstm_hidden: int = 128,
                          transformer_layers: int = 4,
                          d_model: int = 256,
                          num_heads: int = 8,
                          dropout: float = 0.2) -> nn.Module:
        """
        Build a hybrid CNN-LSTM-Transformer model
        
        Args:
            conv_channels: CNN output channels
            lstm_hidden: LSTM hidden size
            transformer_layers: Number of transformer layers
            d_model: Transformer model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        class HybridHARModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # CNN feature extraction
                self.cnn_layers = nn.ModuleList()
                in_channels = self.input_dim
                
                for out_channels in conv_channels:
                    self.cnn_layers.append(ConvBlock1D(in_channels, out_channels, dropout=dropout))
                    in_channels = out_channels
                
                # LSTM for temporal modeling
                self.lstm = nn.LSTM(conv_channels[-1], lstm_hidden, 
                                  batch_first=True, bidirectional=True)
                
                # Transformer for global attention
                self.transformer = TimeSeriesTransformer(
                    input_dim=lstm_hidden * 2,
                    d_model=d_model,
                    num_heads=num_heads,
                    num_layers=transformer_layers,
                    max_seq_len=self.sequence_length,
                    dropout=dropout
                )
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, self.num_classes)
                )
                
            def forward(self, x):
                # CNN feature extraction
                x = x.transpose(1, 2)  # (batch, channels, seq_len)
                for cnn_layer in self.cnn_layers:
                    x = cnn_layer(x)
                x = x.transpose(1, 2)  # (batch, seq_len, channels)
                
                # LSTM temporal modeling
                x, _ = self.lstm(x)
                
                # Transformer global attention
                x = self.transformer(x)
                
                # Classification
                x = x.transpose(1, 2)  # (batch, d_model, seq_len)
                output = self.classifier(x)
                
                return output
        
        return HybridHARModel()
    
    def build_graph_model(self,
                         num_sensors: int,
                         graph_type: str = 'gcn',
                         hidden_dims: List[int] = [64, 128],
                         num_heads: int = 8,
                         dropout: float = 0.2) -> nn.Module:
        """
        Build a Graph Neural Network model for sensor-based HAR
        
        Args:
            num_sensors: Number of sensors (nodes in graph)
            graph_type: Type of GNN ('gcn', 'gat', 'sage')
            hidden_dims: List of hidden dimensions
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout rate
        """
        class GraphHARModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Input projection
                self.input_proj = nn.Linear(self.input_dim, hidden_dims[0])
                
                # Graph layers
                self.graph_layers = nn.ModuleList()
                in_dim = hidden_dims[0]
                
                for out_dim in hidden_dims[1:]:
                    if graph_type == 'gcn':
                        self.graph_layers.append(GCNBlock(in_dim, out_dim, dropout=dropout))
                    elif graph_type == 'gat':
                        self.graph_layers.append(GATBlock(in_dim, out_dim, num_heads, dropout))
                    elif graph_type == 'sage':
                        self.graph_layers.append(GraphSAGELayer(in_dim, out_dim, dropout=dropout))
                    else:
                        raise ValueError(f"Unknown graph type: {graph_type}")
                    in_dim = out_dim
                
                # Global pooling and classification
                self.pooling = GraphPooling('attention')
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dims[-1], self.num_classes)
                )
                
            def forward(self, x, adj):
                # x: (batch, num_sensors, input_dim)
                # adj: (batch, num_sensors, num_sensors) or (num_sensors, num_sensors)
                
                # Input projection
                x = self.input_proj(x)
                
                # Graph convolutions
                for graph_layer in self.graph_layers:
                    x = graph_layer(x, adj)
                
                # Global pooling
                x = self.pooling(x)  # (batch, hidden_dim)
                
                # Classification
                output = self.classifier(x)
                
                return output
        
        return GraphHARModel()


def create_har_model(architecture: str, 
                    input_dim: int, 
                    num_classes: int, 
                    sequence_length: int,
                    **kwargs) -> nn.Module:
    """
    Factory function to create HAR models
    
    Args:
        architecture: Model architecture ('cnn', 'lstm', 'transformer', 'hybrid', 'graph')
        input_dim: Input feature dimension
        num_classes: Number of output classes
        sequence_length: Input sequence length
        **kwargs: Additional model-specific parameters
    
    Returns:
        PyTorch model
    """
    builder = HARModelBuilder(input_dim, num_classes, sequence_length)
    
    if architecture == 'cnn':
        return builder.build_cnn_model(**kwargs)
    elif architecture == 'lstm':
        return builder.build_lstm_model(**kwargs)
    elif architecture == 'transformer':
        return builder.build_transformer_model(**kwargs)
    elif architecture == 'hybrid':
        return builder.build_hybrid_model(**kwargs)
    elif architecture == 'graph':
        return builder.build_graph_model(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Predefined model configurations
MODEL_CONFIGS = {
    'cnn_small': {
        'architecture': 'cnn',
        'conv_channels': [32, 64, 128],
        'kernel_sizes': [3, 3, 3],
        'dropout': 0.2
    },
    'cnn_large': {
        'architecture': 'cnn',
        'conv_channels': [64, 128, 256, 512],
        'kernel_sizes': [5, 3, 3, 3],
        'dropout': 0.3,
        'use_residual': True,
        'use_attention': True
    },
    'lstm_simple': {
        'architecture': 'lstm',
        'hidden_sizes': [128, 64],
        'lstm_types': ['standard', 'standard'],
        'dropout': 0.2,
        'bidirectional': True
    },
    'lstm_advanced': {
        'architecture': 'lstm',
        'hidden_sizes': [256, 128, 64],
        'lstm_types': ['layer_norm', 'peephole', 'indRNN'],
        'dropout': 0.3,
        'bidirectional': True,
        'use_attention': True
    },
    'transformer_small': {
        'architecture': 'transformer',
        'd_model': 128,
        'num_heads': 4,
        'num_layers': 4,
        'dropout': 0.1
    },
    'transformer_large': {
        'architecture': 'transformer',
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 12,
        'dropout': 0.1,
        'use_conv': True
    },
    'hybrid_balanced': {
        'architecture': 'hybrid',
        'conv_channels': [64, 128],
        'lstm_hidden': 128,
        'transformer_layers': 4,
        'd_model': 256,
        'num_heads': 8,
        'dropout': 0.2
    }
}


def get_model_config(config_name: str) -> Dict[str, Any]:
    """Get predefined model configuration"""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[config_name]


def create_model_from_config(config_name: str, 
                           input_dim: int, 
                           num_classes: int, 
                           sequence_length: int) -> nn.Module:
    """Create model from predefined configuration"""
    config = get_model_config(config_name)
    return create_har_model(
        architecture=config['architecture'],
        input_dim=input_dim,
        num_classes=num_classes,
        sequence_length=sequence_length,
        **{k: v for k, v in config.items() if k != 'architecture'}
    )
