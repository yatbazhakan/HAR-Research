"""
HAR Neural Network Modules

This package contains reusable neural network building blocks for Human Activity Recognition tasks.
"""

from .cnn_blocks import *
from .rnn_blocks import *
from .gnn_blocks import *
from .transformer_blocks import *
from .attention_modules import *
from .model_builders import *
from .losses import *

__all__ = [
    # CNN Blocks
    'ConvBlock1D', 'ResidualBlock1D', 'DepthwiseSeparableConv1D', 'TemporalConvNet',
    'InceptionBlock1D', 'SEBlock1D', 'DilatedConvBlock',
    
    # RNN Blocks
    'IndRNN', 'ConvLSTM1D', 'BiDirectionalLSTM', 'StackedLSTM', 'GRUCell',
    'LayerNormLSTM', 'PeepholeLSTM',
    
    # GNN Blocks
    'GraphConvLayer', 'GraphAttentionLayer', 'GraphSAGELayer', 'ChebConvLayer',
    'GATBlock', 'GCNBlock', 'GraphTransformerLayer',
    
    # Transformer Blocks
    'MultiHeadAttention', 'TransformerBlock', 'PositionalEncoding', 
    'FeedForward', 'TimeSeriesTransformer', 'ConvTransformerBlock',
    'TransformerDecoder', 'TransformerEncoderDecoder', 'TransformerWithConv',
    'TransformerWithPooling',
    
    # Attention Modules
    'SelfAttention', 'CrossAttention', 'SpatialAttention', 'ChannelAttention',
    'CBAM', 'ECA', 'CoordinateAttention', 'NonLocalAttention',
    'SqueezeExcitation', 'GatedAttention',
    
    # Model Builders
    'HARModelBuilder', 'create_har_model', 'create_model_from_config',
    'get_model_config', 'MODEL_CONFIGS',
    
    # Loss Functions
    'FocalLoss', 'WeightedFocalLoss', 'LabelSmoothingCrossEntropy',
    'get_loss_function'
]
