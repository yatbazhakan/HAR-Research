"""
Example usage of HAR modules for building different model architectures

This script demonstrates how to use the modular components to build
various HAR models using existing PyTorch transformer architectures.
"""

import torch
import torch.nn as nn
from har.modules import *


def example_basic_usage():
    """Basic usage examples of individual modules"""
    print("=== Basic Module Usage Examples ===")
    
    # Example 1: CNN-based feature extractor
    print("\n1. CNN Feature Extractor:")
    cnn_model = nn.Sequential(
        ConvBlock1D(6, 64, kernel_size=5, dropout=0.2),
        ResidualBlock1D(64, 128, dropout=0.2),
        SEBlock1D(128),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(128, 6)  # 6 classes
    )
    
    x = torch.randn(32, 128, 6)  # (batch, seq_len, features)
    x = x.transpose(1, 2)  # (batch, features, seq_len) for conv1d
    output = cnn_model(x)
    print(f"CNN output shape: {output.shape}")
    
    # Example 2: LSTM with attention
    print("\n2. LSTM with Attention:")
    lstm_model = nn.Sequential(
        nn.Linear(6, 128),
        LayerNormLSTM(128, 64),
        TemporalAttention(64),
        nn.Linear(64, 6)
    )
    
    x = torch.randn(32, 128, 6)  # (batch, seq_len, features)
    output, attn_weights = lstm_model(x)
    print(f"LSTM output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Example 3: Transformer using PyTorch's components
    print("\n3. Transformer Model:")
    transformer = TimeSeriesTransformer(
        input_dim=6,
        d_model=256,
        num_heads=8,
        num_layers=6,
        max_seq_len=128,
        dropout=0.1,
        output_dim=6
    )
    
    x = torch.randn(32, 128, 6)  # (batch, seq_len, features)
    output = transformer(x)
    print(f"Transformer output shape: {output.shape}")


def example_model_builders():
    """Examples using the high-level model builders"""
    print("\n=== Model Builder Examples ===")
    
    # Create a model builder
    builder = HARModelBuilder(input_dim=6, num_classes=6, sequence_length=128)
    
    # Example 1: CNN model
    print("\n1. CNN Model:")
    cnn_model = builder.build_cnn_model(
        conv_channels=[64, 128, 256],
        use_residual=True,
        use_attention=True
    )
    
    x = torch.randn(32, 128, 6)
    x = x.transpose(1, 2)  # CNN expects (batch, channels, seq_len)
    output = cnn_model(x)
    print(f"CNN model output shape: {output.shape}")
    
    # Example 2: LSTM model
    print("\n2. LSTM Model:")
    lstm_model = builder.build_lstm_model(
        hidden_sizes=[128, 64],
        lstm_types=['layer_norm', 'peephole'],
        bidirectional=True,
        use_attention=True
    )
    
    x = torch.randn(32, 128, 6)
    output = lstm_model(x)
    print(f"LSTM model output shape: {output.shape}")
    
    # Example 3: Transformer model
    print("\n3. Transformer Model:")
    transformer_model = builder.build_transformer_model(
        d_model=256,
        num_heads=8,
        num_layers=6,
        use_conv=True,  # Conv-Transformer hybrid
        pooling_type='attention'
    )
    
    x = torch.randn(32, 128, 6)
    output = transformer_model(x)
    print(f"Transformer model output shape: {output.shape}")


def example_predefined_configs():
    """Examples using predefined model configurations"""
    print("\n=== Predefined Configuration Examples ===")
    
    # List available configurations
    print(f"Available model configs: {list(MODEL_CONFIGS.keys())}")
    
    # Example 1: Create model from predefined config
    print("\n1. CNN Small Model:")
    cnn_small = create_model_from_config(
        config_name='cnn_small',
        input_dim=6,
        num_classes=6,
        sequence_length=128
    )
    
    x = torch.randn(32, 128, 6)
    x = x.transpose(1, 2)  # CNN expects (batch, channels, seq_len)
    output = cnn_small(x)
    print(f"CNN small output shape: {output.shape}")
    
    # Example 2: Transformer Large Model
    print("\n2. Transformer Large Model:")
    transformer_large = create_model_from_config(
        config_name='transformer_large',
        input_dim=6,
        num_classes=6,
        sequence_length=128
    )
    
    x = torch.randn(32, 128, 6)
    output = transformer_large(x)
    print(f"Transformer large output shape: {output.shape}")
    
    # Example 3: Hybrid Model
    print("\n3. Hybrid Model:")
    hybrid_model = create_model_from_config(
        config_name='hybrid_balanced',
        input_dim=6,
        num_classes=6,
        sequence_length=128
    )
    
    x = torch.randn(32, 128, 6)
    output = hybrid_model(x)
    print(f"Hybrid model output shape: {output.shape}")


def example_attention_modules():
    """Examples of different attention mechanisms"""
    print("\n=== Attention Module Examples ===")
    
    x = torch.randn(32, 128, 6)  # (batch, seq_len, features)
    
    # Example 1: Self-Attention
    print("\n1. Self-Attention:")
    self_attn = SelfAttention(input_dim=6, hidden_dim=64)
    output, attn_weights = self_attn(x)
    print(f"Self-attention output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Example 2: Channel Attention (for multi-channel signals)
    print("\n2. Channel Attention:")
    x_conv = x.transpose(1, 2)  # (batch, channels, seq_len)
    channel_attn = ChannelAttention(num_channels=6, reduction=2)
    output, channel_weights = channel_attn(x_conv)
    print(f"Channel attention output shape: {output.shape}")
    print(f"Channel weights shape: {channel_weights.shape}")
    
    # Example 3: CBAM (Channel + Spatial Attention)
    print("\n3. CBAM:")
    cbam = CBAM(num_channels=6, reduction=4)
    output, (channel_weights, spatial_weights) = cbam(x_conv)
    print(f"CBAM output shape: {output.shape}")
    print(f"Channel weights shape: {channel_weights.shape}")
    print(f"Spatial weights shape: {spatial_weights.shape}")


def example_graph_models():
    """Examples of Graph Neural Network models"""
    print("\n=== Graph Model Examples ===")
    
    # Create a simple adjacency matrix (sensors are connected)
    num_sensors = 6
    adj_matrix = torch.ones(num_sensors, num_sensors)  # Fully connected
    adj_matrix = adj_matrix.unsqueeze(0).repeat(32, 1, 1)  # (batch, sensors, sensors)
    
    x = torch.randn(32, num_sensors, 6)  # (batch, sensors, features)
    
    # Example 1: Graph Convolution
    print("\n1. Graph Convolution:")
    gcn_layer = GraphConvLayer(in_features=6, out_features=64)
    output = gcn_layer(x, adj_matrix)
    print(f"GCN output shape: {output.shape}")
    
    # Example 2: Graph Attention
    print("\n2. Graph Attention:")
    gat_layer = GraphAttentionLayer(in_features=6, out_features=64, num_heads=4)
    output = gat_layer(x, adj_matrix)
    print(f"GAT output shape: {output.shape}")
    
    # Example 3: Complete Graph Model
    print("\n3. Complete Graph Model:")
    graph_model = create_har_model(
        architecture='graph',
        input_dim=6,
        num_classes=6,
        sequence_length=128,
        num_sensors=num_sensors,
        graph_type='gat',
        hidden_dims=[64, 128],
        num_heads=4
    )
    
    # For graph models, we need to reshape input
    x_graph = x.mean(dim=1)  # Average over time for graph input
    output = graph_model(x_graph, adj_matrix[0])  # Use single adjacency matrix
    print(f"Graph model output shape: {output.shape}")


if __name__ == "__main__":
    print("HAR Modules Usage Examples")
    print("=" * 50)
    
    # Run all examples
    example_basic_usage()
    example_model_builders()
    example_predefined_configs()
    example_attention_modules()
    example_graph_models()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nKey Benefits of the Modular Approach:")
    print("1. ✅ Leverages existing PyTorch transformer architectures")
    print("2. ✅ Easy to mix and match different components")
    print("3. ✅ Predefined configurations for quick experimentation")
    print("4. ✅ High-level builders for complex architectures")
    print("5. ✅ Comprehensive attention mechanisms")
    print("6. ✅ Graph neural networks for sensor relationships")
    print("7. ✅ Production-ready and well-documented")
