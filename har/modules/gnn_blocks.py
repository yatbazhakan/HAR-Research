"""
Graph Neural Network Building Blocks for HAR Tasks

This module contains various GNN architectures for modeling sensor relationships
and spatial-temporal dependencies in human activity recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvLayer(nn.Module):
    """
    Basic Graph Convolution Layer (GCN)
    
    Implements the graph convolution operation: H' = σ(D^(-1/2) A D^(-1/2) H W)
    """
    
    def __init__(self, in_features, out_features, bias=True, dropout=0.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        """
        Forward pass
        
        Args:
            input: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
        """
        support = torch.matmul(input, self.weight)
        
        if adj.dim() == 2:
            # Same adjacency for all batches
            output = torch.matmul(adj, support)
        else:
            # Different adjacency per batch
            output = torch.bmm(adj, support)
            
        if self.bias is not None:
            output += self.bias
            
        return self.dropout(output)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) Layer
    
    Implements attention-based graph convolution with learnable attention weights.
    """
    
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, h, adj):
        """
        Args:
            h: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (batch_size, num_nodes, out_features)
        
        # Attention mechanism
        e = self._prepare_attentional_mechanism_input(Wh)
        
        # Mask non-existing edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Apply attention weights
        h_prime = torch.bmm(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size, N, out_features = Wh.shape
        
        # Repeat for all pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # (batch, N*N, out_features)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)  # (batch, N*N, out_features)
        
        # Concatenate
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        # (batch, N*N, 2*out_features)
        
        # Apply attention function
        e = self.leakyrelu(torch.matmul(all_combinations_matrix, self.a)).squeeze(-1)
        # (batch, N*N)
        
        return e.view(batch_size, N, N)


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Layer
    
    Implements inductive graph learning by sampling and aggregating from node neighborhoods.
    """
    
    def __init__(self, in_features, out_features, aggregator='mean', dropout=0.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        if aggregator == 'mean':
            self.aggr_fn = self.mean_aggregator
        elif aggregator == 'max':
            self.aggr_fn = self.max_aggregator
        elif aggregator == 'lstm':
            self.aggr_fn = self.lstm_aggregator
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
            
        # Linear layers
        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features)
        
        self.dropout = nn.Dropout(dropout)
        
    def mean_aggregator(self, neighbor_features):
        return torch.mean(neighbor_features, dim=-2)
        
    def max_aggregator(self, neighbor_features):
        return torch.max(neighbor_features, dim=-2)[0]
        
    def lstm_aggregator(self, neighbor_features):
        # Flatten batch and node dimensions for LSTM
        batch_size, num_nodes, num_neighbors, features = neighbor_features.shape
        neighbor_features = neighbor_features.view(batch_size * num_nodes, num_neighbors, features)
        
        _, (h_n, _) = self.lstm(neighbor_features)
        return h_n.squeeze(0).view(batch_size, num_nodes, features)
        
    def forward(self, h, adj):
        """
        Args:
            h: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = h.shape
        
        # Sample neighbors (here we use all neighbors for simplicity)
        # In practice, you would sample a fixed number of neighbors
        neighbor_features = []
        
        for i in range(num_nodes):
            # Get neighbors for node i
            neighbors = adj[:, i, :].unsqueeze(-1)  # (batch_size, num_nodes, 1)
            neighbor_feat = h * neighbors  # (batch_size, num_nodes, in_features)
            neighbor_features.append(neighbor_feat.unsqueeze(1))
            
        neighbor_features = torch.cat(neighbor_features, dim=1)  # (batch_size, num_nodes, num_nodes, in_features)
        
        # Aggregate neighbors
        aggregated = self.aggr_fn(neighbor_features)  # (batch_size, num_nodes, in_features)
        
        # Combine self and neighbor information
        self_feat = self.self_linear(h)
        neighbor_feat = self.neighbor_linear(aggregated)
        
        output = self_feat + neighbor_feat
        output = F.relu(output)
        output = F.normalize(output, p=2, dim=-1)  # L2 normalization
        
        return self.dropout(output)


class ChebConvLayer(nn.Module):
    """
    Chebyshev Graph Convolution Layer
    
    Uses Chebyshev polynomials for efficient spectral graph convolution.
    """
    
    def __init__(self, in_features, out_features, K=3, bias=True):
        super().__init__()
        
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x, L):
        """
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            L: Normalized Laplacian matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.shape
        
        # Compute Chebyshev polynomials
        Tx_0 = x  # T_0(L) = I
        Tx_1 = torch.bmm(L, x)  # T_1(L) = L
        
        # Apply first two polynomials
        out = torch.matmul(Tx_0, self.weight[0])
        if self.K > 1:
            out += torch.matmul(Tx_1, self.weight[1])
            
        # Compute higher order polynomials recursively
        for k in range(2, self.K):
            Tx_2 = 2 * torch.bmm(L, Tx_1) - Tx_0  # T_k(L) = 2L*T_{k-1}(L) - T_{k-2}(L)
            out += torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
            
        if self.bias is not None:
            out += self.bias
            
        return out


class GATBlock(nn.Module):
    """Multi-head Graph Attention Block"""
    
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.0, alpha=0.2):
        super().__init__()
        
        self.num_heads = num_heads
        self.out_features = out_features
        
        # Multi-head attention layers
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features // num_heads, dropout, alpha, concat=True)
            for _ in range(num_heads)
        ])
        
        # Output attention layer
        self.out_att = GraphAttentionLayer(out_features, out_features, dropout, alpha, concat=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        # Multi-head attention
        x = self.dropout(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        
        # Output layer
        x = self.dropout(x)
        x = F.elu(self.out_att(x, adj))
        
        return x


class GCNBlock(nn.Module):
    """Graph Convolution Block with residual connections"""
    
    def __init__(self, in_features, out_features, num_layers=2, dropout=0.0, residual=True):
        super().__init__()
        
        self.num_layers = num_layers
        self.residual = residual and (in_features == out_features)
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_in = in_features if i == 0 else out_features
            self.layers.append(GraphConvLayer(layer_in, out_features, dropout=dropout))
            
        if not self.residual and in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = None
            
    def forward(self, x, adj):
        residual = x
        
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:  # Apply activation for all but last layer
                x = F.relu(x)
                
        if self.residual:
            x = x + residual
        elif self.projection is not None:
            x = x + self.projection(residual)
            
        return F.relu(x)


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer
    
    Combines graph structure with transformer-style attention.
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_mask=None):
        """
        Args:
            x: Node features (batch_size, num_nodes, d_model)
            adj_mask: Attention mask based on adjacency (batch_size, num_nodes, num_nodes)
        """
        # Multi-head self-attention with graph structure
        residual = x
        attn_output, _ = self.attention(x, x, x, attn_mask=adj_mask)
        x = self.norm1(residual + self.dropout(attn_output))
        
        # Feed forward
        residual = x
        ff_output = self.feed_forward(x)
        x = self.norm2(residual + self.dropout(ff_output))
        
        return x


class EdgeConvLayer(nn.Module):
    """
    Edge Convolution Layer for dynamic graph construction
    
    Constructs edges dynamically based on feature similarity.
    """
    
    def __init__(self, in_features, out_features, k=20, aggr='max'):
        super().__init__()
        
        self.k = k
        self.aggr = aggr
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
    def knn_graph(self, x):
        """Construct k-NN graph based on feature similarity"""
        # Compute pairwise distances
        inner = -2 * torch.matmul(x, x.transpose(-2, -1))
        xx = torch.sum(x**2, dim=-1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(-2, -1)
        
        # Get k nearest neighbors
        _, idx = pairwise_distance.topk(k=self.k, dim=-1)
        return idx
        
    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, in_features)
        """
        batch_size, num_nodes, in_features = x.shape
        
        # Construct k-NN graph
        idx = self.knn_graph(x)  # (batch_size, num_nodes, k)
        
        # Gather neighbor features
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, in_features)
        neighbors = torch.gather(x.unsqueeze(-2).expand(-1, -1, self.k, -1), 2, idx_expanded)
        
        # Prepare edge features
        central = x.unsqueeze(-2).expand(-1, -1, self.k, -1)
        edge_features = torch.cat([central, neighbors - central], dim=-1)
        
        # Apply MLP to edge features
        edge_features = self.mlp(edge_features)  # (batch_size, num_nodes, k, out_features)
        
        # Aggregate
        if self.aggr == 'max':
            x_new = torch.max(edge_features, dim=-2)[0]
        elif self.aggr == 'mean':
            x_new = torch.mean(edge_features, dim=-2)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggr}")
            
        return x_new


class GraphPooling(nn.Module):
    """Graph pooling operations for graph-level predictions"""
    
    def __init__(self, pooling_type='mean'):
        super().__init__()
        self.pooling_type = pooling_type
        
    def forward(self, x, batch=None):
        """
        Args:
            x: Node features (batch_size, num_nodes, features) or (num_total_nodes, features)
            batch: Batch assignment for each node (only needed for batched graphs)
        """
        if batch is None:
            # Single graph or already batched
            if self.pooling_type == 'mean':
                return torch.mean(x, dim=-2)
            elif self.pooling_type == 'max':
                return torch.max(x, dim=-2)[0]
            elif self.pooling_type == 'sum':
                return torch.sum(x, dim=-2)
            elif self.pooling_type == 'attention':
                # Simple attention pooling
                attention_weights = F.softmax(torch.sum(x, dim=-1), dim=-1)
                return torch.sum(x * attention_weights.unsqueeze(-1), dim=-2)
        else:
            # Batched graphs with different sizes
            # This would require more complex implementation
            raise NotImplementedError("Batched graphs with variable sizes not implemented")
            
        return x
