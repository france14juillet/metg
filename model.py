"""
METG: Memory-enhanced Transformer and Graph network for time series anomaly detection.

This implementation is based on the paper:
"Time Series Anomaly Detection Model Based on Memory-enhanced Transformer and Graph Network Joint Training"
by Qingqing Luo and Jiangang Dong (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class MemoryModule(nn.Module):
    """
    Memory enhancement module that captures and records diverse patterns
    between normal data and memory items.
    
    Args:
        memory_size (int): Number of memory items F
        memory_dim (int): Dimension of each memory item C
        temperature (float): Temperature parameter τ for attention
    """
    
    def __init__(self, memory_size: int, memory_dim: int, temperature: float = 1.0):
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.temperature = temperature
        
        # Initialize memory matrix M ∈ R^(F×C)
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of memory module.
        
        Args:
            query: Query tensor q_t ∈ R^(batch_size, seq_len, C)
            
        Returns:
            updated_query: Memory-enhanced query q̃_t
            attention_weights: Attention weights w
        """
        batch_size, seq_len, _ = query.shape
        
        # Reshape query for batch processing: (batch_size * seq_len, C)
        query_flat = query.view(-1, self.memory_dim)
        
        # Compute attention weights: w_i = exp(<m_i, q_t>/τ) / Σ(exp(<m_j, q_t>/τ))
        # query_flat: (batch_size * seq_len, C), memory: (F, C)
        similarities = torch.matmul(query_flat, self.memory.t()) / self.temperature  # (batch_size * seq_len, F)
        attention_weights = F.softmax(similarities, dim=1)  # (batch_size * seq_len, F)
        
        # Compute memory-enhanced query: q̃_t = w * M = Σ(w_i * m_i)
        updated_query_flat = torch.matmul(attention_weights, self.memory)  # (batch_size * seq_len, C)
        
        # Reshape back to original dimensions
        updated_query = updated_query_flat.view(batch_size, seq_len, self.memory_dim)
        attention_weights = attention_weights.view(batch_size, seq_len, self.memory_size)
        
        return updated_query, attention_weights


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal feature extraction.
    
    Args:
        d_model (int): Model dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dim_feedforward (int): Dimension of feedforward network
        dropout (float): Dropout rate
    """
    
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer encoder.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Encoded features (batch_size, seq_len, d_model)
        """
        x = self.pos_encoding(x)
        return self.transformer_encoder(x)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for reconstruction.
    
    Args:
        d_model (int): Model dimension
        output_dim (int): Output dimension (number of features)
    """
    
    def __init__(self, d_model: int, output_dim: int):
        super(TransformerDecoder, self).__init__()
        
        # Two fully connected layers for weak decoding as mentioned in paper
        self.fc1 = nn.Linear(d_model * 2, d_model)  # *2 because we concatenate original and memory-enhanced queries
        self.fc2 = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer decoder.
        
        Args:
            x: Updated query q̂_t (batch_size, seq_len, d_model * 2)
            
        Returns:
            Reconstructed output (batch_size, seq_len, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph Convolutional Network for spatial feature extraction.
    
    Args:
        input_dim (int): Input feature dimension (sequence length)
        hidden_dim (int): Hidden dimension
        output_dim (int): Output dimension (sequence length)
        k_neighbors (int): Number of nearest neighbors for graph construction
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, k_neighbors: int = 3):
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.input_dim = input_dim  # This will be seq_len when we reshape
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # This will be seq_len
        self.k_neighbors = k_neighbors
        
        # GCN layers - input_dim here represents the time dimension (seq_len)
        self.gc1 = GraphConvLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def construct_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Construct graph adjacency matrix using k-nearest neighbors.
        
        Args:
            x: Input tensor (batch_size, seq_len, num_features)
            
        Returns:
            Adjacency matrix (batch_size, num_features, num_features)
        """
        batch_size, seq_len, num_features = x.shape
        
        # Compute feature vectors for each variable (average over time)
        feature_vectors = x.mean(dim=1)  # (batch_size, num_features)
        
        # Create adjacency matrices for each batch
        adjacency_batch = []
        
        for b in range(batch_size):
            # Compute cosine similarity matrix for this batch
            batch_features = feature_vectors[b]  # (num_features,)
            normalized_features = F.normalize(batch_features.unsqueeze(0), p=2, dim=1)  # (1, num_features)
            
            # Compute pairwise cosine similarities
            similarity_matrix = torch.matmul(normalized_features.t(), normalized_features)  # (num_features, num_features)
            
            # Create adjacency matrix using top-k neighbors
            adjacency = torch.zeros_like(similarity_matrix)
            
            # Ensure k_neighbors doesn't exceed num_features - 1
            k = min(self.k_neighbors, num_features - 1)
            
            for i in range(num_features):
                if k > 0:
                    # Get top-k neighbors for node i
                    _, top_k_indices = torch.topk(similarity_matrix[i], k + 1)  # +1 to exclude self
                    top_k_indices = top_k_indices[1:]  # Remove self-connection
                    adjacency[i, top_k_indices] = 1
                    adjacency[top_k_indices, i] = 1  # Make symmetric
            
            # Add self-loops
            adjacency.fill_diagonal_(1)
            adjacency_batch.append(adjacency)
        
        return torch.stack(adjacency_batch, dim=0)  # (batch_size, num_features, num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN.
        
        Args:
            x: Input tensor (batch_size, seq_len, num_features)
            
        Returns:
            Reconstructed output (batch_size, seq_len, num_features)
        """
        batch_size, seq_len, num_features = x.shape
        
        # Construct adjacency matrix
        adj = self.construct_graph(x)  # (batch_size, num_features, num_features)
        
        # Reshape for GCN processing: treat each time step separately
        # We need to process the data so that nodes represent features and node features are time series
        x_reshaped = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len)
        
        # Apply GCN layers
        h = F.relu(self.gc1(x_reshaped, adj))  # Input: (batch_size, num_features, seq_len)
        h = self.dropout(h)
        h = self.gc2(h, adj)  # Output: (batch_size, num_features, seq_len)
        
        # Reshape back to original format
        output = h.permute(0, 2, 1)  # (batch_size, seq_len, num_features)
        
        return output


class GraphConvLayer(nn.Module):
    """Single graph convolutional layer."""
    
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        # Normalize adjacency matrix
        degree = adj.sum(dim=2, keepdim=True)  # (batch_size, num_nodes, 1)
        degree = torch.where(degree == 0, torch.ones_like(degree), degree)
        adj_normalized = adj / degree
        
        # Apply linear transformation
        x = self.linear(x)  # (batch_size, num_nodes, out_features)
        
        # Graph convolution: A * X * W
        output = torch.bmm(adj_normalized, x)  # (batch_size, num_nodes, out_features)
        
        return output


class METG(nn.Module):
    """
    METG: Memory-enhanced Transformer and Graph network for time series anomaly detection.
    
    Args:
        input_dim (int): Number of input features
        d_model (int): Transformer model dimension
        memory_size (int): Number of memory items
        seq_len (int): Sequence length
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        k_neighbors (int): Number of nearest neighbors for graph construction
        temperature (float): Temperature parameter for memory attention
    """
    
    def __init__(self, input_dim: int, d_model: int = 512, memory_size: int = 100, seq_len: int = 100,
                 nhead: int = 8, num_layers: int = 3, k_neighbors: int = 3, temperature: float = 1.0):
        super(METG, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.memory_size = memory_size
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Memory-enhanced Transformer components
        self.encoder = TransformerEncoder(d_model, nhead, num_layers)
        self.memory_module = MemoryModule(memory_size, d_model, temperature)
        self.decoder = TransformerDecoder(d_model, input_dim)
        
        # Graph Convolutional Network
        # GCN processes features as nodes, so input_dim is seq_len, output_dim is seq_len
        self.gcn = GraphConvolutionalNetwork(seq_len, d_model, seq_len, k_neighbors)
        
        # Loss function weights
        self.lambda1 = 1.0  # Weight for GCN reconstruction loss
        self.lambda2 = 0.01  # Weight for sparsity loss
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of METG model.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            transformer_output: Reconstructed output from transformer branch
            gcn_output: Reconstructed output from GCN branch
            attention_weights: Memory attention weights
        """
        # Transformer branch with memory enhancement
        # 1. Project input to model dimension
        x_proj = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 2. Encode with transformer
        encoded = self.encoder(x_proj)  # q_t
        
        # 3. Enhance with memory module
        memory_enhanced, attention_weights = self.memory_module(encoded)  # q̃_t, w
        
        # 4. Concatenate original and memory-enhanced queries
        updated_query = torch.cat([encoded, memory_enhanced], dim=-1)  # q̂_t
        
        # 5. Decode to reconstruction
        transformer_output = self.decoder(updated_query)
        
        # GCN branch for spatial modeling
        gcn_output = self.gcn(x)
        
        return transformer_output, gcn_output, attention_weights
    
    def compute_loss(self, x: torch.Tensor, transformer_output: torch.Tensor, 
                     gcn_output: torch.Tensor, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute joint optimization loss function.
        
        Args:
            x: Original input
            transformer_output: Transformer reconstruction
            gcn_output: GCN reconstruction
            attention_weights: Memory attention weights
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # L_MSE: Memory-enhanced Transformer reconstruction loss
        l_mse = F.mse_loss(transformer_output, x)
        
        # L_rec: GCN reconstruction loss
        l_rec = F.mse_loss(gcn_output, x)
        
        # L_spar: Sparsity loss for memory weights (entropy minimization)
        # Compute entropy: -Σ(w_i * log(w_i))
        epsilon = 1e-8  # For numerical stability
        log_weights = torch.log(attention_weights + epsilon)
        l_spar = -(attention_weights * log_weights).sum(dim=-1).mean()
        
        # Total loss: Loss = L_MSE + λ1*L_rec + λ2*L_spar
        total_loss = l_mse + self.lambda1 * l_rec + self.lambda2 * l_spar
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'l_mse': l_mse.item(),
            'l_rec': l_rec.item(),
            'l_spar': l_spar.item()
        }
        
        return total_loss, loss_dict
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for input sequences.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Anomaly scores (batch_size, seq_len)
        """
        with torch.no_grad():
            transformer_output, gcn_output, attention_weights = self.forward(x)
            
            # Compute reconstruction errors
            l_mse = F.mse_loss(transformer_output, x, reduction='none').mean(dim=-1)  # (batch_size, seq_len)
            l_rec = F.mse_loss(gcn_output, x, reduction='none').mean(dim=-1)  # (batch_size, seq_len)
            
            # Compute memory importance score: m_score = softmax(||q̃_t - m_i||²)
            # For simplicity, we use the maximum attention weight as importance
            m_score = attention_weights.max(dim=-1)[0]  # (batch_size, seq_len)
            
            # Final anomaly score: Score = m_score * (L_MSE + L_rec)
            anomaly_scores = m_score * (l_mse + l_rec)
            
        return anomaly_scores
    
    def detect_anomalies(self, x: torch.Tensor, threshold: Optional[float] = None, 
                        threshold_percentile: Optional[float] = None,
                        aggregation_method: str = 'max',
                        min_anomaly_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect anomalies in input sequences with configurable thresholds and aggregation.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            threshold: Fixed anomaly threshold (overrides threshold_percentile if provided)
            threshold_percentile: Percentile for threshold (default: None, uses 99th percentile)
            aggregation_method: Method to aggregate time-step scores to sequence level
                               Options: 'max', 'mean', 'min_count'
            min_anomaly_ratio: For 'min_count' method, minimum ratio of anomalous time steps
            
        Returns:
            timestep_scores: Time-step level anomaly scores (batch_size, seq_len)
            sequence_scores: Sequence-level anomaly scores (batch_size,)
            sequence_labels: Binary sequence labels (1 for anomaly, 0 for normal) (batch_size,)
        """
        # Compute time-step level anomaly scores
        timestep_scores = self.compute_anomaly_score(x)  # (batch_size, seq_len)
        
        # Aggregate to sequence level based on method
        if aggregation_method == 'max':
            sequence_scores = timestep_scores.max(dim=1)[0]  # (batch_size,)
        elif aggregation_method == 'mean':
            sequence_scores = timestep_scores.mean(dim=1)  # (batch_size,)
        elif aggregation_method == 'min_count':
            # Determine threshold for time-step level anomalies
            if threshold is not None:
                ts_threshold = threshold
            elif threshold_percentile is not None:
                ts_threshold = torch.quantile(timestep_scores, threshold_percentile / 100.0)
            else:
                ts_threshold = torch.quantile(timestep_scores, 0.99)
            
            # Count anomalous time steps per sequence
            timestep_labels = (timestep_scores > ts_threshold).float()  # (batch_size, seq_len)
            anomaly_ratios = timestep_labels.mean(dim=1)  # (batch_size,)
            
            # Sequence is anomalous if >= min_anomaly_ratio of time steps are anomalous
            sequence_labels = (anomaly_ratios >= min_anomaly_ratio).float()
            return timestep_scores, anomaly_ratios, sequence_labels
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Determine threshold for sequence-level scores
        if threshold is not None:
            seq_threshold = threshold
        elif threshold_percentile is not None:
            seq_threshold = torch.quantile(sequence_scores, threshold_percentile / 100.0)
        else:
            # Use 99th percentile as default (more conservative than 95th)
            seq_threshold = torch.quantile(sequence_scores, 0.99)
        
        sequence_labels = (sequence_scores > seq_threshold).float()
        
        return timestep_scores, sequence_scores, sequence_labels


def create_metg_model(input_dim: int, **kwargs) -> METG:
    """
    Factory function to create METG model with default parameters.
    
    Args:
        input_dim: Number of input features
        **kwargs: Additional model parameters
        
    Returns:
        Initialized METG model
    """
    default_params = {
        'd_model': 512,
        'memory_size': 100,
        'seq_len': 100,
        'nhead': 8,
        'num_layers': 3,
        'k_neighbors': 3,
        'temperature': 1.0
    }
    
    # Update defaults with provided kwargs
    default_params.update(kwargs)
    
    return METG(input_dim=input_dim, **default_params)


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    seq_len = 100
    input_dim = 25
    
    # Create model
    model = create_metg_model(input_dim=input_dim, seq_len=seq_len)
    
    # Generate random input data
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    transformer_out, gcn_out, attention_weights = model(x)
    
    # Compute loss
    total_loss, loss_dict = model.compute_loss(x, transformer_out, gcn_out, attention_weights)
    
    # Compute anomaly scores
    timestep_scores, sequence_scores, sequence_labels = model.detect_anomalies(x)
    
    print(f"Model input shape: {x.shape}")
    print(f"Transformer output shape: {transformer_out.shape}")
    print(f"GCN output shape: {gcn_out.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    print(f"Timestep scores shape: {timestep_scores.shape}")
    print(f"Sequence scores shape: {sequence_scores.shape}")
    print(f"Number of detected anomalies: {sequence_labels.sum().item()}")