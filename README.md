# METG: Memory-enhanced Transformer and Graph Network

This repository contains a Python implementation of the METG (Memory-enhanced Transformer and Graph network) model for multivariate time series anomaly detection, based on the paper:

**"Time Series Anomaly Detection Model Based on Memory-enhanced Transformer and Graph Network Joint Training"**  
*by Qingqing Luo and Jiangang Dong (2024)*

## Overview

METG is an unsupervised anomaly detection model that addresses two key challenges in multivariate time series anomaly detection:

1. **Limited generalization ability** due to scarcity of normal training data
2. **Complex dependencies** among variables in multivariate time series

The model combines:
- **Memory-enhanced Transformer**: Captures temporal dependencies and stores diverse normal patterns
- **Graph Convolutional Network (GCN)**: Models spatial relationships between variables
- **Joint optimization**: Combines strengths of both components for robust anomaly detection

## Architecture

### Key Components

1. **Memory Enhancement Module**
   - Stores F memory items representing different normal patterns
   - Uses attention mechanism to enhance queries with relevant memory patterns
   - Prevents overfitting to limited training data

2. **Transformer Encoder-Decoder**
   - Encodes temporal dependencies in input sequences
   - Decoder reconstructs input using memory-enhanced features
   - Uses positional encoding for sequence modeling

3. **Graph Convolutional Network**
   - Constructs graphs using k-nearest neighbors based on feature similarity
   - Captures spatial dependencies between different time series variables
   - Uses cosine similarity for edge construction

4. **Joint Training Objective**
   ```
   Loss = L_MSE + λ₁ × L_rec + λ₂ × L_spar
   ```
   - `L_MSE`: Transformer reconstruction loss
   - `L_rec`: GCN reconstruction loss  
   - `L_spar`: Sparsity loss for memory weights

## Installation

1. Clone the repository:
```bash
git clone https://github.com/france14juillet/metg.git
cd metg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch 1.12.0+
- torch-geometric 2.1.0+
- NumPy 1.21.0+
- scikit-learn 1.0.0+
- matplotlib 3.5.0+
- pandas 1.3.0+

## Quick Start

### Basic Usage

```python
import torch
from model import create_metg_model

# Create model for 25-dimensional time series
model = create_metg_model(input_dim=25, seq_len=100, d_model=512)

# Generate sample data (batch_size=32, seq_len=100, features=25)
x = torch.randn(32, 100, 25)

# Forward pass
transformer_out, gcn_out, attention_weights = model(x)

# Compute loss for training
total_loss, loss_dict = model.compute_loss(x, transformer_out, gcn_out, attention_weights)

# Detect anomalies
anomaly_scores, anomaly_labels = model.detect_anomalies(x)

print(f"Detected {anomaly_labels.sum().item()} anomalies")
```

### Training Loop Example

```python
import torch.optim as optim

# Create model and optimizer
model = create_metg_model(input_dim=25)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_data in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        transformer_out, gcn_out, attention_weights = model(batch_data)
        
        # Compute loss
        total_loss, loss_dict = model.compute_loss(
            batch_data, transformer_out, gcn_out, attention_weights
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
```

## Model Parameters

### Main Model (`METG`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | - | Number of input features (required) |
| `d_model` | 512 | Transformer model dimension |
| `memory_size` | 100 | Number of memory items (F) |
| `seq_len` | 100 | Input sequence length |
| `nhead` | 8 | Number of attention heads |
| `num_layers` | 3 | Number of transformer layers |
| `k_neighbors` | 3 | Number of nearest neighbors for graph |
| `temperature` | 1.0 | Temperature for memory attention |

### Loss Function Weights

- `lambda1` (λ₁): Weight for GCN reconstruction loss (default: 1.0)
- `lambda2` (λ₂): Weight for sparsity loss (default: 0.01)

## Mathematical Formulation

### Memory Attention Mechanism

The memory module updates queries using attention weights:

```
q̃ₜ = Σᵢ wᵢ × mᵢ

wᵢ = exp(⟨mᵢ, qₜ⟩/τ) / Σⱼ exp(⟨mⱼ, qₜ⟩/τ)
```

Where:
- `qₜ`: Query vector at time t
- `mᵢ`: i-th memory item
- `τ`: Temperature parameter
- `wᵢ`: Attention weight for memory item i

### Graph Construction

Adjacency matrix construction using cosine similarity:

```
eᵢⱼ = α × (vᵢ - vⱼ) / (‖vᵢ‖ × ‖vⱼ‖)

Aᵢⱼ = 1 if j ∈ TopK({eₖᵢ : k ∈ Cᵢ}), 0 otherwise
```

### Anomaly Scoring

```
Score(Xₛ) = m_score × (L_MSE + L_rec)

m_score = softmax(‖q̃ₜ - mᵢ‖²)
```

## Performance

The original paper reports the following results on public datasets:

| Dataset | Precision | Recall | F1-Score |
|---------|-----------|---------|----------|
| SMD | 92.53% | 86.92% | 89.64% |
| MSL | 92.15% | 95.07% | 93.59% |
| PSM | 93.62% | 99.18% | 98.38% |
| SMAP | 94.31% | 98.55% | 96.38% |
| **Average** | - | - | **94.50%** |

## Dataset Format

The model expects input data in the following format:

- **Shape**: `(batch_size, sequence_length, num_features)`
- **Type**: `torch.Tensor` (float32)
- **Normalization**: Min-max normalization recommended

Example preprocessing:
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Normalize each feature
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(raw_data.reshape(-1, num_features))
normalized_data = normalized_data.reshape(num_samples, seq_len, num_features)

# Convert to tensor
tensor_data = torch.FloatTensor(normalized_data)
```

## Customization

### Custom Memory Size

```python
# For complex datasets, increase memory size
model = create_metg_model(
    input_dim=50,
    memory_size=200,  # More memory items
    d_model=512
)
```

### Custom Graph Connectivity

```python
# Adjust k-nearest neighbors for graph construction
model = create_metg_model(
    input_dim=25,
    k_neighbors=5  # More connections in graph
)
```

### Custom Loss Weights

```python
model = create_metg_model(input_dim=25)
model.lambda1 = 2.0  # Increase GCN loss weight
model.lambda2 = 0.05  # Increase sparsity loss weight
```

## Evaluation Metrics

Common metrics for anomaly detection evaluation:

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Convert predictions to numpy
y_true = ground_truth_labels.numpy()
y_pred = anomaly_labels.numpy()

# Compute metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

## Troubleshooting

### Common Issues

1. **CUDA Memory Error**
   - Reduce batch size or sequence length
   - Use gradient checkpointing for large models

2. **Poor Performance**
   - Increase memory size for complex datasets
   - Adjust k_neighbors for graph construction
   - Tune loss function weights (lambda1, lambda2)

3. **Training Instability**
   - Lower learning rate (try 1e-5)
   - Increase temperature parameter for memory attention
   - Add gradient clipping

### Performance Tips

- Use GPU for training: `model.cuda()`
- Enable mixed precision training for faster computation
- Validate on held-out data during training
- Use early stopping to prevent overfitting

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{luo2024metg,
  title={Time Series Anomaly Detection Model Based on Memory-enhanced Transformer and Graph Network Joint Training},
  author={Luo, Qingqing and Dong, Jiangang},
  booktitle={2024 4th International Conference on Computational Modeling, Simulation and Data Analysis (CMSDA2024)},
  year={2024},
  organization={ACM},
  doi={10.1145/3727993.3728054}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.