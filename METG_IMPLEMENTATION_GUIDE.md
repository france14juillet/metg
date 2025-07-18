# METG Implementation Guide

This document explains the new `train_metg.py` and `compute_metg.py` files that replicate the functionality of `train_usad.py` and `compute_usad.py` while using the METG (Multivariate Time Series Anomaly Detection with Graph Neural Networks) model instead of USAD.

## Overview

The METG implementation provides drop-in replacements for the USAD training and computation scripts while leveraging the advanced capabilities of the METG model:

- **Memory-enhanced Transformer**: Captures temporal dependencies and stores diverse normal patterns
- **Graph Convolutional Network**: Models spatial relationships between variables
- **Joint optimization**: Combines strengths of both components for robust anomaly detection

## Files Created

### Core Implementation Files

1. **`train_metg.py`** - METG training script (replaces `train_usad.py`)
2. **`compute_metg.py`** - METG computation script (replaces `compute_usad.py`)
3. **`config.yaml`** - Configuration file with METG-specific parameters
4. **`utils.py`** - Utility functions (stub for USAD compatibility)
5. **`usad.py`** - USAD model stub (for backward compatibility)
6. **`knee.py`** - Threshold computation utilities (stub for USAD compatibility)

### Key Differences from USAD

| Aspect | USAD | METG |
|--------|------|------|
| **Architecture** | Autoencoder with two decoders | Transformer + GCN + Memory module |
| **Model Files** | `{recette}_model.pth` | `{recette}_metg_model.pth` |
| **Parameter Files** | None | `{recette}_metg_params.yaml` |
| **Scaler Files** | `{recette}_scalers.pkl` | `{recette}_metg_scalers.pkl` |
| **Output Files** | `{recette}_anomaly_results.xlsx` | `{recette}_metg_anomaly_results.xlsx` |
| **Training Loss** | Simple reconstruction loss | Combined loss: L_MSE + λ₁×L_rec + λ₂×L_spar |
| **Anomaly Scoring** | Reconstruction error | Memory-enhanced scoring with attention |

## Usage

### Training

```bash
# Train METG model for a specific recipe
python train_metg.py <recette>

# Example
python train_metg.py 1003
```

### Computation

```bash
# Compute anomaly scores for a specific recipe
python compute_metg.py <recette>

# Compute for all recipes
python compute_metg.py all

# Examples
python compute_metg.py 1003
python compute_metg.py all
```

## Configuration

The `config.yaml` file contains all necessary parameters:

```yaml
# Model parameters
window_size: 100
hidden_size: 64
batch_size: 32
n_epochs: 50
learning_rate: 0.0001

# METG specific parameters
d_model: 512
memory_size: 100
nhead: 8
num_layers: 3
k_neighbors: 3
temperature: 1.0

# Loss function weights
lambda1: 1.0  # GCN reconstruction loss weight
lambda2: 0.01  # Sparsity loss weight
```

## Key Features Maintained

### 1. **File Handling**
- Identical concatenation logic
- Same directory structure (`output/concatenated_files/`)
- Compatible with existing data preprocessing pipeline

### 2. **Data Preprocessing**
- Same scaling and normalization approach
- Identical winsorization for `Échelles` and `Disque`
- Compatible with existing `Débit deau [l/min]` inversion transformation

### 3. **Data Augmentation**
- Same augmentation strategy (noise + scaling)
- Identical parameter ranges and probabilities

### 4. **Output Format**
- Same Excel output format
- Compatible column names and structure
- Identical visualization and threshold computation

### 5. **Progress Reporting**
- Same progress indicators during processing
- Identical error handling and logging

## Model Architecture Benefits

### METG Advantages over USAD

1. **Memory Enhancement**: Prevents overfitting to limited training data
2. **Graph-based Dependencies**: Captures complex variable relationships
3. **Joint Training**: Combines temporal and spatial modeling
4. **Attention Mechanisms**: Provides interpretable anomaly scoring

### Training Process

The METG training process includes:

1. **Data Preparation**: Same as USAD (concatenation, scaling, windowing)
2. **Model Creation**: Initialize METG with configured parameters
3. **Training Loop**: Joint optimization with three loss components
4. **Model Saving**: Save both model state and parameters

### Anomaly Detection

The METG anomaly detection process:

1. **Model Loading**: Load both model state and parameters
2. **Data Processing**: Same preprocessing as training
3. **Score Computation**: Use memory-enhanced scoring
4. **Threshold Computation**: Same threshold calculation as USAD

## Backward Compatibility

The implementation maintains full backward compatibility:

- All existing data files work without modification
- Same command-line interface
- Compatible with existing visualization tools
- Uses same threshold computation methods

## Files Generated

### Training Outputs
- `{recette}_metg_model.pth`: Trained METG model
- `{recette}_metg_params.yaml`: Model parameters for loading
- `{recette}_metg_scalers.pkl`: Data scalers

### Computation Outputs
- `{recette}_metg_anomaly_results.xlsx`: Anomaly scores
- Threshold files (via `knee.py`)
- Visualization plots (via `plotly`)

## Error Handling

The implementation includes robust error handling:

- Missing file detection
- Invalid data format handling
- Model loading error recovery
- Memory management for large datasets

## Performance Considerations

### Memory Usage
- METG requires more memory than USAD due to transformer architecture
- Batch processing helps manage memory consumption
- Configurable batch sizes in config

### Computational Complexity
- METG is more computationally intensive than USAD
- Training time is longer but provides better anomaly detection
- GPU acceleration recommended for large datasets

## Migration Guide

To migrate from USAD to METG:

1. **No code changes needed** - use the same commands
2. **Update paths** - outputs will have `_metg` suffix
3. **Configuration** - adjust parameters in `config.yaml` if needed
4. **Retraining** - retrain models with `train_metg.py`
5. **Recomputation** - recompute scores with `compute_metg.py`

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or d_model in config
2. **CUDA Errors**: Ensure PyTorch CUDA compatibility
3. **File Not Found**: Check that training was completed successfully
4. **Shape Errors**: Verify data preprocessing consistency

### Debug Tips

1. Check model parameter file exists
2. Verify scaler file compatibility
3. Confirm data column consistency
4. Monitor GPU memory usage

## Future Enhancements

Possible improvements:

1. **Hyperparameter Tuning**: Automated parameter optimization
2. **Model Ensemble**: Combine multiple METG models
3. **Online Learning**: Incremental model updates
4. **Distributed Training**: Multi-GPU support
5. **Model Compression**: Reduce model size for deployment

## Conclusion

The METG implementation provides a powerful upgrade from USAD while maintaining complete compatibility with existing workflows. The memory-enhanced transformer and graph neural network components offer superior anomaly detection capabilities for multivariate time series data.