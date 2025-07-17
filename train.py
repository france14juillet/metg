#!/usr/bin/env python3
"""
Simple training script for METG model.

Usage:
    python train.py --input_dim 25 --epochs 50 --batch_size 32
"""

import argparse
import torch
import torch.optim as optim
import numpy as np
from model import create_metg_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train METG model')
    parser.add_argument('--input_dim', type=int, required=True,
                        help='Number of input features')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Sequence length (default: 100)')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension (default: 512)')
    parser.add_argument('--memory_size', type=int, default=100,
                        help='Memory size (default: 100)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--save_path', type=str, default='metg_model.pth',
                        help='Path to save trained model')
    
    return parser.parse_args()


def generate_dummy_data(num_samples=1000, seq_len=100, num_features=10, anomaly_rate=0.05):
    """Generate dummy multivariate time series data for testing."""
    print(f"Generating {num_samples} synthetic samples...")
    
    np.random.seed(42)
    data = []
    labels = []
    
    for i in range(num_samples):
        is_anomaly = np.random.random() < anomaly_rate
        
        if is_anomaly:
            # Generate anomalous sequence
            sequence = np.random.normal(0, 1, (seq_len, num_features))
            t = np.linspace(0, 4*np.pi, seq_len)
            for j in range(num_features):
                sequence[:, j] += 0.5 * np.sin(t + j)
            
            # Inject anomaly
            anomaly_start = np.random.randint(10, seq_len - 10)
            anomaly_length = np.random.randint(5, 15)
            anomaly_features = np.random.choice(num_features, max(1, num_features//3), replace=False)
            
            for feat in anomaly_features:
                sequence[anomaly_start:anomaly_start+anomaly_length, feat] += np.random.choice([-3, 3]) * np.random.uniform(2, 4)
            
            labels.append(1)
        else:
            # Generate normal sequence
            sequence = np.random.normal(0, 1, (seq_len, num_features))
            t = np.linspace(0, 4*np.pi, seq_len)
            for j in range(num_features):
                sequence[:, j] += 0.5 * np.sin(t + j)
                if j > 0:
                    sequence[:, j] += 0.2 * sequence[:, j-1]
            
            labels.append(0)
        
        data.append(sequence)
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Normalize
    scaler = MinMaxScaler()
    data_reshaped = data.reshape(-1, num_features)
    data_normalized = scaler.fit_transform(data_reshaped)
    data = data_normalized.reshape(num_samples, seq_len, num_features)
    
    return data, labels, scaler


def train_model(args):
    """Train the METG model."""
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Generate or load data (replace this with your data loading logic)
    print("Loading data...")
    data, labels, scaler = generate_dummy_data(
        num_samples=800,
        seq_len=args.seq_len,
        num_features=args.input_dim,
        anomaly_rate=0.1
    )
    
    # Split data
    train_size = int(0.7 * len(data))
    train_data = torch.FloatTensor(data[:train_size])
    test_data = torch.FloatTensor(data[train_size:])
    
    # Use only normal samples for training (unsupervised)
    normal_indices = np.where(labels[:train_size] == 0)[0]
    train_data_normal = train_data[normal_indices]
    test_labels = labels[train_size:]
    
    print(f"Training samples: {len(train_data_normal)} (normal only)")
    print(f"Test samples: {len(test_data)} ({test_labels.sum()} anomalies)")
    
    # Create model
    model = create_metg_model(
        input_dim=args.input_dim,
        d_model=args.d_model,
        memory_size=args.memory_size,
        seq_len=args.seq_len
    )
    
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    model.train()
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_losses = []
        
        # Process in batches
        num_batches = len(train_data_normal) // args.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = (batch_idx + 1) * args.batch_size
            batch_data = train_data_normal[start_idx:end_idx].to(device)
            
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
            
            epoch_losses.append(total_loss.item())
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:3d}/{args.epochs}, Loss: {avg_loss:.6f}")
    
    print("Training completed!")
    
    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    
    all_scores = []
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), args.batch_size):
            batch_data = test_data[i:i+args.batch_size].to(device)
            timestep_scores, sequence_scores, sequence_predictions = model.detect_anomalies(
                batch_data, aggregation_method='max'
            )
            
            all_scores.extend(sequence_scores.cpu().numpy())
            all_predictions.extend(sequence_predictions.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_predictions = np.array(all_predictions)
    
    # Print results
    print("\nEvaluation Results:")
    print(classification_report(test_labels, all_predictions, 
                              target_names=['Normal', 'Anomaly']))
    
    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"\nModel saved to: {args.save_path}")
    
    return model, scaler


if __name__ == "__main__":
    args = parse_args()
    model, scaler = train_model(args)
    print("Training script completed successfully!")