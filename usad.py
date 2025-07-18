"""
USAD (Unsupervised Anomaly Detection) model implementation.
This is a minimal stub to support the existing USAD code.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader


class UsadModel(nn.Module):
    """
    USAD model with encoder and two decoders.
    This is a minimal implementation for compatibility.
    """
    
    def __init__(self, input_dim, latent_dim):
        super(UsadModel, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # Decoder 1
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim // 4, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim)
        )
        
        # Decoder 2
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim // 4, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim)
        )
    
    def forward(self, x):
        """Forward pass through the model."""
        encoded = self.encoder(x)
        decoded1 = self.decoder1(encoded)
        decoded2 = self.decoder2(encoded)
        return encoded, decoded1, decoded2


def to_device(obj, device):
    """Move object to specified device."""
    if hasattr(obj, 'to'):
        return obj.to(device)
    return obj


def training(n_epochs, model, train_loader, val_loader, lr=1e-3):
    """
    Training function for USAD model.
    
    Args:
        n_epochs (int): Number of training epochs
        model (UsadModel): USAD model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        lr (float): Learning rate
        
    Returns:
        dict: Training history
    """
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            
            encoded, decoded1, decoded2 = model(batch_data)
            
            # USAD loss computation
            loss1 = criterion(decoded1, batch_data)
            loss2 = criterion(decoded2, batch_data)
            loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch_data = batch[0].to(device)
                encoded, decoded1, decoded2 = model(batch_data)
                
                loss1 = criterion(decoded1, batch_data)
                loss2 = criterion(decoded2, batch_data)
                loss = loss1 + loss2
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return history


def testing(model, test_loader):
    """
    Testing function that computes anomaly scores.
    
    Args:
        model (UsadModel): Trained USAD model
        test_loader (DataLoader): Test data loader
        
    Returns:
        list: Anomaly scores
    """
    device = next(model.parameters()).device
    model.eval()
    
    scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_data = batch[0].to(device)
            encoded, decoded1, decoded2 = model(batch_data)
            
            # Compute reconstruction error
            error1 = torch.mean((batch_data - decoded1) ** 2, dim=1)
            error2 = torch.mean((batch_data - decoded2) ** 2, dim=1)
            
            # Combine errors (simple average)
            combined_error = (error1 + error2) / 2
            scores.append(combined_error)
    
    return scores