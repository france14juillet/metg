"""
Utility functions for USAD implementation.
This is a minimal stub to support the existing USAD code.
"""

import numpy as np
import yaml
import os


def get_threshold(recette, threshold_dir):
    """
    Get threshold value for a given recette.
    
    Args:
        recette (str): Recipe identifier
        threshold_dir (str): Directory containing threshold files
        
    Returns:
        float: Threshold value
    """
    threshold_file = os.path.join(threshold_dir, f"{recette}_threshold.yml")
    
    if os.path.exists(threshold_file):
        with open(threshold_file, 'r') as f:
            threshold_data = yaml.safe_load(f)
            return threshold_data.get('threshold', 0.5)
    else:
        # Default threshold if file doesn't exist
        return 0.5


def to_device(obj, device):
    """
    Move object to specified device.
    
    Args:
        obj: Object to move (tensor or model)
        device: Target device
        
    Returns:
        Object moved to device
    """
    if hasattr(obj, 'to'):
        return obj.to(device)
    return obj


def create_windows(data, window_size):
    """
    Create sliding windows from time series data.
    
    Args:
        data (np.ndarray): Input data
        window_size (int): Size of sliding window
        
    Returns:
        np.ndarray: Windowed data
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    if len(data.shape) == 1:
        return sliding_window_view(data, window_size)
    else:
        windows = sliding_window_view(data, (window_size, data.shape[1]))
        return windows.reshape(-1, window_size, data.shape[1])