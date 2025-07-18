"""
Knee detection utilities for threshold computation.
This is a minimal stub to support the existing USAD code.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import yaml
import os


def compute_and_graph_treshold(recette, multiplier=3.0):
    """
    Compute and graph threshold for anomaly detection.
    
    Args:
        recette (str): Recipe identifier
        multiplier (float): Multiplier for threshold computation
        
    Returns:
        float: Computed threshold
    """
    
    # Load anomaly scores
    input_file = f"output/anomaly_scores/{recette}_anomaly_results.xlsx"
    
    if not os.path.exists(input_file):
        print(f"Anomaly scores file not found: {input_file}")
        return 0.5
    
    try:
        df = pd.read_excel(input_file)
        
        if len(df) == 0:
            print(f"Empty anomaly scores file: {input_file}")
            return 0.5
        
        # Assume the anomaly scores are in the second column
        scores = df.iloc[:, 1].values
        
        # Compute threshold using statistical method
        threshold = compute_threshold_statistical(scores, multiplier)
        
        # Save threshold
        threshold_dir = "output/threshold"
        os.makedirs(threshold_dir, exist_ok=True)
        threshold_file = os.path.join(threshold_dir, f"{recette}_threshold.yml")
        
        with open(threshold_file, 'w') as f:
            yaml.dump({'threshold': float(threshold)}, f)
        
        print(f"Computed threshold for {recette}: {threshold:.6f}")
        
        # Create visualization
        create_threshold_plot(scores, threshold, recette)
        
        return threshold
        
    except Exception as e:
        print(f"Error computing threshold for {recette}: {e}")
        return 0.5


def compute_threshold_statistical(scores, multiplier=3.0):
    """
    Compute threshold using statistical method.
    
    Args:
        scores (np.ndarray): Anomaly scores
        multiplier (float): Multiplier for standard deviation
        
    Returns:
        float: Computed threshold
    """
    
    # Use mean + multiplier * std as threshold
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    threshold = mean_score + multiplier * std_score
    
    # Ensure threshold is within reasonable bounds
    threshold = max(threshold, np.percentile(scores, 95))
    threshold = min(threshold, np.percentile(scores, 99.5))
    
    return threshold


def find_knee_point(scores):
    """
    Find knee point in sorted scores using simple method.
    
    Args:
        scores (np.ndarray): Anomaly scores
        
    Returns:
        float: Knee point threshold
    """
    
    # Sort scores
    sorted_scores = np.sort(scores)
    
    # Find point with maximum curvature (simple approximation)
    n = len(sorted_scores)
    if n < 10:
        return np.percentile(sorted_scores, 95)
    
    # Compute second derivative approximation
    diffs = np.diff(sorted_scores)
    second_diffs = np.diff(diffs)
    
    # Find maximum second derivative
    if len(second_diffs) > 0:
        knee_idx = np.argmax(second_diffs) + 1
        knee_point = sorted_scores[knee_idx]
    else:
        knee_point = np.percentile(sorted_scores, 95)
    
    return knee_point


def create_threshold_plot(scores, threshold, recette):
    """
    Create visualization of scores and threshold.
    
    Args:
        scores (np.ndarray): Anomaly scores
        threshold (float): Computed threshold
        recette (str): Recipe identifier
    """
    
    try:
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=50,
            name='Anomaly Scores',
            opacity=0.7
        ))
        
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold:.4f}"
        )
        
        fig.update_layout(
            title=f"Anomaly Score Distribution - Recipe {recette}",
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            showlegend=True
        )
        
        # Save plot
        plot_dir = "output/threshold_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = os.path.join(plot_dir, f"{recette}_threshold_plot.html")
        
        fig.write_html(plot_file)
        print(f"Threshold plot saved: {plot_file}")
        
    except Exception as e:
        print(f"Error creating threshold plot: {e}")
        pass