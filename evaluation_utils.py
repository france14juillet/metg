#!/usr/bin/env python3
"""
Enhanced evaluation utilities for METG anomaly detection.

This module provides advanced evaluation capabilities including:
- ROC curve analysis and optimal threshold finding
- Score distribution visualization
- Detailed confusion matrix metrics
- Configurable sequence-level aggregation methods
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report
)
from typing import Tuple, Dict, List, Optional, Union
import warnings

def evaluate_anomaly_detection(
    model,
    test_data: torch.Tensor,
    test_labels: np.ndarray,
    device: str = 'cpu',
    aggregation_method: str = 'mean',
    threshold_percentiles: List[float] = None,
    min_anomaly_ratio: float = 0.1,
    plot_results: bool = True,
    save_plots: bool = True,
    plot_prefix: str = 'metg_evaluation'
) -> Dict:
    """
    Comprehensive evaluation of anomaly detection model.
    
    Args:
        model: Trained METG model
        test_data: Test data tensor (num_samples, seq_len, num_features)
        test_labels: Ground truth labels (num_samples,)
        device: Device to run evaluation on
        aggregation_method: Method to aggregate time-step scores ('max', 'mean', 'min_count')
        threshold_percentiles: List of percentiles to test for threshold tuning
        min_anomaly_ratio: For 'min_count' method, minimum ratio of anomalous time steps
        plot_results: Whether to generate plots
        save_plots: Whether to save plots to files
        plot_prefix: Prefix for saved plot files
        
    Returns:
        Dictionary containing comprehensive evaluation results
    """
    if threshold_percentiles is None:
        threshold_percentiles = [90, 95, 99, 99.5, 99.9]
    
    model.to(device)
    model.eval()
    
    print(f"Evaluating with aggregation method: {aggregation_method}")
    print(f"Test set: {len(test_data)} samples ({test_labels.sum()} anomalies)")
    
    # Compute anomaly scores for all test data
    all_timestep_scores = []
    all_sequence_scores = []
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_data = test_data[i:i+batch_size].to(device)
            
            if aggregation_method == 'min_count':
                # Use a high percentile for time-step threshold in min_count method
                timestep_scores, sequence_scores, _ = model.detect_anomalies(
                    batch_data, 
                    threshold_percentile=95,
                    aggregation_method=aggregation_method,
                    min_anomaly_ratio=min_anomaly_ratio
                )
                # For min_count, sequence_scores are actually anomaly ratios
            else:
                timestep_scores, sequence_scores, _ = model.detect_anomalies(
                    batch_data,
                    aggregation_method=aggregation_method
                )
            
            all_timestep_scores.append(timestep_scores.cpu().numpy())
            all_sequence_scores.append(sequence_scores.cpu().numpy())
    
    # Concatenate all scores
    timestep_scores = np.concatenate(all_timestep_scores, axis=0)  # (num_samples, seq_len)
    sequence_scores = np.concatenate(all_sequence_scores, axis=0)  # (num_samples,)
    
    # Evaluate different thresholds
    results = {
        'aggregation_method': aggregation_method,
        'timestep_scores': timestep_scores,
        'sequence_scores': sequence_scores,
        'ground_truth': test_labels,
        'threshold_results': {}
    }
    
    print(f"\nSequence score statistics:")
    print(f"  Min: {sequence_scores.min():.6f}")
    print(f"  Max: {sequence_scores.max():.6f}")
    print(f"  Mean: {sequence_scores.mean():.6f}")
    print(f"  Std: {sequence_scores.std():.6f}")
    
    best_f1 = 0
    best_threshold_info = None
    
    # Test different threshold percentiles
    print(f"\nThreshold analysis:")
    print(f"{'Percentile':<10} {'Threshold':<12} {'TP':<4} {'FP':<4} {'TN':<4} {'FN':<4} {'Precision':<9} {'Recall':<7} {'F1':<7}")
    print("-" * 80)
    
    for percentile in threshold_percentiles:
        threshold = np.percentile(sequence_scores, percentile)
        predictions = (sequence_scores > threshold).astype(int)
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        
        # Compute metrics (handle division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{percentile:<10.1f} {threshold:<12.6f} {tp:<4} {fp:<4} {tn:<4} {fn:<4} {precision:<9.4f} {recall:<7.4f} {f1:<7.4f}")
        
        results['threshold_results'][percentile] = {
            'threshold': threshold,
            'predictions': predictions,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_info = {
                'percentile': percentile,
                'threshold': threshold,
                'predictions': predictions,
                'metrics': {'precision': precision, 'recall': recall, 'f1_score': f1},
                'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
            }
    
    results['best_threshold'] = best_threshold_info
    
    print(f"\nBest performance:")
    print(f"  Percentile: {best_threshold_info['percentile']}")
    print(f"  Threshold: {best_threshold_info['threshold']:.6f}")
    print(f"  Precision: {best_threshold_info['metrics']['precision']:.4f}")
    print(f"  Recall: {best_threshold_info['metrics']['recall']:.4f}")
    print(f"  F1-Score: {best_threshold_info['metrics']['f1_score']:.4f}")
    
    cm = best_threshold_info['confusion_matrix']
    print(f"  TP: {cm['tp']}, FP: {cm['fp']}, TN: {cm['tn']}, FN: {cm['fn']}")
    print(f"  Detected {cm['tp'] + cm['fp']} anomalies out of {test_labels.sum()} actual anomalies")
    
    # Generate plots if requested
    if plot_results:
        _plot_evaluation_results(results, save_plots, plot_prefix)
    
    return results


def _plot_evaluation_results(results: Dict, save_plots: bool = True, plot_prefix: str = 'metg_evaluation'):
    """Generate comprehensive evaluation plots."""
    
    sequence_scores = results['sequence_scores']
    ground_truth = results['ground_truth']
    best_threshold_info = results['best_threshold']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'METG Anomaly Detection Evaluation ({results["aggregation_method"]} aggregation)', fontsize=16)
    
    # Plot 1: Score distribution
    ax1 = axes[0, 0]
    normal_scores = sequence_scores[ground_truth == 0]
    anomaly_scores = sequence_scores[ground_truth == 1]
    
    ax1.hist(normal_scores, bins=50, alpha=0.7, label=f'Normal (n={len(normal_scores)})', 
             density=True, color='blue')
    ax1.hist(anomaly_scores, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_scores)})', 
             density=True, color='red')
    
    # Add threshold line
    threshold = best_threshold_info['threshold']
    ax1.axvline(threshold, color='black', linestyle='--', 
                label=f'Optimal threshold ({best_threshold_info["percentile"]}th percentile)')
    
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Score Distribution by Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ROC Curve
    ax2 = axes[0, 1]
    try:
        fpr, tpr, _ = roc_curve(ground_truth, sequence_scores)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        # Mark optimal threshold point
        best_pred = best_threshold_info['predictions']
        cm = confusion_matrix(ground_truth, best_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            opt_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            opt_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            ax2.plot(opt_fpr, opt_tpr, 'ro', markersize=8, 
                    label=f'Optimal point (F1={best_threshold_info["metrics"]["f1_score"]:.3f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        ax2.text(0.5, 0.5, f'ROC curve unavailable\n{str(e)}', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('ROC Curve (Unavailable)')
    
    # Plot 3: Precision-Recall Curve
    ax3 = axes[1, 0]
    try:
        precision, recall, _ = precision_recall_curve(ground_truth, sequence_scores)
        pr_auc = auc(recall, precision)
        
        ax3.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Mark optimal threshold point
        opt_precision = best_threshold_info['metrics']['precision']
        opt_recall = best_threshold_info['metrics']['recall']
        ax3.plot(opt_recall, opt_precision, 'ro', markersize=8,
                label=f'Optimal point (F1={best_threshold_info["metrics"]["f1_score"]:.3f})')
        
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    except Exception as e:
        ax3.text(0.5, 0.5, f'PR curve unavailable\n{str(e)}', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Precision-Recall Curve (Unavailable)')
    
    # Plot 4: Threshold vs Metrics
    ax4 = axes[1, 1]
    percentiles = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for percentile, result in results['threshold_results'].items():
        percentiles.append(percentile)
        precisions.append(result['precision'])
        recalls.append(result['recall'])
        f1_scores.append(result['f1_score'])
    
    ax4.plot(percentiles, precisions, 'b-', label='Precision', marker='o')
    ax4.plot(percentiles, recalls, 'r-', label='Recall', marker='s')
    ax4.plot(percentiles, f1_scores, 'g-', label='F1-Score', marker='^')
    
    # Mark optimal threshold
    optimal_percentile = best_threshold_info['percentile']
    ax4.axvline(optimal_percentile, color='black', linestyle='--', alpha=0.7,
                label=f'Optimal ({optimal_percentile}th percentile)')
    
    ax4.set_xlabel('Threshold Percentile')
    ax4.set_ylabel('Score')
    ax4.set_title('Metrics vs Threshold Percentile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'{plot_prefix}_{results["aggregation_method"]}_evaluation.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nEvaluation plots saved as: {filename}")
    
    plt.show()


def compare_aggregation_methods(
    model,
    test_data: torch.Tensor,
    test_labels: np.ndarray,
    device: str = 'cpu',
    methods: List[str] = None,
    min_anomaly_ratio: float = 0.1,
    save_plots: bool = True
) -> Dict:
    """
    Compare different aggregation methods for sequence-level anomaly detection.
    
    Args:
        model: Trained METG model
        test_data: Test data tensor
        test_labels: Ground truth labels
        device: Device to run on
        methods: List of aggregation methods to compare
        min_anomaly_ratio: For min_count method
        save_plots: Whether to save comparison plots
        
    Returns:
        Dictionary with comparison results
    """
    if methods is None:
        methods = ['max', 'mean', 'min_count']
    
    print("Comparing aggregation methods...")
    
    comparison_results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Evaluating method: {method.upper()}")
        print('='*50)
        
        result = evaluate_anomaly_detection(
            model=model,
            test_data=test_data,
            test_labels=test_labels,
            device=device,
            aggregation_method=method,
            min_anomaly_ratio=min_anomaly_ratio,
            plot_results=False,  # Individual plots disabled for comparison
            save_plots=False
        )
        
        comparison_results[method] = result
    
    # Generate comparison summary
    print(f"\n{'='*80}")
    print("AGGREGATION METHOD COMPARISON")
    print('='*80)
    print(f"{'Method':<10} {'Best %ile':<10} {'Precision':<10} {'Recall':<8} {'F1-Score':<8} {'TP':<4} {'FP':<4}")
    print("-" * 80)
    
    best_overall_f1 = 0
    best_overall_method = None
    
    for method, result in comparison_results.items():
        best = result['best_threshold']
        cm = best['confusion_matrix']
        
        print(f"{method:<10} {best['percentile']:<10.1f} {best['metrics']['precision']:<10.4f} "
              f"{best['metrics']['recall']:<8.4f} {best['metrics']['f1_score']:<8.4f} "
              f"{cm['tp']:<4} {cm['fp']:<4}")
        
        if best['metrics']['f1_score'] > best_overall_f1:
            best_overall_f1 = best['metrics']['f1_score']
            best_overall_method = method
    
    print("-" * 80)
    print(f"Best overall method: {best_overall_method.upper()} (F1={best_overall_f1:.4f})")
    
    # Generate comparison plot
    if save_plots:
        _plot_method_comparison(comparison_results)
    
    return comparison_results


def _plot_method_comparison(comparison_results: Dict):
    """Plot comparison of aggregation methods."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Aggregation Method Comparison', fontsize=16)
    
    methods = list(comparison_results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Score distributions
    ax1 = axes[0]
    for i, (method, result) in enumerate(comparison_results.items()):
        sequence_scores = result['sequence_scores']
        ground_truth = result['ground_truth']
        
        normal_scores = sequence_scores[ground_truth == 0]
        anomaly_scores = sequence_scores[ground_truth == 1]
        
        # Plot histograms with some transparency
        ax1.hist(normal_scores, bins=30, alpha=0.3, label=f'{method} (Normal)', 
                color=colors[i % len(colors)], density=True)
        ax1.hist(anomaly_scores, bins=30, alpha=0.6, label=f'{method} (Anomaly)', 
                color=colors[i % len(colors)], density=True, histtype='step', linewidth=2)
    
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Score Distributions by Method')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance metrics
    ax2 = axes[1]
    method_names = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for method, result in comparison_results.items():
        best = result['best_threshold']
        method_names.append(method.capitalize())
        precisions.append(best['metrics']['precision'])
        recalls.append(best['metrics']['recall'])
        f1_scores.append(best['metrics']['f1_score'])
    
    x = np.arange(len(method_names))
    width = 0.25
    
    ax2.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax2.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax2.set_xlabel('Aggregation Method')
    ax2.set_ylabel('Score')
    ax2.set_title('Performance Metrics by Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for i, (p, r, f1) in enumerate(zip(precisions, recalls, f1_scores)):
        ax2.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('metg_aggregation_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as: metg_aggregation_comparison.png")
    plt.show()


def find_optimal_threshold(
    sequence_scores: np.ndarray,
    ground_truth: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, Dict]:
    """
    Find optimal threshold by maximizing specified metric.
    
    Args:
        sequence_scores: Sequence-level anomaly scores
        ground_truth: Ground truth binary labels
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    # Try many threshold values
    thresholds = np.percentile(sequence_scores, np.linspace(80, 99.9, 100))
    
    best_score = 0
    best_threshold = None
    best_metrics = None
    
    for threshold in thresholds:
        predictions = (sequence_scores > threshold).astype(int)
        
        # Compute metrics
        try:
            precision = precision_score(ground_truth, predictions, zero_division=0)
            recall = recall_score(ground_truth, predictions, zero_division=0)
            f1 = f1_score(ground_truth, predictions, zero_division=0)
            
            metrics = {'precision': precision, 'recall': recall, 'f1_score': f1}
            
            current_score = metrics[metric]
            
            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
                best_metrics = metrics
                
        except Exception:
            continue
    
    return best_threshold, best_metrics