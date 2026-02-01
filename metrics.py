#!/usr/bin/env python3
"""
Evaluation Metrics for Extreme Volatility Prediction
"""

from typing import Dict, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, roc_curve, precision_recall_curve
)


@dataclass
class Metrics:
    """Classification metrics container"""
    auc_roc: float
    avg_precision: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    accuracy: float
    positive_rate: float
    optimal_threshold: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            'AUC-ROC': self.auc_roc,
            'Avg Precision': self.avg_precision,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1': self.f1,
            'Brier Score': self.brier_score,
            'Accuracy': self.accuracy,
            'Positive Rate': self.positive_rate,
            'Optimal Threshold': self.optimal_threshold,
        }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the threshold that maximizes F1 score.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
    
    Returns:
        Optimal threshold value
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Compute F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find the threshold that maximizes F1
    # Note: precision_recall_curve returns n+1 precisions/recalls but n thresholds
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element which has no threshold
    
    if len(thresholds) > 0:
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5
    
    # Ensure threshold is reasonable (between 0.05 and 0.95)
    optimal_threshold = np.clip(optimal_threshold, 0.05, 0.95)
    
    return optimal_threshold


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Metrics:
    """
    Compute all classification metrics.
    
    Note: y_pred is ignored. Instead, we find the optimal threshold from y_prob
    that maximizes F1 score, then compute precision/recall/F1 at that threshold.
    This is more appropriate for imbalanced classification problems.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (ignored, kept for API compatibility)
        y_prob: Predicted probabilities
    
    Returns:
        Metrics object with all computed metrics
    """
    
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return Metrics(
            auc_roc=0.5,
            avg_precision=np.mean(y_true),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            brier_score=0.25,
            accuracy=np.mean(y_true == (y_prob >= 0.5).astype(int)),
            positive_rate=np.mean(y_true),
            optimal_threshold=0.5
        )
    
    # Find optimal threshold that maximizes F1
    optimal_threshold = find_optimal_threshold(y_true, y_prob)
    
    # Generate predictions using optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    # Compute metrics
    return Metrics(
        auc_roc=roc_auc_score(y_true, y_prob),
        avg_precision=average_precision_score(y_true, y_prob),
        precision=precision_score(y_true, y_pred_optimal, zero_division=0),
        recall=recall_score(y_true, y_pred_optimal, zero_division=0),
        f1=f1_score(y_true, y_pred_optimal, zero_division=0),
        brier_score=brier_score_loss(y_true, y_prob),
        accuracy=accuracy_score(y_true, y_pred_optimal),
        positive_rate=np.mean(y_true),
        optimal_threshold=optimal_threshold
    )


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for AUC-ROC.
    
    Returns: (auc, ci_lower, ci_upper)
    """
    np.random.seed(42)
    n = len(y_true)
    aucs = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_boot = y_true[idx]
        p_boot = y_prob[idx]
        
        if len(np.unique(y_boot)) < 2:
            continue
        
        aucs.append(roc_auc_score(y_boot, p_boot))
    
    aucs = np.array(aucs)
    alpha = (1 - confidence) / 2
    
    return (
        roc_auc_score(y_true, y_prob),
        np.percentile(aucs, alpha * 100),
        np.percentile(aucs, (1 - alpha) * 100)
    )


def significance_test(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstrap: int = 2000
) -> Dict:
    """
    Test if model A significantly outperforms model B.
    Uses bootstrap test on AUC difference.
    """
    np.random.seed(42)
    n = len(y_true)
    diffs = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_boot = y_true[idx]
        
        if len(np.unique(y_boot)) < 2:
            continue
        
        auc_a = roc_auc_score(y_boot, probs_a[idx])
        auc_b = roc_auc_score(y_boot, probs_b[idx])
        diffs.append(auc_a - auc_b)
    
    diffs = np.array(diffs)
    
    auc_a_full = roc_auc_score(y_true, probs_a)
    auc_b_full = roc_auc_score(y_true, probs_b)
    
    # P-value: proportion of bootstrap samples where diff <= 0
    p_value = np.mean(diffs <= 0)
    
    return {
        'auc_a': auc_a_full,
        'auc_b': auc_b_full,
        'auc_diff': auc_a_full - auc_b_full,
        'ci_lower': np.percentile(diffs, 2.5),
        'ci_upper': np.percentile(diffs, 97.5),
        'p_value': p_value,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
    }