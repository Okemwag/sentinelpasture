"""
Metrics and Performance Tracking
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime


def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate accuracy"""
    return float(np.mean(predictions == targets))


def calculate_precision_recall(
    predictions: np.ndarray, 
    targets: np.ndarray,
    positive_class: int = 1
) -> Dict[str, float]:
    """Calculate precision and recall"""
    tp = np.sum((predictions == positive_class) & (targets == positive_class))
    fp = np.sum((predictions == positive_class) & (targets != positive_class))
    fn = np.sum((predictions != positive_class) & (targets == positive_class))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall)
    }


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


class MetricsTracker:
    """Track and aggregate metrics over time"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.timestamps: Dict[str, List[datetime]] = defaultdict(list)
    
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value"""
        self.metrics[metric_name].append(value)
        self.timestamps[metric_name].append(datetime.now())
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric"""
        values = self.metrics.get(metric_name, [])
        if not values:
            return 0.0
        
        if last_n:
            values = values[-last_n:]
        
        return float(np.mean(values))
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.timestamps.clear()
