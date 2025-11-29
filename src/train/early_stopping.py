"""Early stopping utility for training."""
import numpy as np


class EarlyStopping:
    """
    Early stopping utility to stop training when monitored metric stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        monitor: Metric name to monitor (e.g., "val_acc", "val_macro_f1")
        mode: "max" to maximize metric, "min" to minimize
        min_delta: Minimum change to qualify as improvement
    """
    def __init__(self, patience: int = 5, monitor: str = "val_acc", mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        assert mode in ["min", "max"], f"Mode must be 'min' or 'max', got {mode}"
        self.is_better = (lambda a, b: a < b - min_delta) if mode == "min" else (lambda a, b: a > b + min_delta)
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score for the monitored metric
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False

