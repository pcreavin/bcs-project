"""Early stopping utility for training."""
import numpy as np


class EarlyStopping:
    """Early stopping utility to stop training when monitored metric stops improving."""
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
        """Check if training should stop."""
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

