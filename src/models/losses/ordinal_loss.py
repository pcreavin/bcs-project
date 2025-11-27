"""Loss functions for ordinal regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class OrdinalLoss(nn.Module):
    """
    CORAL-style ordinal regression loss.
    
    Converts class labels to threshold targets:
    - For class k: thresholds 0..k-1 should be 0, thresholds k..K-2 should be 1
    
    Then applies binary cross-entropy loss with logits across all thresholds.
    
    Args:
        num_classes: Number of ordered classes
        threshold_weights: Optional weights for each threshold [num_thresholds]
            Can be used to weight certain thresholds more (e.g., underweight detection)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, num_classes: int, threshold_weights: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.reduction = reduction
        
        if threshold_weights is not None:
            if len(threshold_weights) != self.num_thresholds:
                raise ValueError(f"threshold_weights length ({len(threshold_weights)}) must equal num_thresholds ({self.num_thresholds})")
            self.register_buffer("threshold_weights", threshold_weights.float())
        else:
            self.register_buffer("threshold_weights", torch.ones(self.num_thresholds))
    
    def class_to_threshold_targets(self, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Convert class labels to threshold binary targets.
        
        For class k:
        - thresholds 0 to k-1: target = 0 (class is NOT >= threshold)
        - thresholds k to K-2: target = 1 (class IS >= threshold)
        
        Args:
            class_labels: Class indices [batch_size] in range [0, num_classes-1]
        
        Returns:
            Threshold targets [batch_size, num_thresholds] with values in {0, 1}
        """
        batch_size = class_labels.shape[0]
        targets = torch.zeros(batch_size, self.num_thresholds, dtype=torch.float32, device=class_labels.device)
        
        # For each sample, set thresholds k to num_thresholds-1 to 1
        for k in range(self.num_thresholds):
            # Threshold k corresponds to "class >= k+1"
            # So if class >= k+1, then threshold target = 1
            targets[:, k] = (class_labels >= (k + 1)).float()
        
        return targets
    
    def forward(self, logits: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal regression loss.
        
        Args:
            logits: Threshold logits [batch_size, num_thresholds]
            class_labels: True class indices [batch_size] in range [0, num_classes-1]
        
        Returns:
            Loss scalar (or tensor if reduction='none')
        """
        # Convert class labels to threshold targets
        targets = self.class_to_threshold_targets(class_labels)
        
        # Binary cross-entropy with logits for each threshold
        # This is more numerically stable than sigmoid + BCE
        loss_per_threshold = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # [batch_size, num_thresholds]
        
        # Apply threshold weights
        loss_per_threshold = loss_per_threshold * self.threshold_weights.unsqueeze(0)
        
        # Average across thresholds first, then across batch
        loss_per_sample = loss_per_threshold.mean(dim=1)  # [batch_size]
        
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        elif self.reduction == "none":
            return loss_per_sample
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def create_ordinal_loss(num_classes: int, threshold_weights: Optional[list] = None, reduction: str = "mean"):
    """
    Convenience function to create ordinal loss.
    
    Args:
        num_classes: Number of ordered classes
        threshold_weights: Optional list of weights for each threshold
            Example: [2.0, 1.0, 1.0, 1.0] to weight first threshold more
        reduction: Reduction method
    
    Returns:
        OrdinalLoss instance
    """
    if threshold_weights is not None:
        threshold_weights = torch.tensor(threshold_weights, dtype=torch.float32)
    return OrdinalLoss(num_classes, threshold_weights=threshold_weights, reduction=reduction)

