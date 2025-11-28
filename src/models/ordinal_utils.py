"""Utility functions for ordinal regression decoding (inference only)."""
import torch
import torch.nn.functional as F
from typing import Literal


def decode_ordinal(logits: torch.Tensor, method: Literal["threshold_count", "expected_value", "max_prob"] = "threshold_count") -> torch.Tensor:
    """
    Decode ordinal threshold logits to class predictions.
    
    For CORAL-style ordinal regression, the model outputs (num_classes - 1) threshold logits.
    This function converts those logits to class predictions using various strategies.
    
    Args:
        logits: Tensor of shape (batch_size, num_thresholds) where num_thresholds = num_classes - 1
        method: Decoding method to use:
            - "threshold_count": Count how many thresholds are >= 0.5 (most common)
            - "expected_value": Use expected value (sum of probabilities)
            - "max_prob": Use threshold with highest probability
    
    Returns:
        Tensor of shape (batch_size,) with predicted class indices (0 to num_classes-1)
    """
    if method == "threshold_count":
        probs = torch.sigmoid(logits)
        predicted_class = (probs >= 0.5).sum(dim=1).long()
        return predicted_class
    
    elif method == "expected_value":
        probs = torch.sigmoid(logits)
        expected = probs.sum(dim=1)
        predicted_class = torch.round(expected).long()
        predicted_class = torch.clamp(predicted_class, 0, logits.shape[1])
        return predicted_class
    
    elif method == "max_prob":
        probs = torch.sigmoid(logits)
        max_threshold_idx = probs.argmax(dim=1)
        predicted_class = torch.where(
            probs.gather(1, max_threshold_idx.unsqueeze(1)).squeeze(1) > 0.5,
            max_threshold_idx + 1,
            max_threshold_idx
        ).long()
        predicted_class = torch.clamp(predicted_class, 0, logits.shape[1])
        return predicted_class
    
    else:
        raise ValueError(f"Unknown decoding method: {method}")


