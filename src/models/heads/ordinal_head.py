"""Ordinal (CORAL) prediction head for ordinal regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalHead(nn.Module):
    """
    CORAL-style ordinal regression head.
    
    For K ordered classes, predicts K-1 cumulative logits.
    Each logit corresponds to a threshold question:
    - "Is class >= threshold_k?"
    
    Args:
        in_features: Number of input features from backbone
        num_classes: Number of ordered classes (e.g., 5 for BCS scores)
    """
    
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # K-1 thresholds
        
        # Linear layer outputs K-1 logits (one per threshold)
        self.fc = nn.Linear(in_features, self.num_thresholds)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, in_features]
        
        Returns:
            Threshold logits [batch_size, num_thresholds]
        """
        return self.fc(x)
    
    @staticmethod
    def decode(logits: torch.Tensor, method: str = "threshold_count") -> torch.Tensor:
        """
        Decode ordinal logits to predicted class indices.
        
        Args:
            logits: Threshold logits [batch_size, num_thresholds]
            method: Decoding method
                - "threshold_count": Count thresholds >= 0.5 (standard CORAL)
                - "expected_value": Use expected class index
                - "max_prob": Use maximum probability threshold
        
        Returns:
            Predicted class indices [batch_size]
        """
        if method == "threshold_count":
            # Standard CORAL decoding: count how many thresholds are >= 0.5
            probs = torch.sigmoid(logits)
            predicted_class = (probs >= 0.5).sum(dim=1).long()
            return predicted_class
        elif method == "expected_value":
            # Expected value: sum of probabilities gives expected class
            probs = torch.sigmoid(logits)
            # Expected class = sum of threshold probabilities
            expected = probs.sum(dim=1)
            # Round to nearest integer class
            predicted_class = torch.round(expected).long()
            # Clip to valid range [0, num_classes-1]
            predicted_class = torch.clamp(predicted_class, 0, logits.shape[1])
            return predicted_class
        elif method == "max_prob":
            # Use threshold with maximum probability
            probs = torch.sigmoid(logits)
            # Find threshold with max prob, then count how many are <= that
            max_threshold_idx = probs.argmax(dim=1)
            # If max prob is at threshold k, class is k+1 (if prob > 0.5) or k (if prob <= 0.5)
            predicted_class = torch.where(
                probs.gather(1, max_threshold_idx.unsqueeze(1)).squeeze(1) > 0.5,
                max_threshold_idx + 1,
                max_threshold_idx
            ).long()
            predicted_class = torch.clamp(predicted_class, 0, logits.shape[1])
            return predicted_class
        else:
            raise ValueError(f"Unknown decoding method: {method}")



