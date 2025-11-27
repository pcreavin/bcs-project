"""Model factory for creating models with different transfer learning strategies."""
import torch.nn as nn
import timm
from typing import Literal, Optional
from .heads import OrdinalHead


def create_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    finetune_mode: Literal["head_only", "last_block", "full", "scratch"] = "full",
    head_type: Literal["classification", "ordinal"] = "classification"
) -> nn.Module:
    """
    Create model with specified transfer learning strategy.
    
    Args:
        backbone: timm model name (e.g., "efficientnet_b0")
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (ignored if finetune_mode="scratch")
        finetune_mode: Transfer learning strategy
            - "scratch": Train from scratch (pretrained=False)
            - "head_only": Freeze backbone, train only classifier head
            - "last_block": Freeze early layers, unfreeze last block(s)
            - "full": Fine-tune all parameters
        head_type: Type of prediction head
            - "classification": Standard classification head (num_classes outputs)
            - "ordinal": Ordinal regression head (num_classes-1 threshold logits)
    
    Returns:
        Configured model with appropriate parameters frozen/unfrozen
    """
    if finetune_mode == "scratch":
        pretrained = False
    
    # Create base model - we'll replace the head if ordinal
    if head_type == "ordinal":
        # Create model with classifier first to get feature dimension
        temp_model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
        
        # Get feature dimension from classifier/head
        in_features = None
        if hasattr(temp_model, "classifier"):
            if isinstance(temp_model.classifier, nn.Linear):
                in_features = temp_model.classifier.in_features
            elif isinstance(temp_model.classifier, nn.Sequential):
                for module in temp_model.classifier:
                    if isinstance(module, nn.Linear):
                        in_features = module.in_features
                        break
        elif hasattr(temp_model, "head"):
            if isinstance(temp_model.head, nn.Linear):
                in_features = temp_model.head.in_features
            elif isinstance(temp_model.head, nn.Sequential):
                for module in temp_model.head:
                    if isinstance(module, nn.Linear):
                        in_features = module.in_features
                        break
        
        if in_features is None:
            raise ValueError("Could not determine feature dimension from backbone")
        
        # Now create model and replace classifier with ordinal head
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        
        # Replace classifier/head with ordinal head
        ordinal_head = OrdinalHead(in_features, num_classes)
        if hasattr(model, "classifier"):
            model.classifier = ordinal_head
        elif hasattr(model, "head"):
            model.head = ordinal_head
        else:
            raise ValueError("Could not find classifier or head attribute in model")
        
        del temp_model  # Clean up
    else:
        # Standard classification model
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    
    if finetune_mode == "head_only":
        # Freeze all layers except classifier
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier/head
        # EfficientNet uses "classifier", other models might use "fc" or "head"
        head_attrs = ["classifier", "head", "fc"]
        head = None
        for attr in head_attrs:
            if hasattr(model, attr):
                head = getattr(model, attr)
                break
        
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True
        else:
            # Fallback: find the last module that looks like a classifier
            for name, module in reversed(list(model.named_children())):
                if isinstance(module, (nn.Linear, nn.Sequential)):
                    for param in module.parameters():
                        param.requires_grad = True
                    break
    
    elif finetune_mode == "last_block":
        # Freeze early layers, unfreeze last block(s)
        _freeze_except_last_blocks(model, backbone)
    
    # Count trainable parameters for logging
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model created: {trainable:,}/{total:,} parameters trainable ({100*trainable/total:.1f}%)")
    
    return model


def _freeze_except_last_blocks(model: nn.Module, backbone: str):
    """
    Freeze all layers except the last block(s).
    Architecture-specific implementation for EfficientNet and generic fallback.
    """
    if "efficientnet" in backbone.lower():
        # EfficientNet structure: blocks (0-6), classifier
        # Unfreeze last 2 blocks + classifier
        for name, param in model.named_parameters():
            if any(x in name for x in ["blocks.6", "blocks.7", "classifier", "head"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif "resnet" in backbone.lower() or "resnext" in backbone.lower():
        # ResNet: typically layer4 is the last block
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name or "head" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # Generic: unfreeze last 20% of parameters (by parameter count)
        params = list(model.named_parameters())
        total = len(params)
        for i, (name, param) in enumerate(params):
            if i >= int(0.8 * total):  # Last 20%
                param.requires_grad = True
            else:
                param.requires_grad = False

