"""Model factory for creating models with different transfer learning strategies."""
import torch.nn as nn
import timm
from typing import Literal


def create_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    finetune_mode: Literal["head_only", "last_block", "full", "scratch"] = "full",
    img_size: int = None
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
        img_size: Input image size (height/width). If None, uses model default.
                  Required for some models like Swin Transformer to match dataset size.
    
    Returns:
        Configured model with appropriate parameters frozen/unfrozen
    """
    if finetune_mode == "scratch":
        pretrained = False
    
    # For most timm models, input size is either hardcoded in the architecture
    # (e.g., Swin models have _224 in their name) or not configurable via img_size parameter.
    # The dataset must match the model's expected size (typically 224 for pretrained models).
    # We don't pass img_size to timm.create_model as most models don't accept it.
    model_kwargs = {"pretrained": pretrained, "num_classes": num_classes}
    
    # Note: Some models might support input_size or img_size, but it's model-specific.
    # For now, we rely on the dataset producing the correct size (224x224 for most pretrained models).
    # If a model needs a different size, it should be specified in the model name or handled separately.
    
    model = timm.create_model(backbone, **model_kwargs)
    
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

