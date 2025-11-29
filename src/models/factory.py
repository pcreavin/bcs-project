"""Model factory for creating models with different transfer learning strategies."""
import torch.nn as nn
import timm
from typing import Literal


def create_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    finetune_mode: Literal["head_only", "last_block", "full", "scratch"] = "full"
) -> nn.Module:
    """Create model with specified transfer learning strategy."""
    if finetune_mode == "scratch":
        pretrained = False
    
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    
    if finetune_mode == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        
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
            for name, module in reversed(list(model.named_children())):
                if isinstance(module, (nn.Linear, nn.Sequential)):
                    for param in module.parameters():
                        param.requires_grad = True
                    break
    
    elif finetune_mode == "last_block":
        _freeze_except_last_blocks(model, backbone)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model created: {trainable:,}/{total:,} parameters trainable ({100*trainable/total:.1f}%)")
    
    return model


def _freeze_except_last_blocks(model: nn.Module, backbone: str):
    """Freeze all layers except the last block(s)."""
    if "efficientnet" in backbone.lower():
        for name, param in model.named_parameters():
            if any(x in name for x in ["blocks.6", "blocks.7", "classifier", "head"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif "resnet" in backbone.lower() or "resnext" in backbone.lower():
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name or "head" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        params = list(model.named_parameters())
        total = len(params)
        for i, (name, param) in enumerate(params):
            if i >= int(0.8 * total):
                param.requires_grad = True
            else:
                param.requires_grad = False

