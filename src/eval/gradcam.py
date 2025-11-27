"""Grad-CAM visualization for BCS classification models."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for model interpretation.
    
    Works with timm models by hooking into the last convolutional layer.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer to visualize
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activation maps during forward pass."""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients during backward pass."""
        self.gradients = grad_output[0]
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = use predicted class)
        
        Returns:
            Grad-CAM heatmap as numpy array (H, W)
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Calculate Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()


def get_target_layer(model: torch.nn.Module, backbone: str):
    """
    Get the target convolutional layer for Grad-CAM.
    
    Args:
        model: PyTorch model
        backbone: Model backbone name (e.g., "efficientnet_b0")
    
    Returns:
        Target layer module
    """
    # For EfficientNet, target the last block's convolution
    if "efficientnet" in backbone.lower():
        # EfficientNet structure: blocks.0 through blocks.6 (or blocks.7 for b6+)
        # Target the last block's depthwise or pointwise conv
        for i in range(10, -1, -1):  # Try blocks.10 down to blocks.0
            if hasattr(model, "blocks") and hasattr(model.blocks, str(i)):
                block = getattr(model.blocks, str(i))
                # Get the last conv layer in the block
                for module in reversed(list(block.modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        return module
        # Fallback: return the last conv in blocks.6
        if hasattr(model, "blocks") and hasattr(model.blocks, "6"):
            for module in reversed(list(model.blocks[6].modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return module
    
    # For ResNet models
    elif "resnet" in backbone.lower():
        if hasattr(model, "layer4"):
            # Get last conv in layer4
            for module in reversed(list(model.layer4.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return module
    
    # Generic fallback: find last Conv2d layer
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def visualize_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    backbone: str,
    target_class: Optional[int] = None,
    alpha: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate and overlay Grad-CAM heatmap on original image.
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed input tensor (1, C, H, W)
        original_image: Original image as numpy array (H, W, 3) in RGB
        backbone: Model backbone name
        target_class: Target class index (None = use predicted class)
        alpha: Transparency for overlay (0-1)
    
    Returns:
        Tuple of (heatmap, overlay_image)
    """
    # Get target layer
    target_layer = get_target_layer(model, backbone)
    if target_layer is None:
        raise ValueError(f"Could not find target layer for backbone: {backbone}")
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = gradcam.generate(input_tensor, target_class)
    
    # Resize heatmap to original image size
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Normalize original image to [0, 1] if needed
    if original_image.max() > 1.0:
        original_image = original_image.astype(np.float32) / 255.0
    
    # Overlay
    overlay = alpha * heatmap_colored.astype(np.float32) / 255.0 + (1 - alpha) * original_image
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    
    return heatmap_resized, overlay


def plot_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    backbone: str,
    class_names: list,
    true_label: Optional[int] = None,
    save_path: Optional[str] = None,
    target_class: Optional[int] = None
):
    """
    Create a visualization showing original image, Grad-CAM heatmap, and overlay.
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed input tensor (1, C, H, W)
        original_image: Original image as numpy array (H, W, 3) in RGB
        backbone: Model backbone name
        class_names: List of class names
        true_label: True label index (optional)
        save_path: Path to save figure (optional)
        target_class: Target class index (None = use predicted class)
    """
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(1).item()
        pred_prob = probs[0, pred_class].item()
    
    # Generate Grad-CAM
    heatmap, overlay = visualize_gradcam(
        model, input_tensor, original_image, backbone, target_class=target_class
    )
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    if true_label is not None:
        axes[0].set_xlabel(f"True: {class_names[true_label]}")
    axes[0].axis("off")
    
    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(overlay)
    title = f"Predicted: {class_names[pred_class]} ({pred_prob:.2%})"
    if target_class is not None and target_class != pred_class:
        title += f"\nVisualizing: {class_names[target_class]}"
    axes[2].set_title(title)
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Grad-CAM visualization to {save_path}")
    
    plt.close()



