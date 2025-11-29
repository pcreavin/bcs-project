"""Grad-CAM visualization for model interpretability."""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class GradCAM:
    """Grad-CAM implementation for visualizing model attention."""
    
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer to compute gradients from (if None, auto-detect)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        if self.target_layer is None:
            # Auto-detect the last convolutional layer for EfficientNet
            self.target_layer = self._find_target_layer()
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def _find_target_layer(self):
        """Find the last convolutional layer in the model."""
        # For EfficientNet from timm, the last conv layer is typically in blocks
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                # Prefer a layer that's not too early (skip first conv)
                if 'blocks' in name or 'conv_head' in name or 'bn2' in name:
                    return module
        # Fallback: return last Conv2d found
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("Could not find a convolutional layer in the model")
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            CAM heatmap as numpy array (H, W)
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        score = output[0, target_class]
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()  # (C, H, W)
        activations = self.activations[0].cpu().data.numpy()  # (C, H, W)
        
        # Compute weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def overlay_heatmap(self, image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H, W, 3) in RGB format, uint8
            cam: CAM heatmap (H, W) normalized to [0, 1]
            alpha: Transparency factor for heatmap overlay
        
        Returns:
            Overlaid image (H, W, 3) in RGB format, uint8
        """
        # Resize CAM to match image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert to heatmap colormap
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]  # (H, W, 3) in [0, 1]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Overlay
        overlaid = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        
        return overlaid


def visualize_gradcam(
    model: torch.nn.Module,
    image: np.ndarray,
    class_names: list,
    device: str = "cpu",
    save_path: Optional[str] = None,
    target_class: Optional[int] = None
) -> Tuple[np.ndarray, int, float]:
    """
    Generate and visualize Grad-CAM for a single image.
    
    Args:
        model: PyTorch model
        image: Input image (H, W, 3) in RGB format, uint8
        class_names: List of class names
        device: Device to run inference on
        save_path: Optional path to save visualization
        target_class: Target class for CAM (if None, use predicted class)
    
    Returns:
        Tuple of (overlaid_image, predicted_class, confidence)
    """
    # Preprocess image
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2
    
    transform = Compose([
        Resize(224, 224),
        Normalize(),
        ToTensorV2()
    ])
    
    # Store original image for overlay
    original_image = image.copy()
    
    # Transform
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Generate CAM
    gradcam = GradCAM(model, target_layer=None)
    cam = gradcam.generate_cam(input_tensor, target_class=target_class)
    
    # Resize original image to model input size for overlay
    original_resized = cv2.resize(original_image, (224, 224))
    overlaid = gradcam.overlay_heatmap(original_resized, cam, alpha=0.4)
    
    # Create visualization
    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_resized)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Heatmap
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        # Overlaid
        axes[2].imshow(overlaid)
        pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
        axes[2].set_title(f"Overlay\nPredicted: {pred_name} ({confidence:.2%})")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved Grad-CAM visualization to {save_path}")
    
    return overlaid, pred_class, confidence

