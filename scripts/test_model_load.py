"""Quick test to verify models can be loaded correctly."""
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import create_model

def test_model(model_name: str):
    """Test if a model can be created and loaded."""
    print(f"\nTesting {model_name}...")
    print("-" * 60)
    
    try:
        device = torch.device('cpu')  # Use CPU for quick test
        model = create_model(
            backbone=model_name,
            num_classes=5,
            pretrained=True,
            finetune_mode='full'
        )
        model.to(device)
        model.eval()
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"[OK] Model loaded successfully")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: (1, 5)")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    models = [
        "convnext_tiny",
        "swin_tiny_patch4_window7_224",
        "regnety_008"
    ]
    
    print("=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)
    
    results = {}
    for model_name in models:
        results[model_name] = test_model(model_name)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{model_name:35s} {status}")
    print("=" * 60)

