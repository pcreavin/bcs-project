"""Quick script to check available devices and suggest optimizations."""
import torch

print("=" * 60)
print("DEVICE CHECK")
print("=" * 60)

cuda_available = torch.cuda.is_available()
mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"MPS available: {mps_available}")

if cuda_available:
    device = "cuda"
elif mps_available:
    device = "mps"
else:
    device = "cpu"

print(f"\nWill use device: {device}")

if device == "cpu":
    print("\n⚠️  WARNING: Training on CPU is VERY slow!")
    print("   Consider using a GPU-enabled environment for faster training.")
else:
    print(f"\n✓ Using {device.upper()} - training should be reasonably fast")

print("\n" + "=" * 60)

