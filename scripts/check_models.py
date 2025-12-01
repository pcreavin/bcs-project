"""Quick script to verify timm model names exist."""
import timm

models_to_check = {
    "convnext_tiny": "configs/convnext_tiny.yaml",
    "swin_tiny_patch4_window7_224": "configs/swin_tiny.yaml",
    "regnety_008": "configs/regnety_8gf.yaml",
}

print("Checking timm model availability...")
print("=" * 60)

all_models = timm.list_models()

for model_name, config_file in models_to_check.items():
    exists = model_name in all_models
    status = "[OK]" if exists else "[NOT FOUND]"
    print(f"{model_name:35s} {status}")
    
    if not exists:
        # Try to find similar names
        similar = [m for m in all_models if model_name.split('_')[0] in m.lower()]
        if similar:
            print(f"  Similar models: {similar[:3]}")

print("=" * 60)

