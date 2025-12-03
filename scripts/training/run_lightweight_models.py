"""Run lightweight model experiments: ConvNeXt-Tiny, Swin-T, RegNetY-8GF.

Usage:
  python scripts/run_lightweight_models.py --model convnext_tiny
  python scripts/run_lightweight_models.py --model all
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = {
    "convnext_tiny": "configs/convnext_tiny.yaml",
    "swin_tiny": "configs/swin_tiny.yaml",
    "regnety_8gf": "configs/regnety_8gf.yaml",
}


def run_training(config_path: str, model_name: str):
    """Run training for a single model."""
    print("=" * 80)
    print(f"Training {model_name.upper()}")
    print("=" * 80)
    print(f"Config: {config_path}")
    print()
    
    cmd = [
        sys.executable, "-m", "src.train.train",
        "--config", config_path
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print(f"\n[SUCCESS] {model_name} training completed successfully!")
    else:
        print(f"\n[FAILED] {model_name} training failed with exit code {result.returncode}")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run lightweight model experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model to train (or 'all' for all models)"
    )
    args = parser.parse_args()
    
    if args.model == "all":
        results = {}
        for model_name, config_path in MODELS.items():
            success = run_training(config_path, model_name)
            results[model_name] = success
            print("\n" + "=" * 80 + "\n")
        
        # Summary
        print("=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        for model_name, success in results.items():
            status = "[SUCCESS]" if success else "[FAILED]"
            print(f"{model_name:20s} {status}")
        print("=" * 80)
    else:
        config_path = MODELS[args.model]
        run_training(config_path, args.model)


if __name__ == "__main__":
    main()

