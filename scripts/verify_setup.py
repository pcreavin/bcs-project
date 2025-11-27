#!/usr/bin/env python
"""Simple verification that setup is correct for module-style imports."""
import sys
from pathlib import Path

print("=" * 60)
print("Verifying Setup for Module Imports")
print("=" * 60)

cwd = Path.cwd()
print(f"\nCurrent directory: {cwd}")

# Check file exists and has correct content
losses_init = cwd / "src/models/losses/__init__.py"
print(f"\n1. Checking {losses_init}...")

if not losses_init.exists():
    print(f"   ✗ File not found!")
    sys.exit(1)

with open(losses_init) as f:
    content = f.read()

if "create_ordinal_loss" in content:
    print("   ✓ File contains create_ordinal_loss export")
else:
    print("   ✗ File missing create_ordinal_loss export")
    sys.exit(1)

if "from .ordinal_loss import" in content:
    print("   ✓ File has correct import statement")
else:
    print("   ✗ File missing import statement")
    sys.exit(1)

# Check ordinal_loss.py exists
ordinal_loss_py = cwd / "src/models/losses/ordinal_loss.py"
print(f"\n2. Checking {ordinal_loss_py}...")

if ordinal_loss_py.exists():
    print("   ✓ ordinal_loss.py exists")
    with open(ordinal_loss_py) as f:
        if "def create_ordinal_loss" in f.read():
            print("   ✓ create_ordinal_loss function exists")
        else:
            print("   ✗ create_ordinal_loss function not found")
            sys.exit(1)
else:
    print("   ✗ ordinal_loss.py not found!")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ File structure looks correct!")
print("=" * 60)
print("\nNote: The import test might fail when running as a script,")
print("but training should work when run as a module:")
print("  python -m src.train.train --config configs/ordinal_b0_224_jitter.yaml")
print("\nIf training fails, check the actual error message.")

