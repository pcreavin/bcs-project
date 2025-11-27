#!/usr/bin/env python
"""Fix imports on GCP by updating __init__.py files if needed."""
import os
import sys
from pathlib import Path

def fix_losses_init():
    """Fix src/models/losses/__init__.py to export create_ordinal_loss."""
    losses_init = Path("src/models/losses/__init__.py")
    
    if not losses_init.exists():
        print(f"ERROR: {losses_init} not found!")
        return False
    
    print(f"Checking {losses_init}...")
    
    with open(losses_init, 'r') as f:
        content = f.read()
    
    # Check if create_ordinal_loss is already exported
    if 'create_ordinal_loss' in content:
        print("  ✓ create_ordinal_loss already exported")
        return True
    
    print("  ✗ create_ordinal_loss not exported, fixing...")
    
    # Update the file
    new_content = """\"\"\"Loss functions for training.\"\"\"
from .ordinal_loss import OrdinalLoss, create_ordinal_loss

__all__ = ["OrdinalLoss", "create_ordinal_loss"]
"""
    
    with open(losses_init, 'w') as f:
        f.write(new_content)
    
    print("  ✓ Fixed!")
    return True

def verify_losses_init():
    """Verify the __init__.py file is correct."""
    losses_init = Path("src/models/losses/__init__.py")
    
    if not losses_init.exists():
        print(f"ERROR: {losses_init} not found!")
        return False
    
    with open(losses_init, 'r') as f:
        content = f.read()
    
    checks = [
        ('from .ordinal_loss import', 'Import statement'),
        ('create_ordinal_loss', 'create_ordinal_loss export'),
        ('OrdinalLoss', 'OrdinalLoss export'),
    ]
    
    all_ok = True
    for check, desc in checks:
        if check in content:
            print(f"  ✓ {desc}: OK")
        else:
            print(f"  ✗ {desc}: MISSING")
            all_ok = False
    
    return all_ok

def test_import():
    """Test if the import works."""
    print("\nTesting import...")
    import sys
    from pathlib import Path
    
    # Add current directory to Python path
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    
    try:
        from src.models.losses import create_ordinal_loss
        print("  ✓ Import successful!")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Fixing GCP Imports")
    print("=" * 60)
    
    # Add current directory to Python path
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    
    print(f"\nCurrent directory: {cwd}")
    print(f"Python path includes: {cwd}")
    
    if not (cwd / "src" / "models").exists():
        print("ERROR: Must be in bcs-project root directory!")
        sys.exit(1)
    
    # Verify first
    print("\n1. Verifying current state...")
    if verify_losses_init():
        print("   File looks correct, but import might still fail.")
        print("   Checking if import works...")
        if test_import():
            print("\n✓ Everything is already correct!")
            sys.exit(0)
    
    # Fix if needed
    print("\n2. Fixing __init__.py...")
    if fix_losses_init():
        print("   ✓ File updated")
    else:
        print("   ✗ Failed to fix file")
        sys.exit(1)
    
    # Test import
    print("\n3. Testing import after fix...")
    if test_import():
        print("\n" + "=" * 60)
        print("✓ SUCCESS! Imports should work now.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ Import still failing. Check error above.")
        print("=" * 60)
        sys.exit(1)

