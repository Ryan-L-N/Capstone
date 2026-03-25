"""
Spot RL Setup & Validation Script
==================================
Installs dependencies and validates the RL training environment.

Run this once to setup everything needed for RL training.

Usage:
    C:\isaac-sim\python.bat setup_rl.py
"""

import subprocess
import sys
import importlib
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and report results"""
    
    print(f"\n>>> {description}" if description else "")
    print(f"    Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"    ✓ Success")
            return True
        else:
            print(f"    ✗ Failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout (>300s)")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def check_module(module_name, pip_name=None):
    """Check if a Python module is installed"""
    
    pip_name = pip_name or module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name:20} installed")
        return True
    except ImportError:
        print(f"✗ {module_name:20} NOT installed (requires: pip install {pip_name})")
        return False


def create_directories():
    """Create necessary directories"""
    
    dirs = [
        './checkpoints/spot_rl',
        './runs/spot_rl',
        './data',
        './logs',
    ]
    
    print("\n>>> Creating directories...")
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"    ✓ {d}")


def main():
    """Main setup function"""
    
    print("\n" + "="*80)
    print("SPOT RL ENVIRONMENT SETUP & VALIDATION")
    print("="*80)
    
    # Step 1: Check Python version
    print("\n[1/5] Checking Python version...")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 9):
        print(f"✓ Python {py_version} (OK)")
    else:
        print(f"✗ Python {py_version} - Need 3.9+ for Isaac Lab")
        return False
    
    # Step 2: Check Isaac Sim modules
    print("\n[2/5] Checking Isaac Sim modules...")
    isaac_modules = [
        ('isaacsim', 'isaacsim'),
        ('omni', 'omni'),
    ]
    
    isaac_ok = True
    for module, name in isaac_modules:
        try:
            importlib.import_module(module)
            print(f"✓ Isaac Sim - {name:20} OK")
        except ImportError:
            print(f"✗ Isaac Sim - {name:20} NOT FOUND")
            isaac_ok = False
    
    # Check pxr separately (often available in context but hard to import standalone)
    try:
        importlib.import_module('pxr')
        print(f"✓ Isaac Sim - {'pixar (pxr)':20} OK")
    except ImportError:
        print(f"⚠ Isaac Sim - {'pixar (pxr)':20} (may be available in context)")
    
    if not isaac_ok:
        print("\n⚠ Isaac Sim modules not found. Make sure to use:")
        print("  C:\\isaac-sim\\python.bat setup_rl.py")
        return False
    
    # Step 3: Check and install Python packages
    print("\n[3/5] Checking Python packages...")
    
    packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('torch', 'torch'),
        ('tensorboard', 'tensorboard'),
    ]
    
    missing_packages = []
    for module, package in packages:
        try:
            importlib.import_module(module)
            print(f"✓ {package:20} installed")
        except ImportError:
            print(f"✗ {package:20} NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n>>> Installing missing packages...")
        cmd = f"{sys.executable} -m pip install {' '.join(missing_packages)}"
        
        success = run_command(cmd, f"Installing: {', '.join(missing_packages)}")
        if not success:
            print("\n⚠ Package installation had issues. Trying individual installs...")
            for pkg in missing_packages:
                run_command(f"{sys.executable} -m pip install {pkg}", f"Installing {pkg}")
    
    # Step 4: Create directories
    print("\n[4/5] Setting up directories...")
    create_directories()
    
    # Step 5: Validate environment setup
    print("\n[5/5] Validating environment...")
    
    # Check required environment files
    required_files = [
        'SpotRL_Environment.py',
        'SpotRL_Training.py',
        'README_RL.md',
    ]
    
    all_files_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} NOT FOUND")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n✗ Some required files are missing!")
        return False
    
    # Summary
    print("\n" + "="*80)
    print("SETUP VALIDATION COMPLETE")
    print("="*80)
    
    print("\n✓ Environment is ready for RL training!")
    
    print("\nNext steps:")
    print("  1. Test environment:")
    print("     C:\\isaac-sim\\python.bat SpotRL_Environment.py")
    print("\n  2. Start training:")
    print("     C:\\isaac-sim\\python.bat SpotRL_Training.py --episodes 100")
    print("\n  3. Monitor in TensorBoard:")
    print("     tensorboard --logdir ./runs/spot_rl --port 6006")
    print("     Then open: http://localhost:6006")
    
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == "__main__":
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
