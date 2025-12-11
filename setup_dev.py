#!/usr/bin/env python3
"""
Equilibrium-3B Development Environment Setup
============================================

Automated setup script for development environment with all dependencies,
pre-commit hooks, and development tools.

Usage:
    python setup_dev.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run command with error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_python_version():
    """Check Python version >= 3.10."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. Current: {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def check_cuda():
    """Check CUDA availability."""
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print("🔍 GPU Information:")
            gpu_info = result.stdout.split('\n')[9:12]  # GPU info lines
            for line in gpu_info:
                if line.strip():
                    print(f"   {line.strip()}")
            return True
        else:
            print("⚠️  No NVIDIA GPU detected - CPU-only mode")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - CPU-only mode")
        return False

def setup_environment():
    """Setup virtual environment and dependencies."""
    
    # Create virtual environment if not exists
    if not Path("venv").exists():
        run_command(f"{sys.executable} -m venv venv", "Creating virtual environment")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch with CUDA if available
    cuda_available = check_cuda()
    if cuda_available:
        torch_cmd = f"{pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        run_command(torch_cmd, "Installing PyTorch with CUDA 12.1")
    else:
        torch_cmd = f"{pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        run_command(torch_cmd, "Installing PyTorch CPU-only")
    
    # Install requirements
    run_command(f"{pip_cmd} install -r requirements.txt", "Installing production requirements")
    run_command(f"{pip_cmd} install -r requirements-dev.txt", "Installing development requirements")
    
    # Install package in development mode
    run_command(f"{pip_cmd} install -e .", "Installing Equilibrium-3B in development mode")
    
    return python_cmd, pip_cmd

def setup_pre_commit(pip_cmd):
    """Setup pre-commit hooks."""
    # Create .pre-commit-config.yaml
    precommit_config = """repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
"""
    
    with open(".pre-commit-config.yaml", "w") as f:
        f.write(precommit_config)
    
    run_command(f"{pip_cmd} install pre-commit", "Installing pre-commit")
    run_command("pre-commit install", "Installing pre-commit hooks")

def create_sample_configs():
    """Create sample configuration files."""
    
    # Sample training config
    sample_config = """# Sample local development config
model:
  name: "Equilibrium-3B-Dev"
  hidden_size: 1280  # Smaller for dev
  num_layers: 12     # Reduced for faster training
  num_attention_layers: 2
  num_experts: 32    # Reduced experts

training:
  max_steps: 1000    # Quick dev training
  batch_size: 4      # Small batch for dev
  eval_interval: 100
  
data:
  total_tokens: 10000000  # 10M tokens for dev
  
optimization:
  muon_lr: 0.01
  soap_lr: 0.005
"""
    
    with open("configs/dev_config.yaml", "w") as f:
        f.write(sample_config)
    
    print("✅ Created sample dev config: configs/dev_config.yaml")

def verify_installation(python_cmd):
    """Verify installation by importing key components."""
    test_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Test key imports
try:
    import mamba_ssm
    print("✅ Mamba SSM installed")
except ImportError:
    print("⚠️  Mamba SSM not available")

try:
    import triton
    print("✅ Triton installed")
except ImportError:
    print("⚠️  Triton not available")

print("🎉 Installation verification complete!")
"""
    
    with open("test_install.py", "w") as f:
        f.write(test_script)
    
    run_command(f"{python_cmd} test_install.py", "Verifying installation")
    os.remove("test_install.py")

def main():
    """Main setup process."""
    print("🚀 Setting up Equilibrium-3B Development Environment")
    print("=" * 60)
    
    # Check requirements
    check_python_version()
    
    # Setup environment
    python_cmd, pip_cmd = setup_environment()
    
    # Setup development tools
    setup_pre_commit(pip_cmd)
    
    # Create sample configs
    create_sample_configs()
    
    # Verify installation
    verify_installation(python_cmd)
    
    print("\n" + "=" * 60)
    print("🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("1. Activate environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Start development:")
    print("   python -c 'import src; print(\"Equilibrium-3B ready!\")'")
    print("3. Run tests:")
    print("   pytest tests/")
    print("4. Start training:")
    print("   python training/pretrain.py --config configs/dev_config.yaml")

if __name__ == "__main__":
    main()