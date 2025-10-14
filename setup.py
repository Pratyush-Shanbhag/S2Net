#!/usr/bin/env python3
"""
Setup script for S2Net: Stochastic Sequential Pointcloud Forecasting
"""

import os
import sys
import subprocess
import torch


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")


def check_cuda():
    """Check CUDA availability."""
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. The model will run on CPU.")
    print(f"PyTorch version: {torch.__version__}")


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)


def create_directories():
    """Create necessary directories."""
    directories = [
        "checkpoints",
        "logs",
        "evaluation_results",
        "data",
        "configs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    try:
        from models import S2Net
        from data import create_dataloader
        from utils import compute_metrics
        print("All imports successful!")
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)


def run_demo():
    """Run the demo script."""
    print("Running demo...")
    try:
        subprocess.check_call([sys.executable, "demo.py"])
        print("Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Demo failed: {e}")
        sys.exit(1)


def main():
    """Main setup function."""
    print("S2Net: Stochastic Sequential Pointcloud Forecasting - Setup")
    print("="*60)
    
    # Check Python version
    check_python_version()
    
    # Check CUDA
    check_cuda()
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Test imports
    test_imports()
    
    # Run demo
    run_demo()
    
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("\nYou can now:")
    print("1. Run 'python demo.py' to see a demonstration")
    print("2. Run 'python train.py --config configs/default.yaml' to train the model")
    print("3. Run 'python evaluate.py --model_path checkpoints/best_model.pth' to evaluate")
    print("\nFor more information, see README.md")


if __name__ == '__main__':
    main()
