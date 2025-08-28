#!/usr/bin/env python3
"""
Setup script for MNIST diffusion experiments.
Downloads MNIST data and creates necessary directories.
"""

import os
from pathlib import Path
import torch
from torchvision import datasets
import argparse


def setup_mnist_experiments():
    """Set up directories and download MNIST dataset."""
    
    print("🚀 Setting up MNIST diffusion experiments...")
    
    # Create base directories
    base_dir = Path(".")
    data_dir = base_dir / "data" / "mnist"
    experiments_dir = base_dir / "experiments_mnist"
    
    # Create experiment subdirectories
    experiment_configs = [
        "ddpm_unet",
        "v_unet", 
        "x0_unet",
        "dit",
        "flow_unet",
        "mmdit_flow"
    ]
    
    print("\n📁 Creating directories...")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for exp_name in experiment_configs:
        exp_path = experiments_dir / exp_name
        exp_path.mkdir(parents=True, exist_ok=True)
        (exp_path / "checkpoints").mkdir(exist_ok=True)
        (exp_path / "samples").mkdir(exist_ok=True)
        (exp_path / "tensorboard").mkdir(exist_ok=True)
        print(f"  ✓ Created {exp_path}")
    
    # Download MNIST dataset
    print("\n📥 Downloading MNIST dataset...")
    try:
        # Download training set
        train_dataset = datasets.MNIST(
            root=str(data_dir),
            train=True,
            download=True
        )
        print(f"  ✓ Training set: {len(train_dataset)} samples")
        
        # Download test set
        test_dataset = datasets.MNIST(
            root=str(data_dir),
            train=False,
            download=True
        )
        print(f"  ✓ Test set: {len(test_dataset)} samples")
        
    except Exception as e:
        print(f"  ✗ Error downloading MNIST: {e}")
        return False
    
    # Test GPU availability
    print("\n🖥️  System check:")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠️  No CUDA available, will use CPU (training will be slower)")
    
    print(f"  ✓ PyTorch version: {torch.__version__}")
    
    # Create a simple test script
    test_script = base_dir / "test_setup.py"
    test_content = '''#!/usr/bin/env python3
"""Quick test to verify setup."""
import torch
from mnist_dataset import MNISTDiffusionDataset
from experiment import UNetTiny

# Test dataset
dataset = MNISTDiffusionDataset(channels=1)
print(f"Dataset loaded: {len(dataset)} samples")

# Test model
model = UNetTiny(in_ch=1, out_ch=1)
x = torch.randn(2, 1, 32, 32)
t = torch.rand(2)
out = model(x, t)
print(f"Model forward pass: input {x.shape} -> output {out.shape}")

print("✓ Setup verified successfully!")
'''
    
    with open(test_script, 'w') as f:
        f.write(test_content)
    os.chmod(test_script, 0o755)
    
    print(f"\n✅ Setup complete!")
    print(f"\n📝 Next steps:")
    print(f"  1. Run './test_setup.py' to verify everything works")
    print(f"  2. Train a model: python train.py --config configs/mnist_ddpm_unet.yaml")
    print(f"  3. Or train all models: python train_all_mnist.py")
    
    # Create a summary file
    summary_file = experiments_dir / "README.md"
    summary_content = '''# MNIST Diffusion Experiments

## Experiment Configurations

| Config | Architecture | Objective | Description |
|--------|-------------|-----------|-------------|
| ddpm_unet | UNet | Epsilon (noise) | Classic DDPM baseline |
| v_unet | UNet | V-prediction | Alternative parameterization |
| x0_unet | UNet | X0 (clean image) | Direct prediction |
| dit | DiT | Epsilon | Vision Transformer |
| flow_unet | UNet | Flow matching | Rectified flow/velocity |
| mmdit_flow | MMDiT | Flow matching | Multi-modal with conditioning |

## Training Commands

```bash
# Train individual model
python train.py --config configs/mnist_ddpm_unet.yaml

# Train all models
python train_all_mnist.py

# Resume training
python train.py --config configs/mnist_ddpm_unet.yaml --resume experiments_mnist/ddpm_unet/checkpoints/latest_model.pt
```

## Sampling Commands

```bash
# Generate samples
python sample.py --checkpoint experiments_mnist/ddpm_unet/checkpoints/best_model.pt --num_samples 64

# Compare all models
python compare_models.py
```

## Results

Results will be saved in each experiment's directory:
- `checkpoints/`: Model checkpoints
- `samples/`: Generated samples during training
- `tensorboard/`: Training metrics
'''
    
    with open(summary_file, 'w') as f:
        f.write(summary_content)
    
    print(f"\n📖 Experiment guide saved to: {summary_file}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup MNIST diffusion experiments")
    args = parser.parse_args()
    
    success = setup_mnist_experiments()
    exit(0 if success else 1)