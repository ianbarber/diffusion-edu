#!/usr/bin/env python3
"""
Comprehensive training and generation script for all diffusion models.
Trains unified models, reference models, and generates comparison outputs.
"""

import os
import subprocess
import time
from pathlib import Path
import argparse
import torch
import yaml
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

def run_command(cmd, description="", capture_output=False):
    """Run a shell command and return success status."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}" if description else f"Running: {cmd}")
    print(f"{'='*60}")
    
    try:
        if capture_output:
            # Capture output for commands that don't need real-time display
            result = subprocess.run(cmd, shell=True, check=True, 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")
        else:
            # Stream output in real-time for training (shows progress bars)
            result = subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if capture_output and hasattr(e, 'stdout'):
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
        return False

def train_model(config_path, description=""):
    """Train a single model with the given config."""
    cmd = f"python train.py --config {config_path}"
    
    config_name = Path(config_path).stem
    desc = description or f"Training {config_name}"
    
    start_time = time.time()
    # Use real-time output for training to show progress bars
    success = run_command(cmd, desc, capture_output=False)
    elapsed = time.time() - start_time
    
    if success:
        print(f"✅ Completed {config_name} in {elapsed/60:.1f} minutes")
    else:
        print(f"⚠️  Failed to train {config_name}")
    
    return success, elapsed

def generate_samples(checkpoint_path, output_dir, num_samples=64):
    """Generate samples from a trained model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = f"python sample.py --checkpoint {checkpoint_path} "
    cmd += f"--num_samples {num_samples} --output_dir {output_dir}"
    
    model_name = Path(checkpoint_path).parts[-3]  # Get model name from path
    # Capture output for sampling since it doesn't have progress bars
    return run_command(cmd, f"Generating samples for {model_name}", capture_output=True)

def train_all_models(mode='all', quick=False):
    """Train all models based on mode."""
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE DIFFUSION MODEL TRAINING")
    print("="*80)
    
    # Define training configurations
    unified_configs = [
        ('configs/unified_sd1_unet_eps.yaml', 'SD1-style: UNet + Epsilon'),
        ('configs/unified_sd2_unet_v.yaml', 'SD2-style: UNet + V-prediction'),
        ('configs/unified_sd3_mmdit_flow.yaml', 'SD3-style: MMDiT + Flow'),
    ]
    
    reference_configs = [
        ('configs/reference/ddpm_original.yaml', 'Original DDPM (2020)'),
        ('configs/reference/unet_x0_prediction.yaml', 'X0 Direct Prediction'),
    ]
    
    # Select configs based on mode
    configs_to_train = []
    if mode in ['all', 'unified']:
        configs_to_train.extend(unified_configs)
    if mode in ['all', 'reference']:
        configs_to_train.extend(reference_configs)
    
    # Quick mode note: epochs controlled by config files
    
    # Training summary
    results = {}
    total_start = time.time()
    
    print(f"\n📋 Training Plan:")
    print(f"  - Mode: {mode}")
    print(f"  - Quick: {quick} (epochs controlled by config)")
    print(f"  - Models to train: {len(configs_to_train)}")
    
    for config, desc in configs_to_train:
        print(f"    • {desc}")
    
    # Train each model
    print(f"\n🏃 Starting training sequence...")
    
    for i, (config_path, description) in enumerate(configs_to_train, 1):
        print(f"\n📊 Training {i}/{len(configs_to_train)}: {description}")
        
        success, elapsed = train_model(config_path, description)
        
        config_name = Path(config_path).stem
        results[config_name] = {
            'success': success,
            'time_minutes': elapsed / 60,
            'description': description
        }
        
        # Small break between models to avoid GPU thermal issues
        if i < len(configs_to_train):
            print("⏸️  Cooling down for 10 seconds...")
            time.sleep(10)
    
    # Summary
    total_time = (time.time() - total_start) / 60
    successful = sum(1 for r in results.values() if r['success'])
    
    print("\n" + "="*80)
    print("📊 TRAINING SUMMARY")
    print("="*80)
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Successful: {successful}/{len(configs_to_train)}")
    
    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {name}: {result['time_minutes']:.1f} min")
    
    # Save results
    results_path = Path('training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {results_path}")
    
    return results

def generate_all_outputs():
    """Generate samples and comparisons for all trained models."""
    
    print("\n" + "="*80)
    print("🎨 GENERATING MODEL OUTPUTS")
    print("="*80)
    
    # Find all trained models
    experiments_dir = Path('experiments')
    checkpoints = []
    
    # Look for checkpoints in both unified and reference directories
    for pattern in ['*/checkpoints/model_epoch_*.pt', 
                   'unified_*/checkpoints/model_epoch_*.pt',
                   'reference/*/checkpoints/model_epoch_*.pt']:
        checkpoints.extend(experiments_dir.glob(pattern))
    
    # Also check for best_model.pt files
    for pattern in ['*/checkpoints/best_model.pt',
                   'unified_*/checkpoints/best_model.pt',
                   'reference/*/checkpoints/best_model.pt']:
        checkpoints.extend(experiments_dir.glob(pattern))
    
    # Get unique model directories
    model_dirs = set(ckpt.parent.parent for ckpt in checkpoints)
    
    print(f"Found {len(model_dirs)} trained models")
    
    # Generate samples for each model
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        # Find best checkpoint (prefer best_model.pt, then latest epoch)
        best_ckpt = model_dir / 'checkpoints' / 'best_model.pt'
        if not best_ckpt.exists():
            epoch_ckpts = sorted(model_dir.glob('checkpoints/model_epoch_*.pt'))
            if epoch_ckpts:
                best_ckpt = epoch_ckpts[-1]
            else:
                print(f"⚠️  No checkpoint found for {model_name}")
                continue
        
        print(f"\n🎨 Generating samples for {model_name}")
        output_dir = Path('outputs/samples') / model_name
        generate_samples(best_ckpt, output_dir, num_samples=64)
    
    # Run comparison script
    print("\n📊 Generating model comparison...")
    run_command("python compare_models.py", "Creating comparison visualizations", capture_output=True)
    
    print("\n✅ Output generation complete!")
    print("  - Samples: outputs/samples/")
    print("  - Comparisons: outputs/comparisons/")

def main():
    parser = argparse.ArgumentParser(description='Train all diffusion models')
    parser.add_argument('--mode', choices=['all', 'unified', 'reference'],
                       default='all', help='Which models to train')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with fewer epochs (10)')
    parser.add_argument('--generate-only', action='store_true',
                       help='Only generate outputs, skip training')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip output generation after training')
    args = parser.parse_args()
    
    if not args.generate_only:
        # Train models
        results = train_all_models(mode=args.mode, quick=args.quick)
        
        # Check if any models were successfully trained
        if not any(r['success'] for r in results.values()):
            print("\n⚠️  No models were successfully trained. Skipping generation.")
            return
    
    if not args.skip_generation:
        # Generate outputs
        generate_all_outputs()
    
    print("\n🎉 All done! Your models are ready.")
    print("\nNext steps:")
    print("  1. View samples: outputs/samples/")
    print("  2. Try interactive demo: python interactive_demo.py --checkpoint <path>")
    print("  3. Compare models: view outputs/comparisons/comparison.png")

if __name__ == "__main__":
    main()