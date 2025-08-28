#!/usr/bin/env python3
"""
Generate samples from trained models with improved settings.
Outputs to organized directories for inspection.
"""

import torch
import yaml
from pathlib import Path
from torchvision.utils import save_image
import argparse
from datetime import datetime

from experiment import UNetTiny, DiTTiny, MMDiTTiny, ddim_sampler, rf_sampler, TargetMode
from train import MNISTLabelEmbedder

def generate_from_checkpoint(checkpoint_path, output_dir="outputs/samples"):
    """Generate high-quality samples from a trained model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(output_dir)
    model_name = Path(checkpoint_path).parts[-3]  # Get model name from path
    output_path = output_dir / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    epoch = checkpoint.get('epoch', 'unknown')
    
    # Build model based on architecture
    arch = config['model']['architecture']
    if arch == 'unet':
        model = UNetTiny(
            in_ch=config['model']['in_channels'],
            base=config['model'].get('base_channels', 64),
            emb_dim=config['model'].get('embed_dim', 256),
            out_ch=config['model']['out_channels']
        )
    elif arch == 'dit':
        model = DiTTiny(
            in_ch=config['model']['in_channels'],
            embed_dim=config['model'].get('embed_dim', 256),
            depth=config['model'].get('depth', 6),
            patch=config['model'].get('patch_size', 2),
            out_ch=config['model']['out_channels']
        )
    elif arch == 'mmdit':
        model = MMDiTTiny(
            in_ch=config['model']['in_channels'],
            img_dim=config['model'].get('embed_dim', 96),
            txt_dim=config['model'].get('text_dim', 96),
            depth=config['model'].get('depth', 3),
            patch=config['model'].get('patch_size', 2),
            out_ch=config['model']['out_channels']
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup label embedder for conditional generation
    label_embedder = None
    if config['model'].get('use_conditioning', False):
        if arch == 'mmdit':
            embed_dim = config['model'].get('text_dim', 96)
        else:
            embed_dim = config['model'].get('embed_dim', 256)
        
        label_embedder = MNISTLabelEmbedder(num_classes=10, embed_dim=embed_dim).to(device)
        if 'label_embedder_state_dict' in checkpoint:
            label_embedder.load_state_dict(checkpoint['label_embedder_state_dict'])
    
    # Get sampling parameters from config
    steps = config['sampling']['steps']
    cfg_scale = config['sampling']['cfg_scale']
    
    print(f"✅ Model: {arch} | Target: {config['training']['target']} | Epoch: {epoch}")
    print(f"   Steps: {steps} | CFG: {cfg_scale}")
    
    with torch.no_grad():
        all_samples = []
        
        if label_embedder is not None:
            # Generate conditional samples (all digits)
            print("🎨 Generating conditional samples (0-9)...")
            
            for digit in range(10):
                batch_size = 5  # 5 samples per digit
                shape = (batch_size, 
                        config['model']['in_channels'],
                        config['data']['image_size'],
                        config['data']['image_size'])
                
                labels = torch.tensor([digit] * batch_size, device=device)
                
                if arch == 'mmdit':
                    label_embeds = label_embedder(labels)
                    cond = label_embeds.unsqueeze(1)
                else:
                    cond = label_embedder(labels)
                
                # Choose sampler based on target
                if config['training']['target'] == 'flow':
                    samples = rf_sampler(
                        model, shape,
                        steps=steps,
                        cond=cond,
                        device=device,
                        cfg_scale=cfg_scale
                    )
                else:
                    samples = ddim_sampler(
                        model, shape,
                        steps=steps,
                        cond=cond,
                        target=config['training']['target'],
                        device=device,
                        cfg_scale=cfg_scale
                    )
                
                all_samples.append(samples)
            
            # Create grid
            all_samples = torch.cat(all_samples, dim=0)
            save_image(
                all_samples,
                output_path / "conditional_grid.png",
                nrow=10,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"   ✅ Saved: {output_path}/conditional_grid.png")
            
        else:
            # Generate unconditional samples
            print("🎨 Generating unconditional samples...")
            
            shape = (64,
                    config['model']['in_channels'],
                    config['data']['image_size'],
                    config['data']['image_size'])
            
            if config['training']['target'] == 'flow':
                all_samples = rf_sampler(
                    model, shape,
                    steps=steps,
                    device=device
                )
            else:
                all_samples = ddim_sampler(
                    model, shape,
                    steps=steps,
                    target=config['training']['target'],
                    device=device
                )
            
            save_image(
                all_samples,
                output_path / "unconditional_grid.png",
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"   ✅ Saved: {output_path}/unconditional_grid.png")
        
        # Also save individual samples
        save_image(
            all_samples[:16],  # First 16 samples
            output_path / "samples_16.png",
            nrow=4,
            normalize=True,
            value_range=(-1, 1)
        )
        
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained models')
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint')
    parser.add_argument('--all', action='store_true', help='Generate from all available models')
    parser.add_argument('--output_dir', type=str, default='outputs/samples',
                       help='Output directory for samples')
    args = parser.parse_args()
    
    print("\n🎨 SAMPLE GENERATION")
    print("="*60)
    
    if args.all or not args.checkpoint:
        # Generate from all available models
        experiments_dir = Path('experiments')
        
        # Find all best models
        checkpoints = list(experiments_dir.glob('*/checkpoints/best_model.pt'))
        checkpoints.extend(experiments_dir.glob('unified_*/checkpoints/best_model.pt'))
        
        if not checkpoints:
            print("❌ No trained models found!")
            print("Train models first with: python train_all.py")
            return
        
        print(f"Found {len(checkpoints)} trained models\n")
        
        for ckpt in checkpoints:
            try:
                generate_from_checkpoint(str(ckpt), args.output_dir)
                print()
            except Exception as e:
                print(f"⚠️  Error with {ckpt}: {e}\n")
                
    else:
        # Generate from specific checkpoint
        generate_from_checkpoint(args.checkpoint, args.output_dir)
    
    print("="*60)
    print(f"✅ Generation complete! Check: {args.output_dir}/")
    print("\nGenerated files:")
    output_path = Path(args.output_dir)
    for model_dir in sorted(output_path.iterdir()):
        if model_dir.is_dir():
            print(f"  {model_dir.name}/")
            for img in sorted(model_dir.glob("*.png")):
                print(f"    - {img.name}")

if __name__ == "__main__":
    main()