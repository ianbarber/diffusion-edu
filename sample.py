#!/usr/bin/env python3
"""
Sampling/inference script for trained diffusion models.
Generates samples from checkpoints with various sampling methods.
"""

import os
import argparse
from pathlib import Path
import torch
import yaml
from PIL import Image
import numpy as np
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from experiment import (
    UNetTiny, DiTTiny, MMDiTTiny, TinyVAE, TinyTextToker,
    ddim_sampler, rf_sampler, TargetMode
)


class Sampler:
    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)  # Set to False for compatibility with config dict
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
        
        # Build model
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup VAE if needed
        self.vae = None
        if self.config['model'].get('use_latents', False):
            self.vae = TinyVAE(
                in_ch=3,
                latent_ch=self.config['model']['in_channels']
            ).to(self.device)
            self.vae.eval()
        
        # Setup text encoder if needed
        self.text_encoder = None
        if self.config['model'].get('use_conditioning', False):
            self.text_encoder = TinyTextToker(
                vocab=self.config['model'].get('vocab_size', 1000),
                dim=self.config['model']['embed_dim']
            ).to(self.device)
            self.text_encoder.eval()
        
        print(f"Loaded model from checkpoint: {checkpoint_path}")
        print(f"Model architecture: {self.config['model']['architecture']}")
        print(f"Training target: {self.config['training']['target']}")
        
    def build_model(self):
        """Build model based on config."""
        arch = self.config['model']['architecture']
        in_ch = self.config['model']['in_channels']
        out_ch = self.config['model']['out_channels']
        
        if arch == 'unet':
            model = UNetTiny(
                in_ch=in_ch,
                base=self.config['model'].get('base_channels', 64),
                emb_dim=self.config['model'].get('embed_dim', 256),
                out_ch=out_ch
            )
        elif arch == 'dit':
            model = DiTTiny(
                in_ch=in_ch,
                embed_dim=self.config['model'].get('embed_dim', 256),
                depth=self.config['model'].get('depth', 6),
                patch=self.config['model'].get('patch_size', 2),
                out_ch=out_ch
            )
        elif arch == 'mmdit':
            model = MMDiTTiny(
                in_ch=in_ch,
                img_dim=self.config['model'].get('embed_dim', 256),
                txt_dim=self.config['model'].get('text_dim', 256),
                depth=self.config['model'].get('depth', 6),
                patch=self.config['model'].get('patch_size', 2),
                out_ch=out_ch
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        return model.to(self.device)
    
    @torch.no_grad()
    def sample(self, 
               num_samples=1, 
               steps=50, 
               text_prompt=None,
               cfg_scale=1.0,
               seed=None):
        """Generate samples from the model."""
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Determine shape
        if self.config['model'].get('use_latents', False):
            h = w = self.config['data']['latent_size']
        else:
            h = w = self.config['data']['image_size'] // 8
        
        shape = (
            num_samples,
            self.config['model']['in_channels'],
            h, w
        )
        
        # Get conditioning
        cond = None
        if self.config['model'].get('use_conditioning', False) and self.text_encoder is not None:
            if text_prompt:
                # In a real implementation, you'd tokenize the text_prompt
                # For now, we'll use random tokens as placeholder
                txt_ids = torch.randint(0, 1000, (num_samples, 32), device=self.device)
            else:
                txt_ids = torch.randint(0, 1000, (num_samples, 32), device=self.device)
            
            if self.config['model']['architecture'] == 'mmdit':
                cond = self.text_encoder(txt_ids)
            else:
                txt_tokens = self.text_encoder(txt_ids)
                cond = txt_tokens.mean(dim=1)
        
        # Sample based on training target
        target = self.config['training']['target']
        
        # Use cfg_scale from config or parameter
        if cfg_scale == 1.0:  # Default, check config
            cfg_scale = self.config.get('sampling', {}).get('cfg_scale', 1.0)
        
        if target == TargetMode.FLOW:
            samples = rf_sampler(
                self.model,
                shape,
                steps=steps,
                cond=cond,
                device=self.device,
                cfg_scale=cfg_scale
            )
        else:
            samples = ddim_sampler(
                self.model,
                shape,
                steps=steps,
                cond=cond,
                target=target,
                device=self.device,
                cfg_scale=cfg_scale
            )
        
        # Decode if using VAE
        if self.vae is not None:
            samples = self.vae.decode(samples)
        
        return samples
    
    def sample_batch(self,
                    num_batches=10,
                    batch_size=4,
                    steps=50,
                    output_dir='./samples',
                    save_individual=True,
                    save_grid=True):
        """Generate multiple batches of samples."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_samples = []
        
        print(f"Generating {num_batches} batches of {batch_size} samples each...")
        
        for batch_idx in tqdm(range(num_batches), desc="Sampling batches"):
            samples = self.sample(
                num_samples=batch_size,
                steps=steps,
                seed=batch_idx * 1000  # Different seed for each batch
            )
            
            all_samples.append(samples)
            
            if save_individual:
                for i in range(batch_size):
                    img_idx = batch_idx * batch_size + i
                    save_image(
                        samples[i],
                        output_dir / f"sample_{img_idx:05d}.png",
                        normalize=True,
                        value_range=(-1, 1)
                    )
        
        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)
        
        if save_grid:
            # Save as grid
            grid = make_grid(
                all_samples,
                nrow=int(np.sqrt(len(all_samples))),
                normalize=True,
                value_range=(-1, 1)
            )
            save_image(grid, output_dir / "sample_grid.png")
            
            # Also save a larger grid if we have many samples
            if len(all_samples) >= 64:
                grid_large = make_grid(
                    all_samples[:64],
                    nrow=8,
                    normalize=True,
                    value_range=(-1, 1)
                )
                save_image(grid_large, output_dir / "sample_grid_8x8.png")
        
        print(f"Saved {len(all_samples)} samples to {output_dir}")
        return all_samples
    
    def interpolate(self,
                   num_samples=2,
                   num_interp=10,
                   steps=50,
                   output_path='interpolation.png'):
        """Generate interpolated samples."""
        
        # Generate latent codes
        z1 = torch.randn(1, self.config['model']['in_channels'], 
                        self.config['data']['latent_size'], 
                        self.config['data']['latent_size'],
                        device=self.device)
        z2 = torch.randn(1, self.config['model']['in_channels'],
                        self.config['data']['latent_size'],
                        self.config['data']['latent_size'],
                        device=self.device)
        
        # Interpolate
        alphas = torch.linspace(0, 1, num_interp, device=self.device)
        interpolated = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Use z_interp as starting point for sampling
            # This is a simplified version - proper interpolation would
            # require modifying the sampler
            sample = self.sample(num_samples=1, steps=steps)
            interpolated.append(sample)
        
        # Save interpolation
        interpolated = torch.cat(interpolated, dim=0)
        save_image(
            interpolated,
            output_path,
            nrow=num_interp,
            normalize=True,
            value_range=(-1, 1)
        )
        
        print(f"Saved interpolation to {output_path}")
        return interpolated


def main():
    parser = argparse.ArgumentParser(description='Sample from trained diffusion models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for sampling')
    parser.add_argument('--steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--output_dir', type=str, default='./samples', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--interpolate', action='store_true', help='Generate interpolations')
    parser.add_argument('--text', type=str, default=None, help='Text prompt for conditional models')
    
    args = parser.parse_args()
    
    # Create sampler
    sampler = Sampler(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    if args.interpolate:
        # Generate interpolations
        sampler.interpolate(
            num_interp=10,
            steps=args.steps,
            output_path=Path(args.output_dir) / 'interpolation.png'
        )
    else:
        # Generate samples
        num_batches = args.num_samples // args.batch_size
        if args.num_samples % args.batch_size != 0:
            num_batches += 1
        
        sampler.sample_batch(
            num_batches=num_batches,
            batch_size=args.batch_size,
            steps=args.steps,
            output_dir=args.output_dir,
            save_individual=True,
            save_grid=True
        )


if __name__ == "__main__":
    main()