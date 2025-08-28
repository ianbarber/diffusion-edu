#!/usr/bin/env python3
"""
Interactive demo for generating MNIST digits with trained diffusion models.
"""

import os
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import yaml

from experiment import (
    UNetTiny, DiTTiny, MMDiTTiny,
    ddim_sampler, rf_sampler, TargetMode
)
from train import MNISTLabelEmbedder


class InteractiveDemo:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)  # Set to False for config dict compatibility
        self.config = checkpoint['config']
        
        # Build model
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load label embedder if needed
        self.label_embedder = None
        if self.config['model'].get('use_conditioning', False):
            embed_dim = self.config['model'].get('text_dim', self.config['model']['embed_dim'])
            self.label_embedder = MNISTLabelEmbedder(num_classes=10, embed_dim=embed_dim).to(self.device)
            if 'label_embedder_state_dict' in checkpoint:
                self.label_embedder.load_state_dict(checkpoint['label_embedder_state_dict'])
        
        print(f"✅ Loaded {self.config['model']['architecture']} model with {self.config['training']['target']} objective")
        print(f"   Device: {self.device}")
        print(f"   Conditional: {self.config['model'].get('use_conditioning', False)}")
    
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
        
        return model.to(self.device)
    
    @torch.no_grad()
    def generate(self, digit=None, num_samples=1, steps=50, seed=None):
        """Generate specific digit(s)."""
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        shape = (
            num_samples,
            self.config['model']['in_channels'],
            self.config['data']['image_size'],
            self.config['data']['image_size']
        )
        
        # Get conditioning
        cond = None
        if self.config['model'].get('use_conditioning', False) and self.label_embedder is not None:
            if digit is not None:
                # Generate specific digit
                class_labels = torch.tensor([digit] * num_samples, device=self.device)
            else:
                # Random digits
                class_labels = torch.randint(0, 10, (num_samples,), device=self.device)
            
            if self.config['model']['architecture'] == 'mmdit':
                label_embeds = self.label_embedder(class_labels)
                cond = label_embeds.unsqueeze(1)
            else:
                cond = self.label_embedder(class_labels)
        elif digit is not None:
            print("⚠️  This model is not conditional, ignoring digit specification")
        
        # Sample with CFG if configured
        cfg_scale = self.config.get('sampling', {}).get('cfg_scale', 1.0)
        if self.config['training']['target'] == TargetMode.FLOW:
            samples = rf_sampler(self.model, shape, steps=steps, cond=cond, 
                               device=self.device, cfg_scale=cfg_scale)
        else:
            samples = ddim_sampler(
                self.model, shape, steps=steps, cond=cond,
                target=self.config['training']['target'], device=self.device,
                cfg_scale=cfg_scale
            )
        
        return samples
    
    def interactive_session(self):
        """Run interactive generation session."""
        print("\n" + "="*60)
        print("🎨 MNIST Diffusion Model - Interactive Demo")
        print("="*60)
        
        if self.config['model'].get('use_conditioning', False):
            print("This model supports conditional generation!")
            print("You can generate specific digits (0-9)")
        else:
            print("This model generates random digits")
        
        print("\nCommands:")
        print("  [0-9]     - Generate specific digit (if conditional)")
        print("  random    - Generate random digits")
        print("  grid      - Generate grid of all digits")
        print("  compare   - Compare different sampling steps")
        print("  save      - Save last generated image")
        print("  quit      - Exit demo")
        print("-"*60)
        
        last_samples = None
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    print("Goodbye!")
                    break
                
                elif command == 'random':
                    num = int(input("How many samples? (1-16): "))
                    num = min(max(num, 1), 16)
                    print(f"Generating {num} random samples...")
                    samples = self.generate(digit=None, num_samples=num, steps=50)
                    self.display_samples(samples)
                    last_samples = samples
                
                elif command == 'grid':
                    if not self.config['model'].get('use_conditioning', False):
                        print("⚠️  Grid generation requires a conditional model")
                        continue
                    
                    print("Generating grid of all digits...")
                    all_samples = []
                    for digit in range(10):
                        samples = self.generate(digit=digit, num_samples=5, steps=50)
                        all_samples.append(samples)
                    
                    grid = torch.cat(all_samples, dim=0)
                    self.display_samples(grid, nrow=10)
                    last_samples = grid
                
                elif command == 'compare':
                    digit = None
                    if self.config['model'].get('use_conditioning', False):
                        digit = int(input("Which digit? (0-9, or -1 for random): "))
                        if digit == -1:
                            digit = None
                    
                    print("Comparing different sampling steps...")
                    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                    
                    for i, steps in enumerate([10, 25, 50, 100]):
                        samples = self.generate(digit=digit, num_samples=1, steps=steps, seed=42)
                        img = samples[0, 0].cpu().numpy()
                        img = (img + 1) / 2  # Denormalize
                        
                        axes[i].imshow(img, cmap='gray')
                        axes[i].set_title(f"{steps} steps")
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                
                elif command == 'save':
                    if last_samples is None:
                        print("No samples to save! Generate some first.")
                        continue
                    
                    filename = input("Filename (without extension): ") + ".png"
                    save_image(last_samples, filename, normalize=True, value_range=(-1, 1))
                    print(f"✅ Saved to {filename}")
                
                elif command.isdigit() and 0 <= int(command) <= 9:
                    if not self.config['model'].get('use_conditioning', False):
                        print("⚠️  This model doesn't support conditional generation")
                        continue
                    
                    digit = int(command)
                    num = int(input(f"How many {digit}'s? (1-16): "))
                    num = min(max(num, 1), 16)
                    
                    print(f"Generating {num} samples of digit {digit}...")
                    samples = self.generate(digit=digit, num_samples=num, steps=50)
                    self.display_samples(samples)
                    last_samples = samples
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
                
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")
    
    def display_samples(self, samples, nrow=None):
        """Display generated samples."""
        if nrow is None:
            nrow = int(np.sqrt(samples.size(0)))
        
        # Create grid
        from torchvision.utils import make_grid
        grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1))
        
        # Convert to numpy
        grid_np = grid[0].cpu().numpy()  # Get first channel for grayscale
        
        # Display
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_np, cmap='gray')
        plt.axis('off')
        plt.title(f"Generated Samples ({samples.size(0)} images)")
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Interactive MNIST generation demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--steps', type=int, default=50,
                       help='Default sampling steps')
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        
        exp_dir = Path("experiments_mnist")
        if exp_dir.exists():
            for exp in exp_dir.iterdir():
                ckpt_dir = exp / "checkpoints"
                if ckpt_dir.exists():
                    checkpoints = list(ckpt_dir.glob("*.pt"))
                    if checkpoints:
                        print(f"  {exp.name}:")
                        for ckpt in checkpoints:
                            print(f"    - {ckpt}")
        return
    
    # Create demo
    demo = InteractiveDemo(args.checkpoint)
    
    # Run interactive session
    demo.interactive_session()


if __name__ == "__main__":
    main()