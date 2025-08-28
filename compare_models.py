#!/usr/bin/env python3
"""
Compare trained diffusion models on MNIST.
Generates samples, plots training curves, and computes metrics.
"""

import os
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import json

from experiment import (
    UNetTiny, DiTTiny, MMDiTTiny,
    ddim_sampler, rf_sampler, TargetMode
)
from train import MNISTLabelEmbedder


class ModelComparator:
    def __init__(self, experiments_dir="./experiments_mnist"):
        self.experiments_dir = Path(experiments_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.configs = {}
        
    def load_model(self, exp_name, checkpoint_name="best_model.pt"):
        """Load a trained model."""
        exp_path = self.experiments_dir / exp_name
        checkpoint_path = exp_path / "checkpoints" / checkpoint_name
        
        if not checkpoint_path.exists():
            checkpoint_path = exp_path / "checkpoints" / "latest_model.pt"
            if not checkpoint_path.exists():
                print(f"⚠️  No checkpoint found for {exp_name}")
                return None
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)  # Set to False for config dict compatibility
        config = checkpoint['config']
        
        # Build model
        arch = config['model']['architecture']
        in_ch = config['model']['in_channels']
        out_ch = config['model']['out_channels']
        
        if arch == 'unet':
            model = UNetTiny(
                in_ch=in_ch,
                base=config['model'].get('base_channels', 64),
                emb_dim=config['model'].get('embed_dim', 256),
                out_ch=out_ch
            )
        elif arch == 'dit':
            model = DiTTiny(
                in_ch=in_ch,
                embed_dim=config['model'].get('embed_dim', 256),
                depth=config['model'].get('depth', 6),
                patch=config['model'].get('patch_size', 2),
                out_ch=out_ch
            )
        elif arch == 'mmdit':
            model = MMDiTTiny(
                in_ch=in_ch,
                img_dim=config['model'].get('embed_dim', 256),
                txt_dim=config['model'].get('text_dim', 256),
                depth=config['model'].get('depth', 6),
                patch=config['model'].get('patch_size', 2),
                out_ch=out_ch
            )
        else:
            print(f"Unknown architecture: {arch}")
            return None
        
        # Load state dict (handle torch.compile prefix)
        state_dict = checkpoint['model_state_dict']
        # Remove '_orig_mod.' prefix if present (from torch.compile)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        # Load label embedder if needed
        label_embedder = None
        if config['model'].get('use_conditioning', False):
            embed_dim = config['model'].get('text_dim', config['model']['embed_dim'])
            label_embedder = MNISTLabelEmbedder(num_classes=10, embed_dim=embed_dim).to(self.device)
            if 'label_embedder_state_dict' in checkpoint:
                label_embedder.load_state_dict(checkpoint['label_embedder_state_dict'])
        
        self.models[exp_name] = {
            'model': model,
            'config': config,
            'label_embedder': label_embedder,
            'checkpoint': checkpoint
        }
        
        print(f"✅ Loaded {exp_name}: {arch} with {config['training']['target']} objective")
        return model
    
    def load_all_models(self):
        """Load all available models."""
        experiments = [
            "ddpm_unet",
            "v_unet",
            "x0_unet",
            "dit",
            "flow_unet",
            "mmdit_flow"
        ]
        
        for exp_name in experiments:
            self.load_model(exp_name)
    
    @torch.no_grad()
    def generate_samples(self, exp_name, num_samples=64, steps=50):
        """Generate samples from a model."""
        if exp_name not in self.models:
            return None
        
        model_info = self.models[exp_name]
        model = model_info['model']
        config = model_info['config']
        label_embedder = model_info['label_embedder']
        
        shape = (
            num_samples,
            config['model']['in_channels'],
            config['data']['image_size'],
            config['data']['image_size']
        )
        
        # Get conditioning
        cond = None
        if config['model'].get('use_conditioning', False) and label_embedder is not None:
            # Generate samples for each class
            class_labels = torch.arange(10).repeat(num_samples // 10 + 1)[:num_samples]
            class_labels = class_labels.to(self.device)
            
            if config['model']['architecture'] == 'mmdit':
                label_embeds = label_embedder(class_labels)
                cond = label_embeds.unsqueeze(1)
            else:
                cond = label_embedder(class_labels)
        
        # Sample based on target
        if config['training']['target'] == TargetMode.FLOW:
            samples = rf_sampler(model, shape, steps=steps, cond=cond, device=self.device)
        else:
            samples = ddim_sampler(
                model, shape, steps=steps, cond=cond,
                target=config['training']['target'], device=self.device
            )
        
        return samples
    
    def compare_samples(self, num_samples=64, steps=50, save_path="comparison.png"):
        """Generate and compare samples from all models."""
        print(f"\n📊 Generating samples for comparison...")
        
        all_samples = {}
        for exp_name in self.models.keys():
            print(f"  Generating from {exp_name}...")
            samples = self.generate_samples(exp_name, num_samples, steps)
            if samples is not None:
                all_samples[exp_name] = samples
        
        if not all_samples:
            print("No samples generated!")
            return
        
        # Create comparison grid
        fig, axes = plt.subplots(len(all_samples), 8, figsize=(16, 2*len(all_samples)))
        if len(all_samples) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (exp_name, samples) in enumerate(all_samples.items()):
            # Show first 8 samples
            for j in range(min(8, samples.size(0))):
                img = samples[j, 0].cpu().numpy()  # Get first channel
                img = (img + 1) / 2  # Denormalize to [0, 1]
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(exp_name.replace('_', ' ').title(), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison saved to {save_path}")
        
        # Also save individual grids
        output_dir = Path("comparison_samples")
        output_dir.mkdir(exist_ok=True)
        
        for exp_name, samples in all_samples.items():
            save_image(
                samples,
                output_dir / f"{exp_name}_samples.png",
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )
        
        return all_samples
    
    def measure_generation_speed(self, steps_list=[10, 30, 50, 100]):
        """Measure generation speed for different models."""
        print(f"\n⏱️  Measuring generation speed...")
        
        results = {}
        batch_size = 16
        
        for exp_name in self.models.keys():
            model_info = self.models[exp_name]
            model = model_info['model']
            config = model_info['config']
            
            results[exp_name] = {}
            
            for steps in steps_list:
                # Warm up
                _ = self.generate_samples(exp_name, num_samples=4, steps=steps)
                
                # Measure
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start.record()
                
                import time
                cpu_start = time.time()
                
                _ = self.generate_samples(exp_name, num_samples=batch_size, steps=steps)
                
                if torch.cuda.is_available():
                    end.record()
                    torch.cuda.synchronize()
                    gpu_time = start.elapsed_time(end) / 1000  # Convert to seconds
                    results[exp_name][steps] = gpu_time
                else:
                    cpu_time = time.time() - cpu_start
                    results[exp_name][steps] = cpu_time
            
            print(f"  {exp_name}: {results[exp_name]}")
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for exp_name, timings in results.items():
            steps = list(timings.keys())
            times = list(timings.values())
            ax.plot(steps, times, marker='o', label=exp_name)
        
        ax.set_xlabel('Sampling Steps')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Generation Speed Comparison (batch_size={batch_size})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig("generation_speed.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Speed comparison saved to generation_speed.png")
        
        return results
    
    def compute_diversity_metric(self, num_samples=100):
        """Compute diversity metrics for generated samples."""
        print(f"\n📊 Computing diversity metrics...")
        
        results = {}
        
        for exp_name in self.models.keys():
            samples = self.generate_samples(exp_name, num_samples=num_samples, steps=50)
            
            if samples is None:
                continue
            
            # Flatten samples
            samples_flat = samples.view(num_samples, -1).cpu().numpy()
            
            # Compute pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(samples_flat, metric='euclidean')
            
            # Compute statistics
            results[exp_name] = {
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances))
            }
            
            print(f"  {exp_name}: mean_dist={results[exp_name]['mean_distance']:.2f}, "
                  f"std={results[exp_name]['std_distance']:.2f}")
        
        return results
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        print(f"\n📝 Creating summary report...")
        
        report = []
        report.append("# MNIST Diffusion Models Comparison Report\n")
        report.append("## Models Evaluated\n")
        
        # Model details
        for exp_name, model_info in self.models.items():
            config = model_info['config']
            checkpoint = model_info['checkpoint']
            
            report.append(f"### {exp_name}\n")
            report.append(f"- Architecture: {config['model']['architecture']}\n")
            report.append(f"- Training objective: {config['training']['target']}\n")
            report.append(f"- Epochs trained: {checkpoint['epoch']}\n")
            report.append(f"- Parameters: {sum(p.numel() for p in model_info['model'].parameters()):,}\n")
            report.append(f"- Conditional: {config['model'].get('use_conditioning', False)}\n\n")
        
        # Generation speed
        speed_results = self.measure_generation_speed()
        report.append("## Generation Speed (seconds for batch_size=16)\n")
        report.append("| Model | 10 steps | 30 steps | 50 steps | 100 steps |\n")
        report.append("|-------|----------|----------|----------|----------|\n")
        
        for exp_name, timings in speed_results.items():
            row = f"| {exp_name} "
            for steps in [10, 30, 50, 100]:
                if steps in timings:
                    row += f"| {timings[steps]:.3f} "
                else:
                    row += "| N/A "
            row += "|\n"
            report.append(row)
        
        # Diversity metrics
        diversity_results = self.compute_diversity_metric()
        report.append("\n## Sample Diversity Metrics\n")
        report.append("| Model | Mean Distance | Std Distance |\n")
        report.append("|-------|--------------|-------------|\n")
        
        for exp_name, metrics in diversity_results.items():
            report.append(f"| {exp_name} | {metrics['mean_distance']:.2f} | {metrics['std_distance']:.2f} |\n")
        
        # Save report
        report_path = "comparison_report.md"
        with open(report_path, 'w') as f:
            f.writelines(report)
        
        print(f"✅ Report saved to {report_path}")
        
        # Also save as JSON for programmatic access
        json_data = {
            'models': {name: {
                'architecture': info['config']['model']['architecture'],
                'target': info['config']['training']['target'],
                'epochs': info['checkpoint']['epoch'],
                'parameters': sum(p.numel() for p in info['model'].parameters())
            } for name, info in self.models.items()},
            'speed': speed_results,
            'diversity': diversity_results
        }
        
        with open('comparison_data.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✅ Data saved to comparison_data.json")


def main():
    parser = argparse.ArgumentParser(description='Compare trained diffusion models')
    parser.add_argument('--experiments_dir', type=str, default='./experiments_mnist',
                       help='Directory containing experiments')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of sampling steps')
    args = parser.parse_args()
    
    comparator = ModelComparator(args.experiments_dir)
    
    print("🔍 Loading trained models...")
    comparator.load_all_models()
    
    if not comparator.models:
        print("❌ No models found! Train some models first.")
        return
    
    print(f"\n✅ Loaded {len(comparator.models)} models")
    
    # Generate comparison
    comparator.compare_samples(num_samples=args.num_samples, steps=args.steps)
    
    # Create full report
    comparator.create_summary_report()
    
    print("\n🎉 Comparison complete! Check the generated files:")
    print("  - comparison.png: Visual comparison of samples")
    print("  - comparison_samples/: Individual sample grids")
    print("  - generation_speed.png: Speed comparison chart")
    print("  - comparison_report.md: Detailed report")
    print("  - comparison_data.json: Raw data")


if __name__ == "__main__":
    main()