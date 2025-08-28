#!/usr/bin/env python3
"""
Training script for diffusion models with multiple architectures and targets.
Supports UNet, DiT, and MMDiT with eps/v/x0/flow objectives.
"""

import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter

# Enable TF32 for better performance on Ampere/Ada GPUs (RTX 30xx, 40xx, A100)
# This provides ~10-20% speedup with minimal accuracy impact
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from experiment import (
    UNetTiny, DiTTiny, MMDiTTiny, TinyVAE, TinyTextToker,
    DiffusionLearner, TrainConfig, TargetMode,
    ddim_sampler, rf_sampler
)


class MNISTLabelEmbedder(nn.Module):
    """Simple embedder for MNIST digit labels (0-9)."""
    def __init__(self, num_classes=10, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        
    def forward(self, labels):
        # labels: (B,) tensor of class indices
        return self.embedding(labels)


class ImageDataset(Dataset):
    """Generic dataset wrapper for image folders."""
    def __init__(self, root_dir, transform=None, use_latents=False, vae=None):
        self.root_dir = Path(root_dir)
        self.use_latents = use_latents
        self.vae = vae
        
        # Use torchvision ImageFolder for convenience
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        self.dataset = datasets.ImageFolder(root=root_dir, transform=self.transform)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if self.use_latents and self.vae is not None:
            with torch.no_grad():
                img = self.vae.encode(img.unsqueeze(0)).squeeze(0)
        
        return {'image': img, 'label': label}


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.build_model()
        
        # Keep reference to uncompiled model for clean checkpoints
        self.model_uncompiled = self.model
        
        # Apply torch.compile for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile') and config.get('use_torch_compile', True):
            print("🔥 Compiling model with torch.compile for better performance...")
            self.model = torch.compile(self.model)
            self.is_compiled = True
        else:
            self.is_compiled = False
        
        # Initialize learner
        train_cfg = TrainConfig(
            target=config['training']['target'],
            lr=config['training']['lr']
        )
        self.learner = DiffusionLearner(self.model, train_cfg).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.learner.parameters(),
            lr=config['training']['lr'],
            betas=(config['training']['beta1'], config['training']['beta2']),
            weight_decay=config['training']['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = None
        if config['training'].get('use_scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs'],
                eta_min=config['training']['lr'] * 0.01
            )
        
        # Initialize VAE if using latent diffusion
        self.vae = None
        if config['model'].get('use_latents', False):
            self.vae = TinyVAE(
                in_ch=3,
                latent_ch=config['model']['in_channels']
            ).to(self.device)
            self.vae.eval()
        
        # Initialize text encoder or label embedder for conditional models
        self.text_encoder = None
        self.label_embedder = None
        
        # Check if this is MNIST
        is_mnist = self._is_mnist_config()
        
        if config['model'].get('use_conditioning', False):
            if is_mnist:
                # Use label embedder for MNIST
                embed_dim = config['model'].get('text_dim', config['model']['embed_dim'])
                self.label_embedder = MNISTLabelEmbedder(
                    num_classes=10,
                    embed_dim=embed_dim
                ).to(self.device)
            elif config['model']['architecture'] in ['dit', 'mmdit']:
                # Use text encoder for general datasets
                self.text_encoder = TinyTextToker(
                    vocab=config['model'].get('vocab_size', 1000),
                    dim=config['model']['embed_dim']
                ).to(self.device)
        
        # Setup data
        self.train_loader = self.setup_dataloader()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        
    def setup_directories(self):
        """Create necessary directories."""
        self.exp_dir = Path(self.config['exp_dir'])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.sample_dir = self.exp_dir / 'samples'
        self.sample_dir.mkdir(exist_ok=True)
        
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
    
    def _is_mnist_config(self):
        """Check if this is an MNIST configuration."""
        return 'mnist' in self.config.get('exp_name', '').lower() or \
               'mnist' in self.config['data'].get('train_dir', '').lower() or \
               self.config['data'].get('mnist_root') is not None
    
    def setup_dataloader(self):
        """Setup data loader - automatically detects MNIST configs."""
        # Check if this is MNIST dataset
        is_mnist = self._is_mnist_config()
        
        if is_mnist:
            # Use MNIST dataset
            from mnist_dataset import get_mnist_dataloader, get_class_conditional_dataloader
            
            if self.config['model']['architecture'] == 'mmdit':
                # MMDiT always needs labels
                return get_class_conditional_dataloader(self.config, train=True)
            else:
                return get_mnist_dataloader(self.config, train=True, vae=self.vae)
        else:
            # Use general image dataset
            transform = transforms.Compose([
                transforms.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
                transforms.CenterCrop(self.config['data']['image_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            
            dataset = ImageDataset(
                root_dir=self.config['data']['train_dir'],
                transform=transform,
                use_latents=self.config['model'].get('use_latents', False),
                vae=self.vae
            )
            
            return DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['data'].get('num_workers', 4),
                pin_memory=True,
                drop_last=True
            )
    
    def setup_logging(self):
        """Setup logging with tensorboard and optionally wandb."""
        # TensorBoard
        self.writer = SummaryWriter(self.exp_dir / 'tensorboard')
        
        # Weights & Biases
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'diffusion'),
                name=self.config.get('exp_name', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=self.config
            )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.learner.train()
        epoch_loss = 0
        num_batches = len(self.train_loader)
        
        # Create progress bar with better formatting
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch + 1}/{self.config['training']['epochs']}",
            ncols=100,  # Fixed width for cleaner display
            leave=True  # Keep the progress bar after completion
        )
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch.get('label')
            if labels is not None:
                labels = labels.to(self.device)
            
            # Get conditioning if needed
            cond = None
            if self.config['model'].get('use_conditioning', False):
                if self.label_embedder is not None and labels is not None:
                    # MNIST with labels
                    if self.config['model']['architecture'] == 'mmdit':
                        label_embeds = self.label_embedder(labels)
                        cond = label_embeds.unsqueeze(1)  # (B, 1, D) for MMDiT
                    else:
                        cond = self.label_embedder(labels)
                elif self.text_encoder is not None:
                    # General dataset with text encoder
                    if self.config['model']['architecture'] == 'mmdit':
                        batch_size = images.size(0)
                        txt_ids = torch.randint(0, 1000, (batch_size, 32), device=self.device)
                        cond = self.text_encoder(txt_ids)
                    else:
                        batch_size = images.size(0)
                        txt_ids = torch.randint(0, 1000, (batch_size, 32), device=self.device)
                        txt_tokens = self.text_encoder(txt_ids)
                        cond = txt_tokens.mean(dim=1)
            
            # Forward pass
            loss = self.learner.loss(images, cond)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.learner.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar with rich information
            current_loss = loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update postfix with formatted values
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.1e}'
            })
            
            # Log to tensorboard
            if self.global_step % self.config['logging']['log_freq'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': self.epoch
                    }, step=self.global_step)
        
        return epoch_loss / len(self.train_loader)
    
    @torch.no_grad()
    def sample(self, num_samples=4, class_labels=None):
        """Generate samples."""
        self.learner.eval()
        
        # Determine image size based on whether using MNIST or general dataset
        is_mnist = self._is_mnist_config()
        if is_mnist:
            img_size = self.config['data']['image_size']
        else:
            img_size = self.config['data']['latent_size'] if self.config['model'].get('use_latents', False) else self.config['data']['image_size'] // 8
        
        shape = (
            num_samples,
            self.config['model']['in_channels'],
            img_size,
            img_size
        )
        
        # Get conditioning if needed
        cond = None
        if self.config['model'].get('use_conditioning', False):
            if self.label_embedder is not None:
                # MNIST with label conditioning
                if class_labels is None:
                    # Generate samples for each class
                    class_labels = torch.arange(10).repeat(num_samples // 10 + 1)[:num_samples]
                class_labels = class_labels.to(self.device)
                
                if self.config['model']['architecture'] == 'mmdit':
                    label_embeds = self.label_embedder(class_labels)
                    cond = label_embeds.unsqueeze(1)
                else:
                    cond = self.label_embedder(class_labels)
            elif self.text_encoder is not None:
                # General dataset with text conditioning
                if self.config['model']['architecture'] == 'mmdit':
                    txt_ids = torch.randint(0, 1000, (num_samples, 32), device=self.device)
                    cond = self.text_encoder(txt_ids)
                else:
                    txt_ids = torch.randint(0, 1000, (num_samples, 32), device=self.device)
                    txt_tokens = self.text_encoder(txt_ids)
                    cond = txt_tokens.mean(dim=1)
        
        # Sample based on target
        if self.config['training']['target'] == TargetMode.FLOW:
            samples = rf_sampler(
                self.learner.model,
                shape,
                steps=self.config['sampling']['steps'],
                cond=cond,
                device=self.device
            )
        else:
            samples = ddim_sampler(
                self.learner.model,
                shape,
                steps=self.config['sampling']['steps'],
                cond=cond,
                target=self.config['training']['target'],
                device=self.device
            )
        
        # Decode if using VAE
        if self.vae is not None:
            samples = self.vae.decode(samples)
        
        return samples
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        # Use uncompiled model for clean state dict (avoids _orig_mod prefix)
        model_to_save = self.model_uncompiled if self.is_compiled else self.learner.model
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.label_embedder is not None:
            checkpoint['label_embedder_state_dict'] = self.label_embedder.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)  # Set to False for config dict compatibility
        
        self.learner.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.label_embedder is not None and 'label_embedder_state_dict' in checkpoint:
            self.label_embedder.load_state_dict(checkpoint['label_embedder_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        best_loss = float('inf')
        
        # Print training configuration
        print("\n" + "="*70)
        print(f"🚀 Starting Training: {self.config.get('exp_name', 'experiment')}")
        print("="*70)
        print(f"   Architecture: {self.config['model']['architecture']}")
        print(f"   Target: {self.config['training']['target']}")
        print(f"   Epochs: {self.config['training']['epochs']}")
        print(f"   Batch size: {self.config['training']['batch_size']}")
        print(f"   Learning rate: {self.config['training']['lr']:.1e}")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Dataset batches: {len(self.train_loader)}")
        print("="*70 + "\n")
        
        for epoch in range(self.epoch, self.config['training']['epochs']):
            self.epoch = epoch
            
            # Train epoch
            avg_loss = self.train_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Sample and save
            if (epoch + 1) % self.config['sampling']['sample_freq'] == 0:
                samples = self.sample(num_samples=self.config['sampling']['num_samples'])
                
                # Save samples as grid
                from torchvision.utils import save_image
                
                # For conditional MNIST models, organize by class
                if self.config['model'].get('use_conditioning', False) and self.label_embedder is not None:
                    save_image(
                        samples,
                        self.sample_dir / f"samples_epoch_{epoch:04d}_conditional.png",
                        nrow=10,  # 10 columns for 10 digits
                        normalize=True,
                        value_range=(-1, 1)
                    )
                else:
                    save_image(
                        samples,
                        self.sample_dir / f"samples_epoch_{epoch:04d}.png",
                        nrow=int(samples.size(0) ** 0.5),
                        normalize=True,
                        value_range=(-1, 1)
                    )
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['checkpoint_freq'] == 0:
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary with more details
            elapsed = time.time() if hasattr(self, '_epoch_start_time') else 0
            if not hasattr(self, '_epoch_start_time'):
                self._epoch_start_time = time.time()
            epoch_time = time.time() - self._epoch_start_time
            self._epoch_start_time = time.time()
            
            print(f"\n📊 Epoch {epoch + 1}/{self.config['training']['epochs']} Summary:")
            print(f"   Average Loss: {avg_loss:.4f} | Best Loss: {best_loss:.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.1e}")
            print(f"   Time: {epoch_time:.1f}s | ETA: {epoch_time * (self.config['training']['epochs'] - epoch - 1):.0f}s")
            print("-" * 60)
        
        # Final checkpoint
        self.save_checkpoint()
        self.writer.close()
        
        if self.config.get('use_wandb', False):
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train diffusion models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()