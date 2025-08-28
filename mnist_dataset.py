#!/usr/bin/env python3
"""
MNIST dataset handler for diffusion model training.
Handles grayscale to multi-channel conversion and proper normalization.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path


class MNISTDiffusionDataset(Dataset):
    """MNIST dataset wrapper for diffusion training."""
    
    def __init__(self, 
                 root='./data/mnist',
                 train=True,
                 image_size=32,
                 channels=1,
                 use_labels=False,
                 download=True):
        """
        Args:
            root: Root directory for MNIST data
            train: Whether to use training or test set
            image_size: Size to resize images to (default 32 for better architecture compatibility)
            channels: Number of output channels (1 for grayscale, 3 for RGB-compatible)
            use_labels: Whether to return labels for conditional training
            download: Whether to download MNIST if not present
        """
        self.channels = channels
        self.use_labels = use_labels
        self.image_size = image_size
        
        # Define transforms
        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ]
        
        # Add channel padding if needed
        if channels == 3:
            # Repeat grayscale channel to create RGB
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        elif channels == 4:
            # For latent-space models, pad with zeros
            transform_list.append(transforms.Lambda(lambda x: torch.cat([
                x, torch.zeros(3, image_size, image_size)
            ], dim=0)))
        
        self.transform = transforms.Compose(transform_list)
        
        # Load MNIST dataset
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=self.transform,
            download=download
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if self.use_labels:
            return {'image': img, 'label': label}
        else:
            # For unconditional training, still return label but it won't be used
            return {'image': img, 'label': 0}


class MNISTLatentDataset(Dataset):
    """MNIST dataset for latent diffusion (with VAE encoding)."""
    
    def __init__(self,
                 vae_model,
                 root='./data/mnist',
                 train=True,
                 image_size=32,
                 use_labels=False,
                 download=True,
                 device='cuda'):
        """
        Args:
            vae_model: Pre-trained VAE model for encoding
            root: Root directory for MNIST data
            train: Whether to use training or test set
            image_size: Size to resize images to
            use_labels: Whether to return labels
            download: Whether to download MNIST
            device: Device for VAE encoding
        """
        self.vae = vae_model
        self.vae.eval()
        self.device = device
        self.use_labels = use_labels
        
        # Standard transforms for VAE input
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # VAE expects 3 channels
        ])
        
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download
        )
        
        # Pre-compute latents for faster training (optional)
        self.precomputed = False
        self.latents = []
        self.labels = []
        
    def precompute_latents(self):
        """Pre-encode all images to latents for faster training."""
        print("Pre-computing latents...")
        self.latents = []
        self.labels = []
        
        with torch.no_grad():
            for img, label in self.dataset:
                img = img.unsqueeze(0).to(self.device)
                latent = self.vae.encode(img).cpu().squeeze(0)
                self.latents.append(latent)
                self.labels.append(label)
        
        self.precomputed = True
        print(f"Pre-computed {len(self.latents)} latents")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.precomputed:
            latent = self.latents[idx]
            label = self.labels[idx]
        else:
            img, label = self.dataset[idx]
            with torch.no_grad():
                img = img.unsqueeze(0).to(self.device)
                latent = self.vae.encode(img).cpu().squeeze(0)
        
        if self.use_labels:
            return {'image': latent, 'label': label}
        else:
            return {'image': latent, 'label': 0}


def get_mnist_dataloader(config, train=True, vae=None):
    """
    Get MNIST dataloader based on config settings.
    
    Args:
        config: Configuration dictionary
        train: Whether to get training or test dataloader
        vae: Optional VAE model for latent diffusion
    
    Returns:
        DataLoader for MNIST
    """
    # Determine number of channels based on model config
    if config['model'].get('use_latents', False):
        # Latent diffusion
        if vae is None:
            raise ValueError("VAE model required for latent diffusion")
        
        dataset = MNISTLatentDataset(
            vae_model=vae,
            root=config['data'].get('mnist_root', './data/mnist'),
            train=train,
            image_size=config['data']['image_size'],
            use_labels=config['model'].get('use_conditioning', False),
            device=config['device']
        )
        
        # Optionally precompute latents
        if config['data'].get('precompute_latents', False):
            dataset.precompute_latents()
    else:
        # Pixel-space diffusion
        in_channels = config['model']['in_channels']
        
        dataset = MNISTDiffusionDataset(
            root=config['data'].get('mnist_root', './data/mnist'),
            train=train,
            image_size=config['data']['image_size'],
            channels=in_channels,
            use_labels=config['model'].get('use_conditioning', False)
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=train,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=train
    )
    
    return dataloader


def get_class_conditional_dataloader(config, train=True):
    """
    Get MNIST dataloader with class labels for conditional training.
    Always returns labels regardless of config settings.
    """
    in_channels = config['model']['in_channels']
    
    dataset = MNISTDiffusionDataset(
        root=config['data'].get('mnist_root', './data/mnist'),
        train=train,
        image_size=config['data']['image_size'],
        channels=in_channels,
        use_labels=True  # Always return labels
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=train,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=train
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    print("Testing MNIST dataset...")
    
    # Test grayscale
    dataset_1ch = MNISTDiffusionDataset(channels=1)
    print(f"1-channel dataset: {len(dataset_1ch)} samples")
    sample = dataset_1ch[0]
    print(f"Sample shape: {sample['image'].shape}")
    
    # Test RGB
    dataset_3ch = MNISTDiffusionDataset(channels=3)
    sample = dataset_3ch[0]
    print(f"3-channel sample shape: {sample['image'].shape}")
    
    # Test with labels
    dataset_labeled = MNISTDiffusionDataset(channels=3, use_labels=True)
    sample = dataset_labeled[0]
    print(f"Labeled sample: image shape={sample['image'].shape}, label={sample['label']}")
    
    print("Dataset tests passed!")