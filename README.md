# Educational Diffusion Models

A clean, educational implementation of diffusion models showcasing the evolution from Stable Diffusion 1 to SD3/Flux architectures. This repository provides a straightforward learning path through the key innovations in diffusion model development.

## 📚 The Evolution Story

This repository demonstrates three key milestones in diffusion model development:

### 1. **SD1-style (UNet + Epsilon)** 
- Classic U-Net architecture with epsilon prediction
- Classifier-Free Guidance (CFG) for conditional generation
- Foundation of Stable Diffusion 1.x

### 2. **SD2-style (UNet + V-Prediction)**
- Improved v-parameterization for better signal-to-noise
- Same U-Net architecture with enhanced training dynamics
- Key innovation in Stable Diffusion 2.x

### 3. **SD3-style (MMDiT + Flow Matching)**
- Multi-Modal Diffusion Transformer (MMDiT) architecture
- Rectified Flow / Flow Matching objective
- Modern approach used in SD3 and Flux

## 📅 Diffusion Model Timeline

A brief history of key developments:

- **2020**: DDPM - Denoising Diffusion Probabilistic Models establish the foundation
- **2021**: 
  - **GLIDE** - Introduces Classifier-Free Guidance (CFG)
  - **LDM** - Latent Diffusion Models work in compressed VAE space
- **2022**: 
  - **DALL-E 2** - CLIP latents + diffusion prior
  - **Imagen** - T5 text encoder + super-resolution cascade
  - **Stable Diffusion 1** - Open-source LDM with CLIP text encoder
- **2023**: 
  - **DiT** - Diffusion Transformers replace U-Nets
  - **Consistency Models** - Single-step generation
  - **SDXL** - Larger U-Net, improved conditioning
- **2024**: 
  - **SD3** - MMDiT architecture with flow matching
  - **Flux** - Improved MMDiT variants
  - **Sora** - Video generation with DiT

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/diffusion
cd diffusion

# Install dependencies
pip install -r requirements.txt

# Set up MNIST dataset
python setup_mnist.py
```

### Training Your First Model

Train all three models to see the evolution:

```bash
# Train SD1-style model (UNet + Epsilon)
python train.py --config configs/unified_sd1_unet_eps.yaml

# Train SD2-style model (UNet + V-prediction)
python train.py --config configs/unified_sd2_unet_v.yaml

# Train SD3-style model (MMDiT + Flow)
python train.py --config configs/unified_sd3_mmdit_flow.yaml
```

### Interactive Demo

Try the interactive MNIST generator:

```bash
# Run with your trained model
python interactive_demo.py --checkpoint experiments/unified_sd3_mmdit_flow/checkpoints/model_epoch_100.pt
```

### Compare Models

Visualize the differences between approaches:

```bash
python compare_models.py
```

## 🏗️ Architecture

```
diffusion/
├── experiment.py          # Core diffusion mathematics and models
├── train.py              # Unified training script
├── interactive_demo.py    # Interactive MNIST generation
├── sample.py             # Batch sampling utilities
├── compare_models.py     # Model comparison visualization
├── configs/
│   ├── unified_sd1_unet_eps.yaml      # SD1-style configuration
│   ├── unified_sd2_unet_v.yaml        # SD2-style configuration  
│   ├── unified_sd3_mmdit_flow.yaml    # SD3-style configuration
│   └── reference/                     # Additional architectures
│       ├── ddpm_original.yaml         # Original DDPM (2020)
│       ├── dit_transformer.yaml       # Pure transformer DiT
│       ├── unet_x0_prediction.yaml    # X0 parameterization
│       └── ldm_latent.yaml            # Latent diffusion
└── experiments/          # Training outputs and checkpoints
```

## 🔬 Key Features

### Unified Training Pipeline
- Single training script for all architectures
- Automatic MNIST dataset handling
- Built-in performance optimizations (torch.compile, TF32)

### Classifier-Free Guidance (CFG)
- Proper CFG implementation with dropout during training
- Configurable guidance scale for inference
- Conditional MNIST digit generation

### Modern Best Practices
- Standard timestep embeddings (10000-based frequencies)
- Proper v-parameterization and flow matching
- Fair iso-parameter comparisons (~1.7M params)

## 📊 Model Specifications

All models are configured with approximately **1.7M parameters** for fair comparison:

| Model | Architecture | Objective | Parameters | CFG Scale |
|-------|-------------|-----------|------------|-----------|
| SD1-style | U-Net | Epsilon | ~1.7M | 7.5 |
| SD2-style | U-Net | V-prediction | ~1.7M | 5.0 |
| SD3-style | MMDiT | Flow | ~1.7M | 3.5 |

## 🎯 Training Details

### Configuration Structure
```yaml
model:
  architecture: "unet"/"mmdit"    # Model type
  use_conditioning: true          # Enable CFG
  
training:
  target: "eps"/"v"/"flow"        # Training objective
  epochs: 100                     # Training duration
  cfg_dropout_prob: 0.1          # CFG dropout rate
  
sampling:
  cfg_scale: 7.5                  # Guidance strength
```

### Performance Optimizations
- Automatic batch size optimization
- torch.compile support (2.6x speedup)
- TF32 enabled for Ampere GPUs
- Efficient MNIST data loading

## 🖼️ Sample Results

The repository trains on MNIST for educational clarity. After 100 epochs:

- **SD1-style**: Clear digits with classic diffusion characteristics
- **SD2-style**: Improved sample quality with v-prediction
- **SD3-style**: Best quality with flow matching

## 📖 Learning Path

1. **Start with SD1**: Understand epsilon prediction and CFG
2. **Move to SD2**: See how v-parameterization improves training
3. **Explore SD3**: Learn transformer architectures and flow matching
4. **Experiment**: Modify configs, try different settings

## 🔬 Reference Architectures

Beyond the main SD1→SD3 path, explore these influential architectures:

### Additional Configurations

```bash
# Original DDPM - Where it all began (2020)
python train.py --config configs/reference/ddpm_original.yaml

# DiT - Pure transformer, no U-Net (2023)
python train.py --config configs/reference/dit_transformer.yaml

# X0 Prediction - Direct denoising approach
python train.py --config configs/reference/unet_x0_prediction.yaml

# Latent Diffusion - VAE compressed space (LDM/SD foundation)
python train.py --config configs/reference/ldm_latent.yaml
```

### Why These Matter:

- **DDPM Original**: The baseline that started modern diffusion (1000 steps!)
- **DiT (Diffusion Transformer)**: Showed transformers could replace CNNs entirely, leading to Sora and MMDiT
- **X0 Prediction**: Alternative to noise prediction - more intuitive but less stable
- **Latent Diffusion**: 16x efficiency by working in VAE space - enabled consumer GPU usage

### Architecture Evolution:

```
CNN-based:         Transformer-based:
DDPM (2020)   →    DiT (2023)
  ↓                   ↓
U-Net + CFG   →    MMDiT (2024)
  ↓                   ↓
Stable Diffusion    SD3/Flux
```

## 🔧 Advanced Usage

### Custom Configuration
```python
# Create your own config mixing features
model:
  architecture: "dit"      # Try DiT architecture
  embed_dim: 256           # Adjust model size
  
training:
  target: "x0"             # Try x0 prediction
  lr: 0.0002              # Tune learning rate
```

### Extending the Code
- Add new architectures in `experiment.py`
- Implement new objectives in `DiffusionLearner`
- Create custom datasets by extending `train.py`

## 📚 References

- [DDPM](https://arxiv.org/abs/2006.11239) - Original diffusion paper
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) - CFG technique
- [v-parameterization](https://arxiv.org/abs/2202.00512) - Progressive Distillation
- [DiT](https://arxiv.org/abs/2212.09748) - Diffusion Transformers
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Rectified flows
- [MMDiT](https://arxiv.org/abs/2403.03206) - SD3 architecture

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is an educational repository. Contributions that improve clarity and understanding are welcome!

---

*Built for learning. Start simple, understand deeply.*