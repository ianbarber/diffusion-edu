# Claude Code Guidelines

## Project Overview
Educational diffusion model implementation showing evolution from Stable Diffusion 1 to SD3/MMDiT.

## Key Principles
- **Educational clarity**: Code should be readable and demonstrate concepts clearly
- **Unified approach**: Single training script, consistent patterns across models
- **Performance matters**: Use torch.compile, optimize batch sizes for GPU efficiency
- **Clean codebase**: No duplicate code, reuse components where possible

## Code Conventions
- Use type hints for function signatures
- Keep line length under 100 characters
- Scientific notation in configs: use decimal form (0.0002, not 2e-4)
- Progress bars: use tqdm with clear epoch/loss/lr display

## Testing Guidelines
Before committing changes:
1. Verify training runs: `python train.py configs/unified_sd1_unet_eps.yaml --epochs 1`
2. Check sampling works: `python sample.py outputs/unified_sd1_unet_eps/checkpoint_final.pt`
3. Ensure no import errors in core modules

## Model Configuration
- Target ~1.7M parameters for fair comparisons
- MNIST: batch_size=576, cfg_scale=5.0, steps=100
- Use CFG dropout of 0.1 during training
- Standard timestep embedding (10000-based frequencies)

## Directory Structure
- `configs/`: Model configuration YAML files
- `outputs/`: Training checkpoints and samples (git-ignored)
- `experiment.py`: Core diffusion implementations
- `train.py`: Unified training script
- `sample.py`: Generation script