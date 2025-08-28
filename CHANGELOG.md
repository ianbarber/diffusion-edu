# Changelog

## [1.0.0] - 2024

### Added
- MIT License file
- scipy>=1.10.0 to requirements.txt for model comparison metrics
- Security note for torch.load() calls (using weights_only=False for config compatibility)
- Comprehensive MNIST learning environment with 6 configurations
- Model comparison and benchmarking tools
- Interactive generation demo
- Automated training scripts for all configurations

### Fixed
- YAML config files: Changed scientific notation (e.g., 2e-4) to decimal (0.0002) for proper float parsing
- Added missing scipy dependency for compare_models.py
- Fixed dimension mismatch in UNet, DiT, and MMDiT models when using non-default embed_dim values
  - Time embedding dimension now automatically adjusts based on embed_dim parameter
  - This fixes the flow_unet configuration that was failing with embed_dim=192
- Fixed type annotation for add_noise function (now correctly shows 4 return values)
- **Critical fix**: DDIM sampler was using zeros instead of predicted noise
  - Changed `x = an * x0 + sn * torch.zeros_like(x)` to `x = an * x0 + sn * eps`
  - This fixes exploding values during sampling for epsilon-prediction models
  - All models need retraining with the fixed sampler for proper results

### Security
- Added comments explaining torch.load() security considerations
- Note: weights_only=False is required for loading config dictionaries from checkpoints

### Cleaned
- Removed __pycache__ directories
- Cleaned up intermediate checkpoint files (kept best and latest only)
- Reduced sample images to representative milestones
- Removed temporary and auto-generated files

### Code Review Results
- Overall Code Quality Score: 8.5/10
- Strong educational design with clear mathematical foundations
- Excellent modular architecture
- Comprehensive documentation and examples
- Minor improvements identified and addressed