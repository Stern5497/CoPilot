# CoPilot - Conditional DDPM Implementation

This repository contains an implementation of a Conditional Denoising Diffusion Probabilistic Model (DDPM) for image-to-image translation tasks, based on [junbopeng/conditional_DDPM](https://github.com/junbopeng/conditional_DDPM).

## Overview

A general framework for image-to-image translation using conditional denoising diffusion probabilistic models. Originally designed for CBCT-to-CT image translation, but can be adapted to various image translation tasks.

## Reference Paper

If you use this code, please cite:
> Peng J, Qiu RLJ, Wynne JF, et al. CBCT-Based synthetic CT image generation using conditional denoising diffusion probabilistic model. Med Phys. 2023; 1-13. https://doi.org/10.1002/mp.16704

## Files

- `Diffusion_condition.py` - Core diffusion process implementation (training and sampling)
- `Model_condition.py` - U-Net architecture with time embeddings
- `datasets.py` - Dataset loader for paired image data
- `Train_condition.py` - Training script
- `Test_condition.py` - Inference/testing script
- `demo.py` - Demo script with dummy data generation

## Quick Start with Dummy Data

Run the demo script to see the model in action with randomly generated dummy data:

```bash
pip install -r requirements.txt
python demo.py
```

This will:
1. Generate dummy training and test data
2. Train a small model for 3 epochs
3. Run inference on test samples
4. Save model checkpoints to `./Checkpoints/demo/`

## Running Tests

To verify the implementation works correctly:

```bash
python test_components.py
```

This runs unit tests for all core components (UNet, diffusion trainer/sampler, and data generation).

## Training with Your Own Data

1. Organize your data in the following structure:
```
data/
├── train/
│   ├── a/  # Target images (e.g., CT scans)
│   └── b/  # Condition images (e.g., CBCT scans)
└── test/
    ├── a/  # Target test images
    └── b/  # Condition test images
```

2. Images should be saved as `.npy` files (numpy arrays)

3. Run training:
```bash
python Train_condition.py
```

4. Run inference:
```bash
python Test_condition.py
```

## Implementation Details

This implementation includes:

- **Conditional DDPM**: The model uses a conditional denoising diffusion process where the generation is guided by a condition image (e.g., CBCT for CT generation)
- **U-Net Architecture**: A time-conditioned U-Net with attention mechanisms, residual blocks, and skip connections
- **Flexible Configuration**: Easy to modify hyperparameters like timesteps, channel counts, and attention layers
- **GPU/CPU Support**: Automatically detects and uses GPU if available, falls back to CPU otherwise

The model works by:
1. Adding noise to target images over T timesteps (forward process)
2. Learning to denoise conditioned on the input image (training)
3. Generating new images from random noise by iterative denoising (inference)

## Model Configuration

Key hyperparameters in the scripts:
- `T`: Number of diffusion timesteps (default: 1000)
- `ch`: Base channel count (default: 128)
- `ch_mult`: Channel multipliers for each resolution level
- `num_res_blocks`: Number of residual blocks per level
- `dropout`: Dropout rate
- `beta_1`, `beta_T`: Diffusion schedule parameters

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Pillow

## Notes

- The model supports both CPU and GPU training
- Checkpoints are saved periodically during training
- Test outputs are saved as raw binary files