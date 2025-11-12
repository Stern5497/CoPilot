# CoPilot - Conditional DDPM Implementation (3D Support)

This repository contains an implementation of a Conditional Denoising Diffusion Probabilistic Model (DDPM) for image-to-image translation tasks, based on [junbopeng/conditional_DDPM](https://github.com/junbopeng/conditional_DDPM).

**Now supports 3D volumetric data!** The implementation works with various 3D image sizes such as (4, 128, 128), (8, 64, 64), or any other depth/height/width combinations.

## Overview

A general framework for image-to-image translation using conditional denoising diffusion probabilistic models. Originally designed for CBCT-to-CT image translation, but can be adapted to various image translation tasks including 3D medical imaging.

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

Run the demo script to see the model in action with randomly generated dummy 3D data:

```bash
pip install -r requirements.txt
python demo.py
```

This will:
1. Generate dummy 3D training and test data (shape: 4x64x64)
2. Train a small 3D model for 3 epochs
3. Run inference on test samples
4. Save model checkpoints to `./Checkpoints/demo/`

### Using Different 3D Image Sizes

You can easily test with different 3D shapes like (4, 128, 128) or (8, 64, 64):

```python
from demo import generate_dummy_data, run_demo

# Example 1: 4 depth slices, 128x128 spatial
generate_dummy_data(image_shape=(4, 128, 128), num_train=10, num_test=2)
run_demo(num_epochs=5, batch_size=2, image_shape=(4, 128, 128))

# Example 2: 8 depth slices, 64x64 spatial
generate_dummy_data(image_shape=(8, 64, 64), num_train=10, num_test=2)
run_demo(num_epochs=5, batch_size=2, image_shape=(8, 64, 64))
```

## Running Tests

To verify the implementation works correctly with 3D data:

```bash
python test_components.py
```

This runs unit tests for all core components including:
- 3D U-Net model with various sizes (4x128x128, 8x64x64, etc.)
- Diffusion trainer/sampler with 3D data
- 3D data generation and loading

## Training with Your Own Data

1. Organize your data in the following structure:
```
data/
├── train/
│   ├── a/  # Target volumes (e.g., CT scans)
│   └── b/  # Condition volumes (e.g., CBCT scans)
└── test/
    ├── a/  # Target test volumes
    └── b/  # Condition test volumes
```

2. Images should be saved as `.npy` files (numpy arrays)
   - For 3D data: shape should be (D, H, W) - e.g., (4, 128, 128) or (8, 64, 64)
   - For 2D data: shape should be (H, W) - e.g., (256, 256) (also supported)

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
- **3D U-Net Architecture**: A time-conditioned 3D U-Net with attention mechanisms, residual blocks, and skip connections
- **Flexible Configuration**: Easy to modify hyperparameters like timesteps, channel counts, and attention layers
- **GPU/CPU Support**: Automatically detects and uses GPU if available, falls back to CPU otherwise
- **3D Data Support**: Works with volumetric data of various sizes (e.g., 4x128x128, 8x64x64, etc.)

The model works by:
1. Adding noise to target images/volumes over T timesteps (forward process)
2. Learning to denoise conditioned on the input image/volume (training)
3. Generating new images/volumes from random noise by iterative denoising (inference)

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
- Supports 3D volumetric data with various depth/height/width combinations
- The architecture uses 3D convolutions (Conv3d) for processing volumetric data
- Checkpoints are saved periodically during training
- Test outputs are saved as raw binary files