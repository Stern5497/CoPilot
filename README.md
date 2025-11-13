# CoPilot - Conditional DDPM & CycleGAN Implementation (3D Support)

This repository contains implementations of two powerful models for image-to-image translation:
1. **Conditional Denoising Diffusion Probabilistic Model (DDPM)** - based on [junbopeng/conditional_DDPM](https://github.com/junbopeng/conditional_DDPM)
2. **CycleGAN** - based on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

**Both models support 3D volumetric data!** They work with various 3D image sizes such as (4, 128, 128), (8, 64, 64), or any other depth/height/width combinations.

## Overview

A flexible framework for image-to-image translation supporting both paired (DDPM) and unpaired (CycleGAN) training scenarios. Originally designed for CBCT-to-CT image translation, but can be adapted to various image translation tasks including 3D medical imaging.

## Reference Paper

If you use this code, please cite:
> Peng J, Qiu RLJ, Wynne JF, et al. CBCT-Based synthetic CT image generation using conditional denoising diffusion probabilistic model. Med Phys. 2023; 1-13. https://doi.org/10.1002/mp.16704

## Files

### Conditional DDPM
- `Diffusion_condition.py` - Core diffusion process implementation (training and sampling)
- `Model_condition.py` - 3D U-Net architecture with time embeddings
- `Train_condition.py` - Training script for DDPM
- `Test_condition.py` - Inference/testing script for DDPM

### CycleGAN
- `Model_CycleGAN.py` - 3D CycleGAN implementation with ResNet generators and PatchGAN discriminators
- `demo_multi_model.py` - Unified demo script supporting both models

### Shared Components
- `datasets.py` - Dataset loader for paired image data
- `demo.py` - Original DDPM-only demo script
- `examples_3d_sizes.py` - Examples with different 3D sizes

## Quick Start with Dummy Data

### Option 1: Unified Demo (Supports Both Models)

Run the multi-model demo script with model selection:

```bash
pip install -r requirements.txt

# Run with Conditional DDPM (default)
python demo_multi_model.py --model ddpm --epochs 3

# Run with CycleGAN
python demo_multi_model.py --model cyclegan --epochs 3

# Customize 3D image size
python demo_multi_model.py --model cyclegan --depth 8 --height 64 --width 64 --epochs 3
```

### Option 2: DDPM-Only Demo

Run the original DDPM demo script:

```bash
pip install -r requirements.txt
python demo.py
```

This will:
1. Generate dummy 3D training and test data (shape: 4x64x64)
2. Train a small 3D model for 3 epochs
3. Run inference on test samples
4. Save model checkpoints to `./Checkpoints/demo/`

## Model Comparison

| Feature | Conditional DDPM | CycleGAN |
|---------|-----------------|----------|
| Training Data | Paired images required | Unpaired images supported |
| Architecture | U-Net with diffusion process | ResNet generators + PatchGAN discriminators |
| Training Time | Slower (iterative denoising) | Faster (direct mapping) |
| Quality | High quality, detailed | Good quality, fast inference |
| Use Case | Medical imaging, precise translations | Style transfer, domain adaptation |

### When to Use Which Model:

- **Use DDPM when:**
  - You have paired training data
  - Quality is more important than speed
  - You need precise pixel-level correspondence
  - Medical imaging applications

- **Use CycleGAN when:**
  - You have unpaired training data
  - Speed is important
  - You need bidirectional translation (A↔B)
  - Style transfer or domain adaptation tasks

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
# Run core component tests
python test_components.py

# Run MHA file format support tests
python test_mha_support.py
```

**Core component tests** (`test_components.py`) include:
- 3D U-Net model with various sizes (4x128x128, 8x64x64, etc.)
- Diffusion trainer/sampler with 3D data
- 3D data generation and loading

**MHA support tests** (`test_mha_support.py`) include:
- MHA/MHD file loading
- Mixed format support (.npy and .mha in same dataset)
- DataLoader integration with MHA files
- Different 3D shapes support
- Backward compatibility with .npy files

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

2. Supported file formats:
   - **`.npy`** - NumPy arrays (original format)
   - **`.mha`** - MetaImage format (medical imaging standard)
   - **`.mhd`** - MetaImage header/raw format (medical imaging standard)
   
   For 3D data: shape should be (D, H, W) - e.g., (4, 128, 128) or (8, 64, 64)
   For 2D data: shape should be (H, W) - e.g., (256, 256) (also supported)
   
   Note: MHA/MHD support requires SimpleITK (automatically installed via requirements.txt)

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
- **CycleGAN**: Unpaired image-to-image translation using cycle consistency loss and adversarial training
- **3D Architectures**: Both models use 3D convolutions (Conv3d) for processing volumetric data
  - DDPM: 3D U-Net with time conditioning and attention mechanisms
  - CycleGAN: 3D ResNet generators and 3D PatchGAN discriminators
- **Flexible Configuration**: Easy to modify hyperparameters like timesteps, channel counts, and attention layers
- **GPU/CPU Support**: Automatically detects and uses GPU if available, falls back to CPU otherwise
- **3D Data Support**: Works with volumetric data of various sizes (e.g., 4x128x128, 8x64x64, etc.)

### How DDPM Works:
1. Adding noise to target images/volumes over T timesteps (forward process)
2. Learning to denoise conditioned on the input image/volume (training)
3. Generating new images/volumes from random noise by iterative denoising (inference)

### How CycleGAN Works:
1. Two generators (G_A: A→B, G_B: B→A) and two discriminators (D_A, D_B)
2. Adversarial loss: generators fool discriminators
3. Cycle consistency loss: G_B(G_A(A)) ≈ A and G_A(G_B(B)) ≈ B
4. Optional identity loss: G_A(B) ≈ B and G_B(A) ≈ A

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