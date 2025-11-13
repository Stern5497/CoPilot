"""
Unit tests for conditional DDPM components (3D support)
"""
import os
import sys
import torch
import numpy as np

# Import the modules
from Diffusion_condition import GaussianDiffusionTrainer_cond, GaussianDiffusionSampler_cond
from Model_condition import UNet


def test_unet_model_3d():
    """Test UNet model initialization and forward pass with 3D data"""
    print("Testing UNet model (3D)...")
    
    T = 50
    ch = 32
    ch_mult = [1, 2, 2]
    attn = [1]
    num_res_blocks = 1
    dropout = 0.1
    
    model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout)
    
    # Test forward pass with 3D data
    batch_size = 2
    x = torch.randn(batch_size, 2, 4, 64, 64)  # 2 channels, 4 depth, 64x64 spatial
    t = torch.randint(0, T, (batch_size,))
    
    output = model(x, t)
    
    assert output.shape == (batch_size, 1, 4, 64, 64), f"Expected shape (2, 1, 4, 64, 64), got {output.shape}"
    print("✓ UNet model test (3D) passed")


def test_unet_model_different_sizes():
    """Test UNet model with different 3D sizes"""
    print("\nTesting UNet model with different sizes...")
    
    T = 50
    ch = 32  # Changed from 16 to 32 to be divisible by GroupNorm groups
    ch_mult = [1, 2]
    attn = []
    num_res_blocks = 1
    dropout = 0.1
    
    model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout)
    
    # Test with shape (8, 64, 64)
    batch_size = 1
    x = torch.randn(batch_size, 2, 8, 64, 64)
    t = torch.randint(0, T, (batch_size,))
    output = model(x, t)
    assert output.shape == (batch_size, 1, 8, 64, 64), f"Expected shape (1, 1, 8, 64, 64), got {output.shape}"
    
    # Test with shape (4, 128, 128)
    x = torch.randn(batch_size, 2, 4, 128, 128)
    output = model(x, t)
    assert output.shape == (batch_size, 1, 4, 128, 128), f"Expected shape (1, 1, 4, 128, 128), got {output.shape}"
    
    print("✓ UNet model test with different sizes passed")


def test_diffusion_trainer_3d():
    """Test diffusion trainer with 3D data"""
    print("\nTesting diffusion trainer (3D)...")
    
    T = 50
    beta_1 = 1e-4
    beta_T = 0.02
    
    model = UNet(T, 32, [1, 2, 2], [1], 1, 0.1)
    trainer = GaussianDiffusionTrainer_cond(model, beta_1, beta_T, T)
    
    # Test training step with 3D data
    batch_size = 2
    x_0 = torch.randn(batch_size, 2, 4, 64, 64)  # 2 channels (target + condition)
    
    loss = trainer(x_0)
    
    assert isinstance(loss.item(), float), "Loss should be a float"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Diffusion trainer test (3D) passed")


def test_diffusion_sampler_3d():
    """Test diffusion sampler with 3D data"""
    print("\nTesting diffusion sampler (3D)...")
    
    T = 20  # Reduced for faster testing
    beta_1 = 1e-4
    beta_T = 0.02
    
    model = UNet(T, 32, [1, 2], [], 1, 0.1)
    sampler = GaussianDiffusionSampler_cond(model, beta_1, beta_T, T)
    
    # Test sampling with 3D data
    batch_size = 1
    x_T = torch.randn(batch_size, 2, 4, 32, 32)  # Smaller size for faster test
    
    with torch.no_grad():
        x_0 = sampler(x_T)
    
    assert x_0.shape == x_T.shape, f"Expected shape {x_T.shape}, got {x_0.shape}"
    assert torch.all(x_0 >= -1) and torch.all(x_0 <= 1), "Output should be clipped to [-1, 1]"
    print("✓ Diffusion sampler test (3D) passed")


def test_data_generation_3d():
    """Test that we can generate and load 3D dummy data"""
    print("\nTesting 3D data generation...")
    
    from demo import generate_dummy_data
    from datasets import ImageDataset
    from torch.utils.data import DataLoader
    
    # Generate test data with 3D shape
    test_dir = "/tmp/test_data_3d"
    generate_dummy_data(data_dir=test_dir, image_shape=(4, 64, 64), num_train=3, num_test=1)
    
    # Try to load it
    dataset = ImageDataset(test_dir, transforms_=False, unaligned=True, mode="train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(dataloader))
    assert "a" in batch and "b" in batch, "Batch should contain 'a' and 'b' keys"
    assert batch["a"].shape == (2, 1, 4, 64, 64), f"Expected shape (2, 1, 4, 64, 64), got {batch['a'].shape}"
    assert batch["b"].shape == (2, 1, 4, 64, 64), f"Expected shape (2, 1, 4, 64, 64), got {batch['b'].shape}"
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)
    
    print("✓ 3D data generation test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Conditional DDPM Tests (3D Support)")
    print("=" * 60)
    
    try:
        test_unet_model_3d()
        test_unet_model_different_sizes()
        test_diffusion_trainer_3d()
        test_diffusion_sampler_3d()
        test_data_generation_3d()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
