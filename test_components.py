"""
Unit tests for conditional DDPM components
"""
import os
import sys
import torch
import numpy as np

# Import the modules
from Diffusion_condition import GaussianDiffusionTrainer_cond, GaussianDiffusionSampler_cond
from Model_condition import UNet


def test_unet_model():
    """Test UNet model initialization and forward pass"""
    print("Testing UNet model...")
    
    T = 50
    ch = 32
    ch_mult = [1, 2, 2]
    attn = [1]
    num_res_blocks = 1
    dropout = 0.1
    
    model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 2, 128, 128)  # 2 channels (noisy image + condition)
    t = torch.randint(0, T, (batch_size,))
    
    output = model(x, t)
    
    assert output.shape == (batch_size, 1, 128, 128), f"Expected shape (2, 1, 128, 128), got {output.shape}"
    print("✓ UNet model test passed")


def test_diffusion_trainer():
    """Test diffusion trainer"""
    print("\nTesting diffusion trainer...")
    
    T = 50
    beta_1 = 1e-4
    beta_T = 0.02
    
    model = UNet(T, 32, [1, 2, 2], [1], 1, 0.1)
    trainer = GaussianDiffusionTrainer_cond(model, beta_1, beta_T, T)
    
    # Test training step
    batch_size = 2
    x_0 = torch.randn(batch_size, 2, 128, 128)  # 2 channels (target + condition)
    
    loss = trainer(x_0)
    
    assert isinstance(loss.item(), float), "Loss should be a float"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Diffusion trainer test passed")


def test_diffusion_sampler():
    """Test diffusion sampler"""
    print("\nTesting diffusion sampler...")
    
    T = 20  # Reduced for faster testing
    beta_1 = 1e-4
    beta_T = 0.02
    
    model = UNet(T, 32, [1, 2], [], 1, 0.1)
    sampler = GaussianDiffusionSampler_cond(model, beta_1, beta_T, T)
    
    # Test sampling
    batch_size = 1
    x_T = torch.randn(batch_size, 2, 64, 64)  # Smaller size for faster test
    
    with torch.no_grad():
        x_0 = sampler(x_T)
    
    assert x_0.shape == x_T.shape, f"Expected shape {x_T.shape}, got {x_0.shape}"
    assert torch.all(x_0 >= -1) and torch.all(x_0 <= 1), "Output should be clipped to [-1, 1]"
    print("✓ Diffusion sampler test passed")


def test_data_generation():
    """Test that we can generate and load dummy data"""
    print("\nTesting data generation...")
    
    from demo import generate_dummy_data
    from datasets import ImageDataset
    from torch.utils.data import DataLoader
    
    # Generate test data
    test_dir = "/tmp/test_data"
    generate_dummy_data(data_dir=test_dir, image_size=64, num_train=3, num_test=1)
    
    # Try to load it
    dataset = ImageDataset(test_dir, transforms_=False, unaligned=True, mode="train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(dataloader))
    assert "a" in batch and "b" in batch, "Batch should contain 'a' and 'b' keys"
    assert batch["a"].shape[0] == 2, "Batch size should be 2"
    assert batch["a"].shape[1] == 1, "Should have 1 channel"
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)
    
    print("✓ Data generation test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Conditional DDPM Tests")
    print("=" * 60)
    
    try:
        test_unet_model()
        test_diffusion_trainer()
        test_diffusion_sampler()
        test_data_generation()
        
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
