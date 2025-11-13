"""
Unit tests for MHA data support
Tests the loading and processing of MHA (MetaImage) format files
"""
import os
import sys
import torch
import numpy as np
import tempfile
import shutil

try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False
    print("Warning: SimpleITK not available, MHA tests will be skipped")

from datasets import ImageDataset
from torch.utils.data import DataLoader


def test_mha_file_loading():
    """Test loading MHA files using ImageDataset"""
    if not HAS_SIMPLEITK:
        print("✓ MHA file loading test skipped (SimpleITK not available)")
        return
        
    print("Testing MHA file loading...")
    
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create directories
        os.makedirs(os.path.join(test_dir, "train/a"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "train/b"), exist_ok=True)
        
        # Generate test 3D data (4 depth slices, 32x32 spatial)
        shape = (4, 32, 32)
        data_a = np.random.randn(*shape).astype(np.float32)
        data_b = np.random.randn(*shape).astype(np.float32)
        
        # Save as MHA files using SimpleITK
        image_a = sitk.GetImageFromArray(data_a)
        image_b = sitk.GetImageFromArray(data_b)
        
        mha_path_a = os.path.join(test_dir, "train/a/test_001.mha")
        mha_path_b = os.path.join(test_dir, "train/b/test_001.mha")
        
        sitk.WriteImage(image_a, mha_path_a)
        sitk.WriteImage(image_b, mha_path_b)
        
        # Load using ImageDataset
        dataset = ImageDataset(test_dir, transforms_=False, unaligned=True, mode="train")
        
        assert len(dataset) == 1, f"Expected 1 sample, got {len(dataset)}"
        
        # Get a sample
        sample = dataset[0]
        
        assert "a" in sample and "b" in sample, "Sample should contain 'a' and 'b' keys"
        assert sample["a"].shape == (1, 4, 32, 32), f"Expected shape (1, 4, 32, 32), got {sample['a'].shape}"
        assert sample["b"].shape == (1, 4, 32, 32), f"Expected shape (1, 4, 32, 32), got {sample['b'].shape}"
        
        # Verify data is loaded correctly (within floating point precision)
        loaded_a = sample["a"].squeeze(0).numpy()
        assert np.allclose(loaded_a, data_a, rtol=1e-5, atol=1e-6), "Loaded data doesn't match original"
        
        print("✓ MHA file loading test passed")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)


def test_mhd_file_loading():
    """Test loading MHD files (MetaImage header/raw format)"""
    if not HAS_SIMPLEITK:
        print("✓ MHD file loading test skipped (SimpleITK not available)")
        return
        
    print("\nTesting MHD file loading...")
    
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create directories
        os.makedirs(os.path.join(test_dir, "train/a"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "train/b"), exist_ok=True)
        
        # Generate test 3D data (8 depth slices, 64x64 spatial)
        shape = (8, 64, 64)
        data_a = np.random.randn(*shape).astype(np.float32)
        data_b = np.random.randn(*shape).astype(np.float32)
        
        # Save as MHD files using SimpleITK
        image_a = sitk.GetImageFromArray(data_a)
        image_b = sitk.GetImageFromArray(data_b)
        
        mhd_path_a = os.path.join(test_dir, "train/a/test_001.mhd")
        mhd_path_b = os.path.join(test_dir, "train/b/test_001.mhd")
        
        sitk.WriteImage(image_a, mhd_path_a)
        sitk.WriteImage(image_b, mhd_path_b)
        
        # Load using ImageDataset
        dataset = ImageDataset(test_dir, transforms_=False, unaligned=True, mode="train")
        
        assert len(dataset) == 1, f"Expected 1 sample, got {len(dataset)}"
        
        # Get a sample
        sample = dataset[0]
        
        assert "a" in sample and "b" in sample, "Sample should contain 'a' and 'b' keys"
        assert sample["a"].shape == (1, 8, 64, 64), f"Expected shape (1, 8, 64, 64), got {sample['a'].shape}"
        assert sample["b"].shape == (1, 8, 64, 64), f"Expected shape (1, 8, 64, 64), got {sample['b'].shape}"
        
        print("✓ MHD file loading test passed")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)


def test_mixed_format_loading():
    """Test loading mixed .npy and .mha files in the same dataset"""
    if not HAS_SIMPLEITK:
        print("✓ Mixed format loading test skipped (SimpleITK not available)")
        return
        
    print("\nTesting mixed format loading...")
    
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create directories
        os.makedirs(os.path.join(test_dir, "train/a"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "train/b"), exist_ok=True)
        
        shape = (4, 32, 32)
        
        # Save first sample as .npy
        data_a1 = np.random.randn(*shape).astype(np.float32)
        data_b1 = np.random.randn(*shape).astype(np.float32)
        np.save(os.path.join(test_dir, "train/a/sample_001.npy"), data_a1)
        np.save(os.path.join(test_dir, "train/b/sample_001.npy"), data_b1)
        
        # Save second sample as .mha
        data_a2 = np.random.randn(*shape).astype(np.float32)
        data_b2 = np.random.randn(*shape).astype(np.float32)
        image_a2 = sitk.GetImageFromArray(data_a2)
        image_b2 = sitk.GetImageFromArray(data_b2)
        sitk.WriteImage(image_a2, os.path.join(test_dir, "train/a/sample_002.mha"))
        sitk.WriteImage(image_b2, os.path.join(test_dir, "train/b/sample_002.mha"))
        
        # Load using ImageDataset
        dataset = ImageDataset(test_dir, transforms_=False, unaligned=True, mode="train")
        
        assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
        
        # Verify both samples load correctly
        for i in range(2):
            sample = dataset[i]
            assert "a" in sample and "b" in sample, f"Sample {i} should contain 'a' and 'b' keys"
            assert sample["a"].shape == (1, 4, 32, 32), f"Sample {i}: Expected shape (1, 4, 32, 32), got {sample['a'].shape}"
            assert sample["b"].shape == (1, 4, 32, 32), f"Sample {i}: Expected shape (1, 4, 32, 32), got {sample['b'].shape}"
        
        print("✓ Mixed format loading test passed")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)


def test_mha_dataloader():
    """Test MHA files with PyTorch DataLoader"""
    if not HAS_SIMPLEITK:
        print("✓ MHA DataLoader test skipped (SimpleITK not available)")
        return
        
    print("\nTesting MHA files with DataLoader...")
    
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create directories
        os.makedirs(os.path.join(test_dir, "train/a"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "train/b"), exist_ok=True)
        
        # Generate multiple test samples
        num_samples = 5
        shape = (4, 32, 32)
        
        for i in range(num_samples):
            data_a = np.random.randn(*shape).astype(np.float32)
            data_b = np.random.randn(*shape).astype(np.float32)
            
            image_a = sitk.GetImageFromArray(data_a)
            image_b = sitk.GetImageFromArray(data_b)
            
            sitk.WriteImage(image_a, os.path.join(test_dir, f"train/a/sample_{i:03d}.mha"))
            sitk.WriteImage(image_b, os.path.join(test_dir, f"train/b/sample_{i:03d}.mha"))
        
        # Load using DataLoader
        dataset = ImageDataset(test_dir, transforms_=False, unaligned=True, mode="train")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert "a" in batch and "b" in batch, "Batch should contain 'a' and 'b' keys"
            assert batch["a"].shape[1:] == (1, 4, 32, 32), f"Unexpected batch shape: {batch['a'].shape}"
            assert batch["b"].shape[1:] == (1, 4, 32, 32), f"Unexpected batch shape: {batch['b'].shape}"
        
        expected_batches = (num_samples + 1) // 2  # Ceiling division
        assert batch_count == expected_batches, f"Expected {expected_batches} batches, got {batch_count}"
        
        print("✓ MHA DataLoader test passed")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)


def test_different_mha_shapes():
    """Test MHA files with different 3D shapes"""
    if not HAS_SIMPLEITK:
        print("✓ Different MHA shapes test skipped (SimpleITK not available)")
        return
        
    print("\nTesting different MHA shapes...")
    
    shapes_to_test = [
        (4, 64, 64),
        (8, 32, 32),
        (16, 128, 128),
        (2, 256, 256),
    ]
    
    for shape in shapes_to_test:
        test_dir = tempfile.mkdtemp()
        
        try:
            # Create directories
            os.makedirs(os.path.join(test_dir, "train/a"), exist_ok=True)
            os.makedirs(os.path.join(test_dir, "train/b"), exist_ok=True)
            
            # Generate test data
            data_a = np.random.randn(*shape).astype(np.float32)
            data_b = np.random.randn(*shape).astype(np.float32)
            
            # Save as MHA
            image_a = sitk.GetImageFromArray(data_a)
            image_b = sitk.GetImageFromArray(data_b)
            
            sitk.WriteImage(image_a, os.path.join(test_dir, "train/a/test.mha"))
            sitk.WriteImage(image_b, os.path.join(test_dir, "train/b/test.mha"))
            
            # Load and verify
            dataset = ImageDataset(test_dir, transforms_=False, unaligned=True, mode="train")
            sample = dataset[0]
            
            expected_shape = (1,) + shape
            assert sample["a"].shape == expected_shape, f"For shape {shape}: Expected {expected_shape}, got {sample['a'].shape}"
            assert sample["b"].shape == expected_shape, f"For shape {shape}: Expected {expected_shape}, got {sample['b'].shape}"
            
        finally:
            # Clean up
            shutil.rmtree(test_dir)
    
    print(f"✓ Different MHA shapes test passed ({len(shapes_to_test)} shapes tested)")


def test_backward_compatibility():
    """Test that existing .npy functionality still works"""
    print("\nTesting backward compatibility with .npy files...")
    
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create directories
        os.makedirs(os.path.join(test_dir, "train/a"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "train/b"), exist_ok=True)
        
        # Generate test 3D data
        shape = (4, 64, 64)
        data_a = np.random.randn(*shape).astype(np.float32)
        data_b = np.random.randn(*shape).astype(np.float32)
        
        # Save as .npy files
        np.save(os.path.join(test_dir, "train/a/test_001.npy"), data_a)
        np.save(os.path.join(test_dir, "train/b/test_001.npy"), data_b)
        
        # Load using ImageDataset
        dataset = ImageDataset(test_dir, transforms_=False, unaligned=True, mode="train")
        
        assert len(dataset) == 1, f"Expected 1 sample, got {len(dataset)}"
        
        # Get a sample
        sample = dataset[0]
        
        assert "a" in sample and "b" in sample, "Sample should contain 'a' and 'b' keys"
        assert sample["a"].shape == (1, 4, 64, 64), f"Expected shape (1, 4, 64, 64), got {sample['a'].shape}"
        assert sample["b"].shape == (1, 4, 64, 64), f"Expected shape (1, 4, 64, 64), got {sample['b'].shape}"
        
        # Verify data is loaded correctly
        loaded_a = sample["a"].squeeze(0).numpy()
        assert np.allclose(loaded_a, data_a, rtol=1e-5, atol=1e-6), "Loaded data doesn't match original"
        
        print("✓ Backward compatibility test passed")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)


def run_all_tests():
    """Run all MHA support tests"""
    print("=" * 60)
    print("Running MHA Data Support Tests")
    print("=" * 60)
    
    if not HAS_SIMPLEITK:
        print("\nWARNING: SimpleITK not available.")
        print("Most MHA tests will be skipped.")
        print("Install SimpleITK to run full tests: pip install SimpleITK")
        print()
    
    try:
        test_backward_compatibility()
        test_mha_file_loading()
        test_mhd_file_loading()
        test_mixed_format_loading()
        test_mha_dataloader()
        test_different_mha_shapes()
        
        print("\n" + "=" * 60)
        print("All MHA tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
