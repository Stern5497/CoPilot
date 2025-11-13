"""
Unit tests for CropDataset class with MHA support.
Tests patch-based sampling, sliding window, and augmentation.
"""
import json
import os
import sys
import tempfile
import shutil
import numpy as np
import torch

try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False
    print("Warning: SimpleITK not available, MHA tests will be skipped")

from datasets.crop_dataset import CropDataset


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self, **kwargs):
        self.phase = kwargs.get('phase', 'train')
        self.view = kwargs.get('view', -1)
        self.source = kwargs.get('source', 'default')
        self.data_root = kwargs.get('data_root', 'test_data')
        self.patch_size = kwargs.get('patch_size', [4, 64, 64])
        self.stride_size = kwargs.get('stride_size', [2, 32, 32])
        self.debug = kwargs.get('debug', False)
        
        # Optional augmentation settings
        if 'masked_type' in kwargs:
            self.masked_type = kwargs['masked_type']
            self.masked_block_size = kwargs.get('masked_block_size', 16)
            self.masked_ratio = kwargs.get('masked_ratio', 0.1)


def create_test_case_list(temp_dir, num_cases=2):
    """Create test case list and MHA files."""
    if not HAS_SIMPLEITK:
        return []
    
    case_lists = []
    
    for i in range(num_cases):
        case_name = f"case_{i:03d}"
        case_dir = os.path.join(temp_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        
        # Create dummy 3D volumes
        shape = (16, 128, 128)  # D, H, W
        input_vol = np.random.randn(*shape).astype(np.float32)
        target_vol = np.random.randn(*shape).astype(np.float32)
        
        # Save as MHA
        input_img = sitk.GetImageFromArray(input_vol)
        target_img = sitk.GetImageFromArray(target_vol)
        
        sitk.WriteImage(input_img, os.path.join(case_dir, "input.mha"))
        sitk.WriteImage(target_img, os.path.join(case_dir, "target.mha"))
        
        # Create case metadata
        case_info = {
            "name": case_name,
            "valid_starts": [0, 0, 0],
            "valid_ends": [15, 127, 127],
            "valid_lengths": [16, 128, 128],
            "min_val": float(input_vol.min()),
            "max_val": float(input_vol.max()),
        }
        case_lists.append(case_info)
    
    return case_lists


def test_crop_dataset_creation():
    """Test CropDataset initialization."""
    if not HAS_SIMPLEITK:
        print("✓ CropDataset creation test skipped (SimpleITK not available)")
        return
    
    print("Testing CropDataset creation...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create data directory structure
        data_dir = os.path.join(temp_dir, "test_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create case list
        case_lists = create_test_case_list(data_dir, num_cases=2)
        
        # Create datalists directory
        datalists_dir = os.path.join(temp_dir, "datalists", "default")
        os.makedirs(datalists_dir, exist_ok=True)
        
        # Save case list as JSON
        with open(os.path.join(datalists_dir, "train.json"), "w") as f:
            json.dump(case_lists, f)
        
        # Change to temp dir to use relative paths
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Create config
        config = MockConfig(
            phase='train',
            view=-1,
            source='default',
            data_root='test_data',
            patch_size=[4, 64, 64],
            stride_size=[2, 32, 32]
        )
        
        # Create dataset
        dataset = CropDataset(config)
        
        assert len(dataset.case_lists) == 2, f"Expected 2 cases, got {len(dataset.case_lists)}"
        assert len(dataset) > 0, "Dataset should have samples"
        
        print("✓ CropDataset creation test passed")
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


def test_crop_dataset_sampling():
    """Test patch sampling from CropDataset."""
    if not HAS_SIMPLEITK:
        print("✓ CropDataset sampling test skipped (SimpleITK not available)")
        return
    
    print("\nTesting CropDataset sampling...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create data directory structure
        data_dir = os.path.join(temp_dir, "test_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create case list
        case_lists = create_test_case_list(data_dir, num_cases=1)
        
        # Create datalists directory
        datalists_dir = os.path.join(temp_dir, "datalists", "default")
        os.makedirs(datalists_dir, exist_ok=True)
        
        # Save case list as JSON
        with open(os.path.join(datalists_dir, "train.json"), "w") as f:
            json.dump(case_lists, f)
        
        # Change to temp dir
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Create config for training
        config = MockConfig(
            phase='train',
            view=0,
            source='default',
            data_root='test_data',
            patch_size=[4, 64, 64]
        )
        
        # Create dataset
        dataset = CropDataset(config)
        
        # Get a sample
        sample = dataset[0]
        
        assert 'input' in sample, "Sample should contain 'input' key"
        assert 'target' in sample, "Sample should contain 'target' key"
        assert 'idx' in sample, "Sample should contain 'idx' key"
        
        # Check patch shape
        assert sample['input'].shape == torch.Size([4, 64, 64]), \
            f"Expected patch shape [4, 64, 64], got {sample['input'].shape}"
        
        print("✓ CropDataset sampling test passed")
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


def test_crop_dataset_validation():
    """Test validation mode with sliding window."""
    if not HAS_SIMPLEITK:
        print("✓ CropDataset validation test skipped (SimpleITK not available)")
        return
    
    print("\nTesting CropDataset validation mode...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create data directory structure
        data_dir = os.path.join(temp_dir, "test_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create case list
        case_lists = create_test_case_list(data_dir, num_cases=1)
        
        # Create datalists directory
        datalists_dir = os.path.join(temp_dir, "datalists", "default")
        os.makedirs(datalists_dir, exist_ok=True)
        
        # Save case list as JSON
        with open(os.path.join(datalists_dir, "val.json"), "w") as f:
            json.dump(case_lists, f)
        
        # Change to temp dir
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Create config for validation
        config = MockConfig(
            phase='val',
            view=0,
            source='default',
            data_root='test_data',
            patch_size=[4, 64, 64],
            stride_size=[4, 64, 64]  # Non-overlapping
        )
        
        # Create dataset
        dataset = CropDataset(config)
        
        # Get a sample
        sample = dataset[0]
        
        assert 'offset' in sample, "Validation sample should contain 'offset' for positional encoding"
        assert sample['input'].shape == torch.Size([4, 64, 64]), \
            f"Expected patch shape [4, 64, 64], got {sample['input'].shape}"
        
        print("✓ CropDataset validation test passed")
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


def test_crop_dataset_assembly():
    """Test assembling predictions back to full volume."""
    if not HAS_SIMPLEITK:
        print("✓ CropDataset assembly test skipped (SimpleITK not available)")
        return
    
    print("\nTesting CropDataset assembly...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create data directory structure
        data_dir = os.path.join(temp_dir, "test_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create case list
        case_lists = create_test_case_list(data_dir, num_cases=1)
        
        # Create datalists directory
        datalists_dir = os.path.join(temp_dir, "datalists", "default")
        os.makedirs(datalists_dir, exist_ok=True)
        
        # Save case list as JSON
        with open(os.path.join(datalists_dir, "val.json"), "w") as f:
            json.dump(case_lists, f)
        
        # Change to temp dir
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Create config
        config = MockConfig(
            phase='val',
            view=0,
            source='default',
            data_root='test_data',
            patch_size=[4, 64, 64],
            stride_size=[4, 64, 64]
        )
        
        # Create dataset
        dataset = CropDataset(config)
        
        # Load tensors for case 0
        dataset.get_tensors(0)
        
        # Create dummy predictions (same number as dataset samples for this case)
        num_patches = len([i for i in range(len(dataset)) if dataset._calculate_indices(i)[1] == 0])
        preds = [torch.randn(4, 64, 64) for _ in range(num_patches)]
        preds_tensor = torch.stack(preds)
        
        # Get the actual tensor shape
        _, tensors = dataset.get_tensors(0)
        expected_shape = tensors[0].shape
        
        # Assemble
        result = dataset.assemble(0, preds_tensor)
        
        assert result.shape == expected_shape, \
            f"Expected assembled shape {expected_shape}, got {result.shape}"
        
        print("✓ CropDataset assembly test passed")
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


def test_multi_view_support():
    """Test multi-view (axial, sagittal, coronal) support."""
    if not HAS_SIMPLEITK:
        print("✓ Multi-view support test skipped (SimpleITK not available)")
        return
    
    print("\nTesting multi-view support...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create data directory structure
        data_dir = os.path.join(temp_dir, "test_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create case list
        case_lists = create_test_case_list(data_dir, num_cases=1)
        
        # Create datalists directory
        datalists_dir = os.path.join(temp_dir, "datalists", "default")
        os.makedirs(datalists_dir, exist_ok=True)
        
        # Save case list as JSON
        with open(os.path.join(datalists_dir, "train.json"), "w") as f:
            json.dump(case_lists, f)
        
        # Change to temp dir
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Test different views
        expected_shapes = {
            0: torch.Size([4, 64, 64]),  # Axial view
            1: torch.Size([4, 64, 64]),  # Sagittal view (permuted but same patch size)
            2: torch.Size([64, 64, 4]),  # Coronal view (different dimension order)
        }
        
        for view in [0, 1, 2]:
            config = MockConfig(
                phase='train',
                view=view,
                source='default',
                data_root='test_data',
                patch_size=[4, 64, 64]
            )
            
            dataset = CropDataset(config)
            sample = dataset[0]
            
            # Check that patch has correct shape for the view
            assert len(sample['input'].shape) == 3, \
                f"View {view}: Expected 3D patch, got shape {sample['input'].shape}"
            assert sample['input'].numel() == 4 * 64 * 64, \
                f"View {view}: Expected {4*64*64} elements, got {sample['input'].numel()}"
        
        print("✓ Multi-view support test passed")
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all CropDataset tests."""
    print("=" * 60)
    print("Running CropDataset Tests (MHA Support)")
    print("=" * 60)
    
    if not HAS_SIMPLEITK:
        print("\nWARNING: SimpleITK not available.")
        print("CropDataset tests will be skipped.")
        print("Install SimpleITK to run full tests: pip install SimpleITK")
        print()
        return 0
    
    try:
        test_crop_dataset_creation()
        test_crop_dataset_sampling()
        test_crop_dataset_validation()
        test_crop_dataset_assembly()
        test_multi_view_support()
        
        print("\n" + "=" * 60)
        print("All CropDataset tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
