"""
Example usage of the unified CropDataset class.
Demonstrates patch-based training and inference with MHA files.
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
    print("ERROR: SimpleITK not installed. Install with: pip install SimpleITK")
    sys.exit(1)

from datasets import CropDataset


class ExampleConfig:
    """Example configuration for CropDataset."""
    def __init__(self, **kwargs):
        # Required parameters
        self.phase = kwargs.get('phase', 'train')
        self.source = kwargs.get('source', 'default')
        self.data_root = kwargs.get('data_root', 'example_data')
        
        # View configuration
        self.view = kwargs.get('view', -1)  # -1 = all views
        
        # Patch configuration
        self.patch_size = kwargs.get('patch_size', [4, 64, 64])
        self.stride_size = kwargs.get('stride_size', [2, 32, 32])
        
        # Augmentation (training only)
        if self.phase == 'train':
            self.masked_type = kwargs.get('masked_type', 'block')
            self.masked_block_size = kwargs.get('masked_block_size', 16)
            self.masked_ratio = kwargs.get('masked_ratio', 0.1)
            self.aug_rotation = kwargs.get('aug_rotation', 10)
            self.aug_scale = kwargs.get('aug_scale', 0.1)


def create_example_data():
    """Create example MHA data for demonstration."""
    print("Creating example MHA data...")
    
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "example_data")
    
    # Create case directories
    for i in range(3):
        case_name = f"patient_{i:03d}"
        case_dir = os.path.join(data_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        
        # Create 3D volumes (simulated CT and CBCT)
        shape = (32, 256, 256)  # Z, Y, X
        ct_volume = np.random.randn(*shape).astype(np.float32) * 500 + 50
        cbct_volume = ct_volume + np.random.randn(*shape).astype(np.float32) * 100
        
        # Save as MHA files
        ct_img = sitk.GetImageFromArray(ct_volume)
        cbct_img = sitk.GetImageFromArray(cbct_volume)
        
        # Set realistic spacing (mm)
        ct_img.SetSpacing((1.0, 1.0, 2.5))
        cbct_img.SetSpacing((1.0, 1.0, 2.5))
        
        sitk.WriteImage(ct_img, os.path.join(case_dir, "target.mha"))
        sitk.WriteImage(cbct_img, os.path.join(case_dir, "input.mha"))
    
    # Create case list JSON files
    case_lists = {
        'train': [],
        'val': [],
    }
    
    for i in range(3):
        case_info = {
            "name": f"patient_{i:03d}",
            "valid_starts": [0, 0, 0],
            "valid_ends": [31, 255, 255],
            "valid_lengths": [32, 256, 256],
            "min_val": -500.0,
            "max_val": 2000.0,
        }
        
        # First 2 for training, last 1 for validation
        if i < 2:
            case_lists['train'].append(case_info)
        else:
            case_lists['val'].append(case_info)
    
    # Save JSON files
    datalists_dir = os.path.join(temp_dir, "datalists", "default")
    os.makedirs(datalists_dir, exist_ok=True)
    
    for split, cases in case_lists.items():
        json_path = os.path.join(datalists_dir, f"{split}.json")
        with open(json_path, "w") as f:
            json.dump(cases, f, indent=2)
    
    print(f"✓ Created example data in {temp_dir}")
    print(f"  - {len(case_lists['train'])} training cases")
    print(f"  - {len(case_lists['val'])} validation cases")
    
    return temp_dir


def example_training():
    """Example: Training with random augmented patches."""
    print("\n" + "="*60)
    print("Example 1: Training Mode (Random Augmented Patches)")
    print("="*60)
    
    temp_dir = create_example_data()
    original_dir = os.getcwd()
    
    try:
        os.chdir(temp_dir)
        
        # Create training configuration
        config = ExampleConfig(
            phase='train',
            view=0,  # Axial view only
            data_root='example_data',
            patch_size=[8, 128, 128],
            masked_type='block',
            masked_block_size=16,
            masked_ratio=0.15,
        )
        
        # Create dataset
        dataset = CropDataset(config)
        
        print(f"\nDataset created:")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Number of cases: {len(dataset.case_lists)}")
        
        # Get a few samples
        print(f"\nSampling training patches:")
        for i in range(3):
            sample = dataset[i]
            print(f"  Sample {i}:")
            print(f"    - Input shape: {sample['input'].shape}")
            print(f"    - Target shape: {sample['target'].shape}")
            print(f"    - Case index: {sample['idx']}")
            print(f"    - Value range: [{sample['input'].min():.2f}, {sample['input'].max():.2f}]")
        
        print("\n✓ Training mode example completed")
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


def example_validation():
    """Example: Validation with sliding window patches."""
    print("\n" + "="*60)
    print("Example 2: Validation Mode (Sliding Window)")
    print("="*60)
    
    temp_dir = create_example_data()
    original_dir = os.getcwd()
    
    try:
        os.chdir(temp_dir)
        
        # Create validation configuration
        config = ExampleConfig(
            phase='val',
            view=0,  # Axial view
            data_root='example_data',
            patch_size=[8, 128, 128],
            stride_size=[4, 64, 64],  # 50% overlap
        )
        
        # Create dataset
        dataset = CropDataset(config)
        
        print(f"\nDataset created:")
        print(f"  - Total patches: {len(dataset)}")
        
        # Get patches and show positional info
        print(f"\nSampling validation patches:")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"  Patch {i}:")
            print(f"    - Shape: {sample['input'].shape}")
            print(f"    - Case: {sample['idx']}")
            print(f"    - Offset: {sample['offset'].numpy()}")
        
        print("\n✓ Validation mode example completed")
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


def example_inference():
    """Example: Full inference pipeline with patch assembly."""
    print("\n" + "="*60)
    print("Example 3: Inference with Patch Assembly")
    print("="*60)
    
    temp_dir = create_example_data()
    original_dir = os.getcwd()
    
    try:
        os.chdir(temp_dir)
        
        # Create validation configuration
        config = ExampleConfig(
            phase='val',
            view=0,
            data_root='example_data',
            patch_size=[8, 128, 128],
            stride_size=[8, 128, 128],  # Non-overlapping for speed
        )
        
        # Create dataset
        dataset = CropDataset(config)
        
        # Simulate inference on first case
        case_idx = 0
        print(f"\nRunning inference on case {case_idx}...")
        
        # Load case
        case_name, tensors = dataset.get_tensors(case_idx)
        print(f"  Case name: {case_name}")
        print(f"  Volume shape: {tensors[0].shape}")
        
        # Collect patches for this case
        predictions = []
        num_patches = 0
        
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Only process patches from this case
            if sample['idx'] == case_idx:
                # Simulate model prediction (identity for demonstration)
                pred = sample['input'].clone()
                predictions.append(pred)
                num_patches += 1
        
        print(f"  Processed {num_patches} patches")
        
        # Stack predictions
        predictions_tensor = torch.stack(predictions)
        print(f"  Predictions tensor shape: {predictions_tensor.shape}")
        
        # Assemble into full volume
        assembled = dataset.assemble(case_idx, predictions_tensor)
        print(f"  Assembled volume shape: {assembled.shape}")
        
        # Reconstruct to original size
        normalization_factors = {'min': -500.0, 'max': 2000.0}
        reconstructed = dataset.reconstruct(case_idx, assembled, normalization_factors)
        print(f"  Reconstructed shape: {reconstructed.shape}")
        
        print("\n✓ Inference example completed")
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


def example_multiview():
    """Example: Multi-view (axial, sagittal, coronal) processing."""
    print("\n" + "="*60)
    print("Example 4: Multi-View Processing")
    print("="*60)
    
    temp_dir = create_example_data()
    original_dir = os.getcwd()
    
    try:
        os.chdir(temp_dir)
        
        view_names = {0: 'Axial', 1: 'Sagittal', 2: 'Coronal', -1: 'All Views'}
        
        for view_idx in [0, 1, 2, -1]:
            config = ExampleConfig(
                phase='train',
                view=view_idx,
                data_root='example_data',
                patch_size=[8, 128, 128],
            )
            
            dataset = CropDataset(config)
            sample = dataset[0]
            
            print(f"\n{view_names[view_idx]} (view={view_idx}):")
            print(f"  - Dataset size: {len(dataset)}")
            print(f"  - Patch shape: {sample['input'].shape}")
        
        print("\n✓ Multi-view example completed")
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("="*60)
    print("CropDataset Usage Examples")
    print("="*60)
    
    if not HAS_SIMPLEITK:
        print("\nERROR: SimpleITK is required for these examples.")
        print("Install it with: pip install SimpleITK")
        sys.exit(1)
    
    # Run all examples
    example_training()
    example_validation()
    example_inference()
    example_multiview()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. Single CropDataset class handles all modes")
    print("  2. MHA files loaded automatically via SimpleITK")
    print("  3. Training uses random augmented patches")
    print("  4. Validation uses sliding window patches")
    print("  5. Full inference pipeline with assembly supported")
    print("  6. Multi-view processing available")
