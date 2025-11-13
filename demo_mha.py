"""
Demo script showing MHA file format support
Generates dummy MHA data and demonstrates loading it with the ImageDataset
"""
import os
import sys
import numpy as np
import tempfile
import shutil

try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False
    print("ERROR: SimpleITK not installed. Install with: pip install SimpleITK")
    sys.exit(1)

from datasets import ImageDataset
from torch.utils.data import DataLoader


def generate_mha_demo_data(data_dir="mha_demo_data", image_shape=(4, 64, 64), num_train=6, num_test=2):
    """
    Generate dummy 3D image data in MHA format for demonstration
    
    Args:
        data_dir: Directory to save data
        image_shape: Shape of 3D images, e.g., (4, 64, 64) or (8, 128, 128)
        num_train: Number of training samples
        num_test: Number of test samples
    """
    print(f"Generating dummy MHA data in {data_dir}...")
    
    # Create directories
    os.makedirs(f"{data_dir}/train/a", exist_ok=True)
    os.makedirs(f"{data_dir}/train/b", exist_ok=True)
    os.makedirs(f"{data_dir}/test/a", exist_ok=True)
    os.makedirs(f"{data_dir}/test/b", exist_ok=True)
    
    # Generate training data
    for i in range(num_train):
        # Image A (target)
        img_a = np.random.randn(*image_shape).astype(np.float32)
        image_a = sitk.GetImageFromArray(img_a)
        sitk.WriteImage(image_a, f"{data_dir}/train/a/img_{i:04d}.mha")
        
        # Image B (condition) - slightly correlated with A
        img_b = img_a + 0.5 * np.random.randn(*image_shape).astype(np.float32)
        image_b = sitk.GetImageFromArray(img_b)
        sitk.WriteImage(image_b, f"{data_dir}/train/b/img_{i:04d}.mha")
    
    # Generate test data
    for i in range(num_test):
        img_a = np.random.randn(*image_shape).astype(np.float32)
        image_a = sitk.GetImageFromArray(img_a)
        sitk.WriteImage(image_a, f"{data_dir}/test/a/img_{i:04d}.mha")
        
        img_b = img_a + 0.5 * np.random.randn(*image_shape).astype(np.float32)
        image_b = sitk.GetImageFromArray(img_b)
        sitk.WriteImage(image_b, f"{data_dir}/test/b/img_{i:04d}.mha")
    
    shape_str = "x".join(map(str, image_shape))
    print(f"✓ Generated {num_train} training samples and {num_test} test samples")
    print(f"  Image shape: {shape_str}")
    print(f"  Format: MHA (MetaImage)")


def demonstrate_mha_loading(data_dir="mha_demo_data"):
    """Demonstrate loading MHA files with the ImageDataset"""
    print(f"\nDemonstrating MHA file loading from {data_dir}...")
    
    # Create dataset
    train_dataset = ImageDataset(data_dir, transforms_=False, unaligned=True, mode="train")
    test_dataset = ImageDataset(data_dir, transforms_=False, unaligned=True, mode="test")
    
    print(f"✓ Loaded dataset:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Get a sample
    sample = train_dataset[0]
    print(f"\n✓ Sample data shape:")
    print(f"  Image A (target): {sample['a'].shape}")
    print(f"  Image B (condition): {sample['b'].shape}")
    
    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    print(f"\n✓ DataLoader created with batch_size=2")
    
    # Get a batch
    batch = next(iter(train_dataloader))
    print(f"  Batch A shape: {batch['a'].shape}")
    print(f"  Batch B shape: {batch['b'].shape}")
    
    print(f"\n✓ MHA data loading successful!")


def demonstrate_mixed_formats(data_dir="mixed_demo_data", image_shape=(4, 64, 64)):
    """Demonstrate loading mixed .npy and .mha files"""
    print(f"\n{'='*60}")
    print(f"Demonstrating mixed format support (.npy + .mha)")
    print(f"{'='*60}")
    
    # Create directories
    os.makedirs(f"{data_dir}/train/a", exist_ok=True)
    os.makedirs(f"{data_dir}/train/b", exist_ok=True)
    
    # Save first sample as .npy
    print(f"Creating sample 1 in .npy format...")
    img_a1 = np.random.randn(*image_shape).astype(np.float32)
    img_b1 = np.random.randn(*image_shape).astype(np.float32)
    np.save(f"{data_dir}/train/a/sample_001.npy", img_a1)
    np.save(f"{data_dir}/train/b/sample_001.npy", img_b1)
    
    # Save second sample as .mha
    print(f"Creating sample 2 in .mha format...")
    img_a2 = np.random.randn(*image_shape).astype(np.float32)
    img_b2 = np.random.randn(*image_shape).astype(np.float32)
    image_a2 = sitk.GetImageFromArray(img_a2)
    image_b2 = sitk.GetImageFromArray(img_b2)
    sitk.WriteImage(image_a2, f"{data_dir}/train/a/sample_002.mha")
    sitk.WriteImage(image_b2, f"{data_dir}/train/b/sample_002.mha")
    
    # Load mixed dataset
    dataset = ImageDataset(data_dir, transforms_=False, unaligned=True, mode="train")
    
    print(f"\n✓ Mixed format dataset loaded:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Sample 0 (from .npy): {dataset[0]['a'].shape}")
    print(f"  Sample 1 (from .mha): {dataset[1]['a'].shape}")
    
    print(f"\n✓ Mixed format loading successful!")


if __name__ == "__main__":
    print("=" * 60)
    print("MHA File Format Support Demo")
    print("=" * 60)
    print()
    
    if not HAS_SIMPLEITK:
        print("SimpleITK is required for MHA support.")
        print("Install it with: pip install SimpleITK")
        sys.exit(1)
    
    # Demo 1: Pure MHA format
    data_dir = "mha_demo_data"
    try:
        generate_mha_demo_data(data_dir, image_shape=(4, 64, 64), num_train=6, num_test=2)
        demonstrate_mha_loading(data_dir)
    finally:
        # Clean up
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"\n✓ Cleaned up demo data directory: {data_dir}")
    
    # Demo 2: Mixed formats
    mixed_dir = "mixed_demo_data"
    try:
        demonstrate_mixed_formats(mixed_dir, image_shape=(4, 64, 64))
    finally:
        # Clean up
        if os.path.exists(mixed_dir):
            shutil.rmtree(mixed_dir)
            print(f"\n✓ Cleaned up demo data directory: {mixed_dir}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
