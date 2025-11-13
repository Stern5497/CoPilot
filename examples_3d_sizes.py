"""
Example: Using Conditional DDPM with Different 3D Image Sizes

This script demonstrates how to use the conditional DDPM with various
3D image sizes as requested.
"""
from demo import generate_dummy_data, run_demo
import os
import shutil

def cleanup():
    """Remove existing data directories"""
    for dir_name in ["data", "Checkpoints", "test"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

print("=" * 70)
print("Conditional DDPM - 3D Image Size Examples")
print("=" * 70)

# Example 1: Small depth, large spatial (4x128x128)
print("\n" + "=" * 70)
print("Example 1: Shape (4, 128, 128) - 4 depth slices, 128x128 spatial")
print("=" * 70)
cleanup()
generate_dummy_data(image_shape=(4, 128, 128), num_train=6, num_test=2)
run_demo(num_epochs=2, batch_size=2, image_shape=(4, 128, 128))

# Example 2: More depth, smaller spatial (8x64x64)
print("\n" + "=" * 70)
print("Example 2: Shape (8, 64, 64) - 8 depth slices, 64x64 spatial")
print("=" * 70)
cleanup()
generate_dummy_data(image_shape=(8, 64, 64), num_train=6, num_test=2)
run_demo(num_epochs=2, batch_size=2, image_shape=(8, 64, 64))

# Example 3: Custom size (6x96x96)
print("\n" + "=" * 70)
print("Example 3: Shape (6, 96, 96) - Custom size")
print("=" * 70)
cleanup()
generate_dummy_data(image_shape=(6, 96, 96), num_train=6, num_test=2)
run_demo(num_epochs=2, batch_size=2, image_shape=(6, 96, 96))

print("\n" + "=" * 70)
print("All examples completed successfully!")
print("The model works with any 3D image size (D, H, W)")
print("=" * 70)
