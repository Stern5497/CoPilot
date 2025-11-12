"""
Example comparing DDPM and CycleGAN on the same dataset
This demonstrates when to use each model
"""
import os
import shutil
from demo_multi_model import generate_dummy_data, run_ddpm_demo, run_cyclegan_demo

def cleanup():
    """Remove existing data directories"""
    for dir_name in ["data", "Checkpoints", "test"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

print("=" * 70)
print("Comparison: Conditional DDPM vs CycleGAN on 3D Medical Imaging")
print("=" * 70)

# Generate dataset
image_shape = (4, 64, 64)
print(f"\nGenerating shared dataset with shape {image_shape}...")
cleanup()
generate_dummy_data(image_shape=image_shape, num_train=8, num_test=2)

# Test 1: Conditional DDPM
print("\n" + "=" * 70)
print("Test 1: Conditional DDPM (Paired Training)")
print("=" * 70)
print("\nCharacteristics:")
print("  - Requires paired training data (A,B correspondence)")
print("  - Slower training (iterative denoising process)")
print("  - High quality output with fine details")
print("  - Best for: Medical imaging, precise pixel correspondence")
print("\nRunning DDPM...")
run_ddpm_demo(num_epochs=2, batch_size=2, image_shape=image_shape)

# Save DDPM checkpoints
if os.path.exists("./Checkpoints/demo_ddpm"):
    shutil.copytree("./Checkpoints/demo_ddpm", "./Checkpoints/comparison_ddpm")

# Test 2: CycleGAN
print("\n" + "=" * 70)
print("Test 2: CycleGAN (Unpaired Training Capable)")
print("=" * 70)
print("\nCharacteristics:")
print("  - Can work with unpaired data (A and B don't need correspondence)")
print("  - Faster training (direct mapping with adversarial loss)")
print("  - Good quality with cycle consistency")
print("  - Best for: Style transfer, domain adaptation, when unpaired data")
print("\nRunning CycleGAN...")
run_cyclegan_demo(num_epochs=2, batch_size=2, image_shape=image_shape)

# Comparison
print("\n" + "=" * 70)
print("Summary: Which Model to Choose?")
print("=" * 70)
print("""
┌─────────────────────┬──────────────────────┬─────────────────────────┐
│ Scenario            │ Recommended Model    │ Reason                  │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ Medical CT/MRI      │ DDPM                 │ Precise correspondence  │
│ Paired datasets     │ DDPM                 │ Can leverage pairing    │
│ Unpaired datasets   │ CycleGAN             │ No pairing required     │
│ Style transfer      │ CycleGAN             │ Faster, flexible        │
│ Real-time inference │ CycleGAN             │ Single forward pass     │
│ High quality needed │ DDPM                 │ Iterative refinement    │
│ Limited data        │ CycleGAN             │ Unpaired augmentation   │
└─────────────────────┴──────────────────────┴─────────────────────────┘

Both models support:
  ✓ 3D volumetric data
  ✓ Flexible image sizes (e.g., 4x128x128, 8x64x64)
  ✓ GPU/CPU training
  ✓ Checkpoint saving/loading
""")

print("\nCheckpoints saved to:")
print("  - ./Checkpoints/comparison_ddpm/")
print("  - ./Checkpoints/demo_cyclegan/")

print("\n" + "=" * 70)
print("Comparison completed!")
print("=" * 70)
