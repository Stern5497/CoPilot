"""
Test script for CycleGAN 3D model
"""
import torch
import numpy as np
from Model_CycleGAN import CycleGAN3D, ResnetGenerator3D, NLayerDiscriminator3D

print("=" * 60)
print("Testing CycleGAN 3D Implementation")
print("=" * 60)

# Test 1: Generator forward pass
print("\n1. Testing 3D ResNet Generator...")
gen = ResnetGenerator3D(input_nc=1, output_nc=1, ngf=32, n_blocks=2)
x = torch.randn(2, 1, 4, 64, 64)  # [B, C, D, H, W]
output = gen(x)
assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
assert torch.all(output >= -1) and torch.all(output <= 1), "Output should be in [-1, 1] range (Tanh)"
print(f"   ✓ Generator output shape: {output.shape}")

# Test 2: Discriminator forward pass
print("\n2. Testing 3D PatchGAN Discriminator...")
disc = NLayerDiscriminator3D(input_nc=1, ndf=32, n_layers=2)
x = torch.randn(2, 1, 4, 64, 64)
output = disc(x)
print(f"   ✓ Discriminator output shape: {output.shape}")

# Test 3: Full CycleGAN model
print("\n3. Testing full CycleGAN model...")
model = CycleGAN3D(input_nc=1, output_nc=1, ngf=32, ndf=32, n_blocks=2)
print("   ✓ Model initialized successfully")

# Test 4: Forward pass
print("\n4. Testing forward pass...")
real_A = torch.randn(2, 1, 4, 64, 64)
real_B = torch.randn(2, 1, 4, 64, 64)
model.set_input(real_A, real_B)
model.forward()
assert hasattr(model, 'fake_B'), "fake_B should be generated"
assert hasattr(model, 'rec_A'), "rec_A should be generated"
assert hasattr(model, 'fake_A'), "fake_A should be generated"
assert hasattr(model, 'rec_B'), "rec_B should be generated"
print(f"   ✓ fake_B shape: {model.fake_B.shape}")
print(f"   ✓ rec_A shape: {model.rec_A.shape}")

# Test 5: Training step
print("\n5. Testing training step...")
model.optimize_parameters(lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5)
losses = model.get_current_losses()
print(f"   ✓ Losses: D_A={losses['D_A']:.4f}, G_A={losses['G_A']:.4f}, cycle_A={losses['cycle_A']:.4f}")

# Test 6: Different image sizes
print("\n6. Testing with different 3D sizes...")

# Size 1: (8, 64, 64)
model2 = CycleGAN3D(input_nc=1, output_nc=1, ngf=32, ndf=32, n_blocks=2)
real_A = torch.randn(1, 1, 8, 64, 64)
real_B = torch.randn(1, 1, 8, 64, 64)
model2.set_input(real_A, real_B)
model2.forward()
assert model2.fake_B.shape == (1, 1, 8, 64, 64)
print(f"   ✓ Works with shape (8, 64, 64)")

# Size 2: (4, 128, 128)
model3 = CycleGAN3D(input_nc=1, output_nc=1, ngf=32, ndf=32, n_blocks=2)
real_A = torch.randn(1, 1, 4, 128, 128)
real_B = torch.randn(1, 1, 4, 128, 128)
model3.set_input(real_A, real_B)
model3.forward()
assert model3.fake_B.shape == (1, 1, 4, 128, 128)
print(f"   ✓ Works with shape (4, 128, 128)")

print("\n" + "=" * 60)
print("All CycleGAN tests passed! ✓")
print("=" * 60)
