"""
Demo script for conditional DDPM with dummy data
This script generates dummy image data and runs a simple training/inference demo
"""
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys

from Diffusion_condition import GaussianDiffusionTrainer_cond, GaussianDiffusionSampler_cond
from Model_condition import UNet
from datasets import ImageDataset


def generate_dummy_data(data_dir="data", image_shape=(4, 128, 128), num_train=6, num_test=2):
    """
    Generate dummy 3D image data for training and testing
    
    Args:
        data_dir: Directory to save data
        image_shape: Shape of 3D images, e.g., (4, 128, 128) or (8, 64, 64)
                    Can also be 2D like (128, 128) for backward compatibility
        num_train: Number of training samples
        num_test: Number of test samples
    """
    print("Generating dummy data...")
    
    # Create directories
    os.makedirs(f"{data_dir}/train/a", exist_ok=True)
    os.makedirs(f"{data_dir}/train/b", exist_ok=True)
    os.makedirs(f"{data_dir}/test/a", exist_ok=True)
    os.makedirs(f"{data_dir}/test/b", exist_ok=True)
    
    # Generate training data
    for i in range(num_train):
        # Image A (target)
        img_a = np.random.randn(*image_shape).astype(np.float32)
        np.save(f"{data_dir}/train/a/img_{i:04d}.npy", img_a)
        
        # Image B (condition) - slightly correlated with A
        img_b = img_a + 0.5 * np.random.randn(*image_shape).astype(np.float32)
        np.save(f"{data_dir}/train/b/img_{i:04d}.npy", img_b)
    
    # Generate test data
    for i in range(num_test):
        img_a = np.random.randn(*image_shape).astype(np.float32)
        np.save(f"{data_dir}/test/a/img_{i:04d}.npy", img_a)
        
        img_b = img_a + 0.5 * np.random.randn(*image_shape).astype(np.float32)
        np.save(f"{data_dir}/test/b/img_{i:04d}.npy", img_b)
    
    shape_str = "x".join(map(str, image_shape))
    print(f"Generated {num_train} training samples and {num_test} test samples (image shape: {shape_str})")


def run_demo(num_epochs=3, batch_size=2, image_shape=(4, 64, 64)):
    """
    Run a simple demo of the conditional DDPM
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        image_shape: Shape of images (D, H, W) for 3D or (H, W) for 2D
    """
    
    # Configuration
    dataset_name = "data"
    out_name = "demo"
    T = 50  # Reduced timesteps for faster demo
    ch = 32  # Reduced channels for faster demo
    ch_mult = [1, 2, 2]
    attn = [1]
    num_res_blocks = 1
    dropout = 0.1
    lr = 1e-4
    beta_1 = 1e-4
    beta_T = 0.02
    grad_clip = 1
    save_weight_dir = f"./Checkpoints/{out_name}"
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(save_weight_dir, exist_ok=True)
    os.makedirs(f"test/{out_name}", exist_ok=True)
    
    # Data loader
    train_dataloader = DataLoader(
        ImageDataset(f"./{dataset_name}", transforms_=False, unaligned=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Model setup
    print("Initializing model...")
    net_model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    trainer = GaussianDiffusionTrainer_cond(net_model, beta_1, beta_T, T).to(device)
    
    # Training
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Get images
            ct = Variable(batch["a"].type(Tensor))
            cbct = Variable(batch["b"].type(Tensor))
            x_0 = torch.cat((ct, cbct), 1)
            
            # Forward pass
            loss = trainer(x_0)
            epoch_loss += loss.item() / 65536
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), grad_clip)
            optimizer.step()
            
            num_batches += 1
            
            # Print progress for first epoch
            if epoch == 0 and (i + 1) % 2 == 0:
                print(f"  Batch [{i+1}/{len(train_dataloader)}] - Loss: {loss.item() / 65536:.6f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] - MSE Loss: {avg_loss:.6f}")
    
    # Save model
    model_path = os.path.join(save_weight_dir, f'ckpt_demo.pt')
    torch.save(net_model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Inference demo
    print("\nRunning inference demo...")
    net_model.eval()
    sampler = GaussianDiffusionSampler_cond(net_model, beta_1, beta_T, T).to(device)
    
    # Test data loader
    test_dataloader = DataLoader(
        ImageDataset(f"./{dataset_name}", transforms_=False, unaligned=True, mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    with torch.no_grad():
        for ii, batch in enumerate(test_dataloader):
            ct = Variable(batch["a"].type(Tensor))
            cbct = Variable(batch["b"].type(Tensor))
            
            # Generate from random noise conditioned on cbct
            # Get the actual shape from the batch (works for both 2D and 3D)
            # cbct shape: [B, 1, D, H, W] for 3D or [B, 1, H, W] for 2D
            noise_shape = list(cbct.shape)
            noisyImage = torch.randn(size=noise_shape, device=device)
            x_in = torch.cat((noisyImage, cbct), 1)
            x_out = sampler(x_in)
            
            print(f"Processed test sample {ii+1}/{len(test_dataloader)}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    print("=" * 60)
    print("Conditional DDPM Demo (3D Support)")
    print("=" * 60)
    
    # Generate dummy 3D data with shape (4, 64, 64)
    image_shape = (4, 64, 64)
    generate_dummy_data(num_train=6, num_test=2, image_shape=image_shape)
    
    # Run demo
    run_demo(num_epochs=3, batch_size=2, image_shape=image_shape)
    
    print("\n" + "=" * 60)
    print("All operations completed successfully!")
    print("=" * 60)
