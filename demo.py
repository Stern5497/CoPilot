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


def generate_dummy_data(data_dir="data", image_size=128, num_train=6, num_test=2):
    """Generate dummy image data for training and testing"""
    print("Generating dummy data...")
    
    # Create directories
    os.makedirs(f"{data_dir}/train/a", exist_ok=True)
    os.makedirs(f"{data_dir}/train/b", exist_ok=True)
    os.makedirs(f"{data_dir}/test/a", exist_ok=True)
    os.makedirs(f"{data_dir}/test/b", exist_ok=True)
    
    # Generate training data
    for i in range(num_train):
        # Image A (target)
        img_a = np.random.randn(image_size, image_size).astype(np.float32)
        np.save(f"{data_dir}/train/a/img_{i:04d}.npy", img_a)
        
        # Image B (condition) - slightly correlated with A
        img_b = img_a + 0.5 * np.random.randn(image_size, image_size).astype(np.float32)
        np.save(f"{data_dir}/train/b/img_{i:04d}.npy", img_b)
    
    # Generate test data
    for i in range(num_test):
        img_a = np.random.randn(image_size, image_size).astype(np.float32)
        np.save(f"{data_dir}/test/a/img_{i:04d}.npy", img_a)
        
        img_b = img_a + 0.5 * np.random.randn(image_size, image_size).astype(np.float32)
        np.save(f"{data_dir}/test/b/img_{i:04d}.npy", img_b)
    
    print(f"Generated {num_train} training samples and {num_test} test samples (image size: {image_size}x{image_size})")


def run_demo(num_epochs=3, batch_size=2):
    """Run a simple demo of the conditional DDPM"""
    
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
            # Get the actual image size from the batch
            img_size = cbct.shape[-1]
            noisyImage = torch.randn(size=[1, 1, img_size, img_size], device=device)
            x_in = torch.cat((noisyImage, cbct), 1)
            x_out = sampler(x_in)
            
            print(f"Processed test sample {ii+1}/{len(test_dataloader)}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    print("=" * 60)
    print("Conditional DDPM Demo")
    print("=" * 60)
    
    # Generate dummy data
    generate_dummy_data(num_train=6, num_test=2, image_size=128)
    
    # Run demo
    run_demo(num_epochs=3, batch_size=2)
    
    print("\n" + "=" * 60)
    print("All operations completed successfully!")
    print("=" * 60)
