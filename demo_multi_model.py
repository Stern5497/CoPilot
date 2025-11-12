"""
Demo script with model selection: Conditional DDPM or CycleGAN
Supports 3D volumetric data with flexible image sizes
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse

from datasets import ImageDataset


def generate_dummy_data(data_dir="data", image_shape=(4, 64, 64), num_train=6, num_test=2):
    """
    Generate dummy 3D image data for training and testing
    
    Args:
        data_dir: Directory to save data
        image_shape: Shape of 3D images, e.g., (4, 128, 128) or (8, 64, 64)
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


def run_ddpm_demo(num_epochs=3, batch_size=2, image_shape=(4, 64, 64)):
    """Run Conditional DDPM demo"""
    from Diffusion_condition import GaussianDiffusionTrainer_cond, GaussianDiffusionSampler_cond
    from Model_condition import UNet
    
    print("\n" + "=" * 60)
    print("Running Conditional DDPM Demo")
    print("=" * 60)
    
    # Configuration
    dataset_name = "data"
    out_name = "demo_ddpm"
    T = 50
    ch = 32
    ch_mult = [1, 2, 2]
    attn = [1]
    num_res_blocks = 1
    dropout = 0.1
    lr = 1e-4
    beta_1 = 1e-4
    beta_T = 0.02
    grad_clip = 1
    save_weight_dir = f"./Checkpoints/{out_name}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(save_weight_dir, exist_ok=True)
    
    # Data loader
    train_dataloader = DataLoader(
        ImageDataset(f"./{dataset_name}", transforms_=False, unaligned=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Model setup
    print("Initializing DDPM model...")
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
            
            ct = Variable(batch["a"].type(Tensor))
            cbct = Variable(batch["b"].type(Tensor))
            x_0 = torch.cat((ct, cbct), 1)
            
            loss = trainer(x_0)
            epoch_loss += loss.item() / 65536
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), grad_clip)
            optimizer.step()
            
            num_batches += 1
            
            if epoch == 0 and (i + 1) % 2 == 0:
                print(f"  Batch [{i+1}/{len(train_dataloader)}] - Loss: {loss.item() / 65536:.6f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] - MSE Loss: {avg_loss:.6f}")
    
    # Save model
    model_path = os.path.join(save_weight_dir, f'ckpt_demo.pt')
    torch.save(net_model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Inference
    print("\nRunning inference...")
    net_model.eval()
    sampler = GaussianDiffusionSampler_cond(net_model, beta_1, beta_T, T).to(device)
    
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
            
            noise_shape = list(cbct.shape)
            noisyImage = torch.randn(size=noise_shape, device=device)
            x_in = torch.cat((noisyImage, cbct), 1)
            x_out = sampler(x_in)
            
            print(f"Processed test sample {ii+1}/{len(test_dataloader)}")
    
    print("\nDDPM demo completed successfully!")


def run_cyclegan_demo(num_epochs=3, batch_size=2, image_shape=(4, 64, 64)):
    """Run CycleGAN demo"""
    from Model_CycleGAN import CycleGAN3D
    
    print("\n" + "=" * 60)
    print("Running CycleGAN Demo")
    print("=" * 60)
    
    # Configuration
    dataset_name = "data"
    out_name = "demo_cyclegan"
    save_weight_dir = f"./Checkpoints/{out_name}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(save_weight_dir, exist_ok=True)
    
    # Data loader
    train_dataloader = DataLoader(
        ImageDataset(f"./{dataset_name}", transforms_=False, unaligned=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Model setup
    print("Initializing CycleGAN model...")
    model = CycleGAN3D(input_nc=1, output_nc=1, ngf=32, ndf=32, n_blocks=2, 
                       norm="instance", gan_mode="lsgan")
    
    # Training
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_losses = {'D_A': 0, 'G_A': 0, 'cycle_A': 0, 'D_B': 0, 'G_B': 0, 'cycle_B': 0}
        num_batches = 0
        
        for i, batch in enumerate(train_dataloader):
            real_A = Variable(batch["a"].type(Tensor))
            real_B = Variable(batch["b"].type(Tensor))
            
            model.set_input(real_A, real_B)
            model.optimize_parameters(lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5)
            
            losses = model.get_current_losses()
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            
            num_batches += 1
            
            if epoch == 0 and (i + 1) % 2 == 0:
                print(f"  Batch [{i+1}/{len(train_dataloader)}] - D_A: {losses['D_A']:.4f}, G_A: {losses['G_A']:.4f}, Cycle_A: {losses['cycle_A']:.4f}")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - D_A: {epoch_losses['D_A']:.4f}, G_A: {epoch_losses['G_A']:.4f}, Cycle_A: {epoch_losses['cycle_A']:.4f}")
    
    # Save model
    model.save_networks(save_weight_dir, num_epochs)
    print(f"\nModel saved to {save_weight_dir}")
    
    # Inference
    print("\nRunning inference...")
    model.netG_A.eval()
    model.netG_B.eval()
    
    test_dataloader = DataLoader(
        ImageDataset(f"./{dataset_name}", transforms_=False, unaligned=True, mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    with torch.no_grad():
        for ii, batch in enumerate(test_dataloader):
            real_A = Variable(batch["a"].type(Tensor))
            real_B = Variable(batch["b"].type(Tensor))
            
            fake_B = model.netG_A(real_A)  # A -> B
            fake_A = model.netG_B(real_B)  # B -> A
            
            print(f"Processed test sample {ii+1}/{len(test_dataloader)}")
    
    print("\nCycleGAN demo completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Conditional DDPM or CycleGAN Demo with 3D Support')
    parser.add_argument('--model', type=str, default='ddpm', choices=['ddpm', 'cyclegan'],
                        help='Model to use: ddpm or cyclegan (default: ddpm)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--depth', type=int, default=4, help='Depth dimension (default: 4)')
    parser.add_argument('--height', type=int, default=64, help='Height dimension (default: 64)')
    parser.add_argument('--width', type=int, default=64, help='Width dimension (default: 64)')
    parser.add_argument('--num_train', type=int, default=6, help='Number of training samples (default: 6)')
    parser.add_argument('--num_test', type=int, default=2, help='Number of test samples (default: 2)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Model Demo: 3D Image-to-Image Translation")
    print("=" * 60)
    print(f"Selected model: {args.model.upper()}")
    
    # Generate dummy data
    image_shape = (args.depth, args.height, args.width)
    generate_dummy_data(num_train=args.num_train, num_test=args.num_test, image_shape=image_shape)
    
    # Run selected model
    if args.model == 'ddpm':
        run_ddpm_demo(num_epochs=args.epochs, batch_size=args.batch_size, image_shape=image_shape)
    elif args.model == 'cyclegan':
        run_cyclegan_demo(num_epochs=args.epochs, batch_size=args.batch_size, image_shape=image_shape)
    
    print("\n" + "=" * 60)
    print("All operations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
