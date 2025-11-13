"""Masking functions for data augmentation."""
import numpy as np
import torch


def no_mask(shape, *args):
    """No masking - return all ones."""
    return torch.ones(1, 1, *shape)


def square_mask(shape, block_size):
    """Create a square mask with a single block removed."""
    mask = torch.ones(1, 1, *shape)
    h, w = shape
    
    # Random position for the square
    max_h = max(h - block_size, 0)
    max_w = max(w - block_size, 0)
    
    if max_h > 0 and max_w > 0:
        start_h = np.random.randint(0, max_h)
        start_w = np.random.randint(0, max_w)
        mask[:, :, start_h:start_h+block_size, start_w:start_w+block_size] = 0
    
    return mask


def block_mask(shape, block_size, ratio):
    """Create a block mask with multiple blocks removed."""
    mask = torch.ones(1, 1, *shape)
    h, w = shape
    
    num_blocks = int(h * w * ratio / (block_size * block_size))
    
    for _ in range(num_blocks):
        max_h = max(h - block_size, 1)
        max_w = max(w - block_size, 1)
        
        start_h = np.random.randint(0, max_h)
        start_w = np.random.randint(0, max_w)
        mask[:, :, start_h:start_h+block_size, start_w:start_w+block_size] = 0
    
    return mask


def shift_mask(shape, block_size, ratio):
    """Create a shifted mask pattern."""
    mask = torch.ones(1, 1, *shape)
    h, w = shape
    
    # Create checkerboard-like pattern with shifts
    step = int(block_size / ratio)
    
    for i in range(0, h, step):
        for j in range(0, w, step):
            if np.random.rand() < ratio:
                end_i = min(i + block_size, h)
                end_j = min(j + block_size, w)
                mask[:, :, i:end_i, j:end_j] = 0
    
    return mask


def random_mask(shape, block_size, ratio):
    """Create a random mask."""
    mask = torch.rand(1, 1, *shape) > ratio
    return mask.float()
