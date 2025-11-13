"""Utility functions for dataset processing."""
import numpy as np
import torch
import torch.nn.functional as F


def rescale01(tensor, min_val, max_val):
    """Rescale tensor to [0, 1] range."""
    return (tensor - min_val) / (max_val - min_val + 1e-8)


def rescale_back(tensor, min_val, max_val):
    """Rescale tensor back to original range."""
    return tensor * (max_val - min_val) + min_val


def get_size_with_view(size, view):
    """Get size for specific view (permutation)."""
    if isinstance(size, int):
        return np.array([size, size, size], dtype=np.int32)
    size = np.array(size, dtype=np.int32)
    if view == 0:
        return size
    elif view == 1:
        return size[[1, 0, 2]]
    elif view == 2:
        return size[[2, 0, 1]]
    return size


def crop_and_pad(tensor, target_size):
    """Crop or pad tensor to target size."""
    current_size = np.array(tensor.shape)
    target_size = np.array(target_size)
    
    # Calculate crop/pad amounts
    diff = current_size - target_size
    
    # Crop if larger
    if (diff > 0).any():
        start = np.maximum(diff // 2, 0).astype(int)
        end = (start + target_size).astype(int)
        tensor = tensor[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    # Pad if smaller
    current_size = np.array(tensor.shape)
    diff = target_size - current_size
    if (diff > 0).any():
        pad = np.maximum(diff, 0).astype(int)
        pad_start = pad // 2
        pad_end = pad - pad_start
        pads = [pad_start[2], pad_end[2], pad_start[1], pad_end[1], pad_start[0], pad_end[0]]
        tensor = F.pad(tensor, pads)
    
    return tensor


def crop_and_pad_with_params(tensor, params):
    """Crop and pad tensor using pre-computed parameters."""
    starts = params['starts']
    ends = params['ends']
    pads = params['pads']
    
    # Crop
    tensor = tensor[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    
    # Pad if needed
    if (pads > 0).any():
        pad_list = [pads[2], pads[2], pads[1], pads[1], pads[0], pads[0]]
        tensor = F.pad(tensor, pad_list)
    
    return tensor


def get_crop_pad_params(current_size, target_size):
    """Get crop and pad parameters."""
    current_size = np.array(current_size)
    target_size = np.array(target_size)
    
    diff = current_size - target_size
    starts = np.maximum(diff // 2, 0).astype(int)
    ends = np.minimum(starts + target_size, current_size).astype(int)
    
    actual_size = ends - starts
    pads = np.maximum(target_size - actual_size, 0).astype(int)
    
    return {'starts': starts, 'ends': ends, 'pads': pads}


def sample_coords(config, view, shape):
    """Sample random coordinates for augmentation."""
    patch_size = get_size_with_view(config.patch_size, view)
    shape = np.array(shape)
    
    # Random center position
    max_offset = np.maximum(shape - patch_size, 0)
    offset = np.random.rand(3) * max_offset
    
    # Create grid
    z, y, x = patch_size
    grid_z = torch.linspace(-1, 1, z)
    grid_y = torch.linspace(-1, 1, y)
    grid_x = torch.linspace(-1, 1, x)
    
    grid = torch.stack(torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij'), dim=-1)
    
    # Add random rotation and scaling if configured
    if hasattr(config, 'aug_rotation') and config.aug_rotation > 0:
        angle = (np.random.rand() - 0.5) * 2 * config.aug_rotation * np.pi / 180
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
        
        # Apply rotation to y-x plane
        grid_yx = grid[..., [1, 2]]
        grid_yx = torch.matmul(grid_yx, rot_matrix.T)
        grid[..., 1] = grid_yx[..., 0]
        grid[..., 2] = grid_yx[..., 1]
    
    if hasattr(config, 'aug_scale') and config.aug_scale > 0:
        scale = 1 + (np.random.rand() - 0.5) * 2 * config.aug_scale
        grid = grid * scale
    
    # Normalize coordinates to actual volume space
    center = offset + patch_size / 2
    normalized_center = 2 * center / shape - 1
    
    # Offset grid to center
    for i in range(3):
        grid[..., i] = grid[..., i] * (patch_size[i] / shape[i]) + normalized_center[i]
    
    # Reshape for grid_sample [1, D, H, W, 3]
    grid = grid[None, ...]
    
    return grid, offset
