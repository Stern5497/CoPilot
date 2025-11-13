"""Dataset readers for different medical image formats."""
import os
import numpy as np

try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False


# Define reader dictionaries for different data sources
READERS = {
    'default': {
        'input': 'input',
        'target': 'target',
    },
    'ct_cbct': {
        'ct': 'ct',
        'cbct': 'cbct',
    },
}


def read_mha_file(filepath):
    """Read MHA/MHD file using SimpleITK."""
    if not HAS_SIMPLEITK:
        raise ImportError(
            "SimpleITK is required to load MHA/MHD files. "
            "Install it with: pip install SimpleITK"
        )
    
    image = sitk.ReadImage(filepath)
    array = sitk.GetArrayFromImage(image)
    
    # Get metadata
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    size = image.GetSize()
    
    params = {
        'spacing': spacing,
        'origin': origin,
        'direction': direction,
        'size': size,
    }
    
    return array.astype(np.float32), params


def read_npy_file(filepath):
    """Read NPY file."""
    array = np.load(filepath, allow_pickle=True).astype(np.float32)
    
    # Create default params
    params = {
        'spacing': (1.0, 1.0, 1.0),
        'origin': (0.0, 0.0, 0.0),
        'direction': tuple(np.eye(3).flatten()),
        'size': array.shape[::-1],  # Reverse for ITK convention
    }
    
    return array, params


def read_files(folder, reader_dict, ext='mha'):
    """
    Read all files for a case.
    
    Args:
        folder: Path to the folder containing the files
        reader_dict: Dictionary mapping names to file prefixes
        ext: File extension ('mha', 'mhd', or 'npy')
    
    Returns:
        tensors: List of tensors
        params: Dictionary of parameters (from first file)
    """
    tensors = []
    params = None
    
    for name, prefix in reader_dict.items():
        filepath = os.path.join(folder, f"{prefix}.{ext}")
        
        if not os.path.exists(filepath):
            # Try alternative extensions
            for alt_ext in ['mha', 'mhd', 'npy']:
                alt_path = os.path.join(folder, f"{prefix}.{alt_ext}")
                if os.path.exists(alt_path):
                    filepath = alt_path
                    ext = alt_ext
                    break
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Read based on extension
        if ext in ['mha', 'mhd']:
            array, file_params = read_mha_file(filepath)
        elif ext == 'npy':
            array, file_params = read_npy_file(filepath)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        import torch
        tensors.append(torch.from_numpy(array))
        
        # Use params from first file
        if params is None:
            params = file_params
    
    return tensors, params


def get_read_func(source):
    """Get the read function for a specific source."""
    return read_files
