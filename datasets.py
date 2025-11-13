# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
# Modified to support 3D data and MHA format
import glob
import random
import os
import numpy as np #V
import torch as torch #V

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        self.unaligned = unaligned

        # Get all files, then filter out .raw files (they are companion files to .mhd)
        all_files_A = sorted(glob.glob(os.path.join(root, "%s/a" % mode) + "/*.*"))
        all_files_B = sorted(glob.glob(os.path.join(root, "%s/b" % mode) + "/*.*"))
        
        # Filter out .raw files since they're companion files to .mhd
        self.files_A = [f for f in all_files_A if not f.endswith('.raw')]
        self.files_B = [f for f in all_files_B if not f.endswith('.raw')]

    def _load_image(self, filepath):
        """
        Load image from file. Supports .npy, .mha, and .mhd formats.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            numpy array of the image data
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.npy':
            # Load numpy array
            return np.load(filepath, allow_pickle=True)
        elif ext in ['.mha', '.mhd']:
            # Load MHA/MHD file using SimpleITK
            if not HAS_SIMPLEITK:
                raise ImportError(
                    f"SimpleITK is required to load {ext} files. "
                    "Install it with: pip install SimpleITK"
                )
            image = sitk.ReadImage(filepath)
            # Convert to numpy array and ensure correct axis order (D, H, W) for 3D
            array = sitk.GetArrayFromImage(image)
            return array.astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: .npy, .mha, .mhd")

    def __getitem__(self, index):

        image_A = self._load_image(self.files_A[index % len(self.files_A)])
        image_B = self._load_image(self.files_B[index % len(self.files_B)])

        item_A = torch.from_numpy(image_A)
        item_B = torch.from_numpy(image_B)
        
        # For 3D data, expect shape (D, H, W), add channel dimension to get (1, D, H, W)
        # For 2D data, expect shape (H, W), add channel dimension to get (1, H, W)
        if item_A.ndim == 2:
            # 2D case: (H, W) -> (1, H, W)
            item_A = torch.unsqueeze(item_A, 0)
            item_B = torch.unsqueeze(item_B, 0)
        elif item_A.ndim == 3:
            # 3D case: (D, H, W) -> (1, D, H, W)
            item_A = torch.unsqueeze(item_A, 0)
            item_B = torch.unsqueeze(item_B, 0)
        
        return {"a": item_A, "b": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))