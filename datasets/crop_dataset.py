"""
Unified CropDataset class combining SliceDataset, CropDataset, and AugCropDataset.
This single class handles all functionality with MHA file support.
"""
import json
import os
from multiprocessing import Manager

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from datasets.masked import block_mask, no_mask, random_mask, shift_mask, square_mask
from datasets.readers import READERS, read_files
from datasets.utils import (
    crop_and_pad,
    get_size_with_view,
    rescale01,
    rescale_back,
    sample_coords,
)


class CropDataset(data.Dataset):
    """
    Unified dataset class with sliding window sampling and augmentation support.
    Combines functionality from SliceDataset, CropDataset, and AugCropDataset.
    
    Features:
    - MHA/MHD file loading support via SimpleITK
    - Sliding window patch extraction for inference
    - Random patch sampling with augmentation for training
    - Multi-view support (axial, sagittal, coronal)
    - Patch assembly and reconstruction
    
    Args:
        config: Config object with dataset parameters
    """

    def __init__(self, config):
        self.config = config
        self.view = config.view if hasattr(config, 'view') else -1
        self.dataset_reader_dict = READERS.get(config.source, READERS['default']) if hasattr(config, 'source') else READERS['default']
        
        # Load case list from JSON
        json_path = os.path.join("datalists", config.source if hasattr(config, 'source') else 'default', config.phase + ".json")
        if not os.path.exists(json_path):
            # Create a minimal case list if file doesn't exist
            self.case_lists = []
            print(f"Warning: {json_path} not found. Using empty case list.")
        else:
            with open(json_path, "r") as fp:
                self.case_lists = json.load(fp)
                self.case_lists = sorted(self.case_lists, key=lambda x: x["name"])
        
        if hasattr(config, 'debug') and config.debug:
            self.case_lists = self.case_lists[:2]
        
        # Adjust validation set patches
        if config.phase == 'val' and hasattr(config, "patch_size"):
            self._adjust_validation_patches()
        
        # Initialize data structures
        self.names = []
        self.valid_starts = []
        self.valid_ends = []
        self.valid_lengths = []
        self.min_vals = []
        self.max_vals = []
        
        self._accumulate_indices()
        
        # Shared memory for caching
        manager = Manager()
        self.tensors = manager.dict()
        self.params = manager.dict()

    def _adjust_validation_patches(self):
        """Adjust validation set to focus on central patches."""
        target_size = self.config.patch_size if hasattr(self.config, 'patch_size') else [4, 128, 128]
        
        for idx in range(len(self.case_lists)):
            case = self.case_lists[idx]
            valid_starts = case["valid_starts"]
            valid_ends = case["valid_ends"]
            valid_lengths = case["valid_lengths"]
            
            for dim, size in enumerate(target_size):
                center = valid_starts[dim] + valid_lengths[dim] // 2
                valid_start = max(center - size // 2, valid_starts[dim])
                valid_end = min(center + size // 2, valid_ends[dim])
                valid_length = size
                
                self.case_lists[idx]["valid_starts"][dim] = valid_start
                self.case_lists[idx]["valid_ends"][dim] = valid_end
                self.case_lists[idx]["valid_lengths"][dim] = valid_length

    def _count_slides(self, full_size, view):
        """Count number of sliding window positions."""
        full_size = np.array(full_size, dtype=np.int32)
        patch_size = get_size_with_view(self.config.patch_size, view)
        stride_size = get_size_with_view(self.config.stride_size, view) if hasattr(self.config, 'stride_size') else patch_size
        
        if self.config.phase == "train":
            # For training, use full size as we do random sampling
            slide_count = full_size[view] if self.view >= 0 else np.prod(full_size)
        else:
            # For validation/test, calculate actual sliding window positions
            slide_nums = (
                np.ceil(np.maximum((full_size - patch_size) / stride_size, 0)) + 1
            )
            slide_count = int(np.prod(slide_nums))
        
        self.slide_counts.append(slide_count)

    def _accumulate_indices(self):
        """Accumulate indices for all cases."""
        self.slide_counts = []
        
        for case in self.case_lists:
            self.names.append(case["name"])
            self.min_vals.append(case.get("min_val", 0.0))
            self.max_vals.append(case.get("max_val", 1.0))
            self.valid_starts.extend(case["valid_starts"])
            self.valid_ends.extend(case["valid_ends"])
            self.valid_lengths.extend(case["valid_lengths"])
            
            full_size = np.array(case["valid_lengths"], dtype=np.int32)
            
            if hasattr(self.config, 'patch_size'):
                # Patch-based dataset
                if self.view >= 0:
                    self._count_slides(full_size, self.view)
                else:
                    for _view in range(3):
                        self._count_slides(full_size, _view)
            else:
                # Slice-based dataset
                if self.view >= 0:
                    self.slide_counts.append(case["valid_lengths"][self.view])
                else:
                    self.slide_counts.extend(case["valid_lengths"])
        
        self.cum_lens = np.cumsum(self.slide_counts)
        self.case_idxs = np.arange(len(self.cum_lens))

    def __len__(self):
        """Return total number of samples."""
        return self.cum_lens[-1] if len(self.cum_lens) > 0 else 0

    def _cal_offsets(self, full_size, slide_index, view):
        """Calculate patch offsets for sliding window."""
        config = self.config
        full_size = np.array(full_size, dtype=np.int32)
        patch_size = get_size_with_view(config.patch_size, view)
        stride_size = get_size_with_view(config.stride_size if hasattr(config, 'stride_size') else config.patch_size, view)
        slide_nums = np.ceil(np.maximum((full_size - patch_size) / stride_size, 0)) + 1
        
        # Calculate 3D position from linear index
        z = slide_index // (slide_nums[1] * slide_nums[2])
        rem = slide_index % (slide_nums[1] * slide_nums[2])
        y = rem // slide_nums[2]
        x = rem % slide_nums[2]
        
        # Calculate start positions
        starts = (
            np.maximum(full_size - patch_size, 0)
            / np.maximum(slide_nums - 1, 1)
            * np.array([z, y, x])
        ).astype(np.int32)
        
        ends = np.minimum(starts + patch_size, full_size)
        pad_size = np.maximum(patch_size - (ends - starts), 0)
        pad_start = pad_size // 2
        pad_end = pad_size - pad_start
        
        return starts, ends, pad_size, pad_start, pad_end

    def _calculate_indices(self, index):
        """Calculate view, case index, and slide index from global index."""
        plane_index = self.case_idxs[index < self.cum_lens].min()
        
        if plane_index == 0:
            slide_index = index
        else:
            slide_index = index - self.cum_lens[plane_index - 1]
        
        if self.view >= 0:
            view = self.view
            case_index = plane_index
        else:
            view = plane_index % 3
            case_index = plane_index // 3
        
        return view, case_index, slide_index

    def _permute_vols_fore(self, view, *vols):
        """Permute volumes to align with view."""
        permute_indices = {0: [0, 1, 2], 1: [1, 0, 2], 2: [2, 0, 1]}
        vols = [vol.permute(*permute_indices[view]) for vol in vols]
        return vols

    def _permute_vols_back(self, view, *vols):
        """Permute volumes back from view alignment."""
        permute_indices = {0: [0, 1, 2], 1: [1, 0, 2], 2: [1, 2, 0]}
        vols = [vol.permute(*permute_indices[view]) for vol in vols]
        return vols

    def _pre_process(self, tensors):
        """Pre-process tensors (normalization)."""
        config = self.config
        if hasattr(config, 'data_normalization_factors'):
            data_normalization_factors = config.data_normalization_factors
            for i, input_type in enumerate(self.dataset_reader_dict.keys()):
                if input_type in data_normalization_factors.keys():
                    factors = data_normalization_factors[input_type]
                    assert 'min' in factors and 'max' in factors
                    tensors[i] = rescale01(tensors[i], factors['min'], factors['max'])
        return tensors

    def get_tensors(self, case_index):
        """Load and cache tensors for a case."""
        config = self.config
        if case_index not in self.tensors:
            folder = os.path.join(config.data_root, self.names[case_index])
            tensors, params = read_files(folder, self.dataset_reader_dict, ext='mha')
            tensors = self._pre_process(tensors)
            
            # Crop to valid region
            starts = self.valid_starts[3 * case_index : 3 * case_index + 3]
            ends = self.valid_ends[3 * case_index : 3 * case_index + 3]
            tensors = [
                tensor[
                    starts[0] : ends[0] + 1,
                    starts[1] : ends[1] + 1,
                    starts[2] : ends[2] + 1,
                ]
                for tensor in tensors
            ]
            
            self.tensors[case_index] = tensors
            self.params[case_index] = params
        
        return self.names[case_index], self.tensors[case_index]

    def _masked_aug(self, res_dct):
        """Apply masked augmentation."""
        config = self.config
        
        if not hasattr(config, 'masked_type'):
            return res_dct
        
        aug_func = {
            "none": no_mask,
            "square": square_mask,
            "block": block_mask,
            "shift": shift_mask,
            "random": random_mask,
        }[config.masked_type]
        
        aug_params = {
            "none": [],
            "square": [config.masked_block_size] if hasattr(config, 'masked_block_size') else [16],
            "block": [
                config.masked_block_size if hasattr(config, 'masked_block_size') else 16,
                config.masked_ratio if hasattr(config, 'masked_ratio') else 0.1
            ],
            "shift": [
                config.masked_block_size if hasattr(config, 'masked_block_size') else 16,
                config.masked_ratio if hasattr(config, 'masked_ratio') else 0.1
            ],
            "random": [
                config.masked_block_size if hasattr(config, 'masked_block_size') else 16,
                config.masked_ratio if hasattr(config, 'masked_ratio') else 0.1
            ],
        }[config.masked_type]
        
        shape = res_dct[list(self.dataset_reader_dict.keys())[0]].shape[-2:]
        aug_mask = aug_func(shape, *aug_params)
        
        for key in self.dataset_reader_dict.keys():
            res_dct[key] = res_dct[key] * aug_mask
        
        return res_dct

    def __getitem__(self, index):
        """Get a single sample."""
        config = self.config
        view, case_index, slide_index = self._calculate_indices(index)
        _, tensors = self.get_tensors(case_index)
        
        # Training: random augmented patches
        if config.phase == "train" and hasattr(config, 'patch_size'):
            coords, _ = sample_coords(config, view, tensors[0].shape)
            
            # Sample patches using grid_sample
            tensors = [
                F.grid_sample(
                    tensor[None, None].float(), coords, 'bilinear', align_corners=False
                ).squeeze(0).squeeze(0)
                for tensor in tensors
            ]
            
            # Permute for view
            tensors = self._permute_vols_fore(view, *tensors)
            
            res = {name: tensor * 2 - 1 for name, tensor in zip(self.dataset_reader_dict.keys(), tensors)}
            res["idx"] = case_index
            res = self._masked_aug(res)
            
            return res
        
        # Validation/Test: sliding window patches
        elif hasattr(config, 'patch_size'):
            starts, ends, pad_size, pad_s, pad_e = self._cal_offsets(
                tensors[0].shape, slide_index, view
            )
            
            # Calculate offset for positional encoding
            mid = starts + (ends - starts) // 2
            full_size = np.array(tensors[0].shape, dtype=np.int32)
            offset = 2 * mid / full_size - 1
            offset = torch.from_numpy(offset)
            offset = torch.flip(offset, dims=(0,))
            
            # Crop patches
            tensors = [
                tensor[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
                for tensor in tensors
            ]
            
            # Pad if needed
            if (pad_size > 0).any():
                pads = [pad_s[2], pad_e[2], pad_s[1], pad_e[1], pad_s[0], pad_e[0]]
                tensors = [F.pad(tensor, pads) for tensor in tensors]
            
            # Permute for view
            tensors = self._permute_vols_fore(view, *tensors)
            
            res = {name: tensor * 2 - 1 for name, tensor in zip(self.dataset_reader_dict.keys(), tensors)}
            res["idx"] = case_index
            res["clinic"] = case_index // (len(self.names) + 1 // 3)
            res["offset"] = offset
            
            return res
        
        # Slice-based (no patch_size)
        else:
            res = {
                name: torch.select(tensor, view, slide_index)[None]
                for name, tensor in zip(self.dataset_reader_dict.keys(), tensors)
            }
            if 'mask' in res.keys():
                res["mask"] = res["mask"].bool()
            res["idx"] = case_index
            return res

    def _assemble_view(self, view, case_index, preds, result, count):
        """Assemble predictions for a specific view."""
        config = self.config
        tensors = self.tensors[case_index]
        
        for slide_index, pred in enumerate(preds):
            starts, ends, pad_size, pad_s, pad_e = self._cal_offsets(
                tensors[0].shape, slide_index, view
            )
            
            # Remove padding
            patch_size = np.array(config.patch_size)
            crop_s, crop_e = pad_s, patch_size - pad_e
            if (pad_size > 0).any():
                pred = pred[
                    crop_s[0]:crop_e[0], crop_s[1]:crop_e[1], crop_s[2]:crop_e[2]
                ]
            
            # Add to result
            result[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] += \
                self._permute_vols_back(view, pred.squeeze())[0]
            count[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] += 1

    def assemble(self, case_index, pred):
        """Assemble predictions from patches into full volume."""
        if not hasattr(self.config, 'patch_size'):
            # Slice-based assembly
            if self.view >= 0:
                start = self.valid_starts[case_index]
                end = self.valid_ends[case_index]
            else:
                starts = self.valid_starts[case_index * 3 : (case_index + 1) * 3]
                ends = self.valid_ends[case_index * 3 : (case_index + 1) * 3]
                lengths = self.valid_lengths[case_index * 3 : (case_index + 1) * 3]
            
            result = torch.zeros(*self.config.crop_size).to(pred.device)
            
            if self.view == 0:
                result[start:end + 1, :, :] = pred
            elif self.view == 1:
                result[:, start:end + 1, :] = pred.permute(1, 0, 2)
            elif self.view == 2:
                result[:, :, start:end + 1] = pred.permute(1, 2, 0)
            else:
                preds = torch.split(pred, lengths, dim=0)
                count = torch.zeros(*self.config.crop_size).to(pred.device)
                
                result[starts[0]:ends[0] + 1, :, :] += preds[0]
                count[starts[0]:ends[0] + 1, :, :] += 1
                
                result[:, starts[1]:ends[1] + 1, :] += preds[1].permute(1, 0, 2)
                count[:, starts[1]:ends[1] + 1, :] += 1
                
                result[:, :, starts[2]:ends[2] + 1] += preds[2].permute(1, 2, 0)
                count[:, :, starts[2]:ends[2] + 1] += 1
                
                result = torch.where(count > 0, result / count, result)
            
            return result
        
        # Patch-based assembly
        tensors = self.tensors[case_index]
        result = torch.zeros(tensors[0].shape).to(pred.device)
        count = torch.zeros(tensors[0].shape).to(pred.device)
        
        if self.view >= 0:
            self._assemble_view(self.view, case_index, pred, result, count)
        else:
            slide_counts = self.slide_counts[case_index * 3 : (case_index + 1) * 3]
            preds = torch.split(pred, slide_counts, dim=0)
            for _view in range(3):
                self._assemble_view(_view, case_index, preds[_view], result, count)
        
        result = torch.where(count > 0, result / count, result)
        return result

    def reconstruct(self, case_index, pred, data_normalization_factors):
        """Reconstruct full resolution output."""
        params = self.params[case_index]
        shape = np.array(params["size"])[::-1]
        full = torch.zeros(*shape, dtype=torch.float32, device=pred.device)
        
        starts = self.valid_starts[3 * case_index : 3 * (case_index + 1)]
        ends = self.valid_ends[3 * case_index : 3 * (case_index + 1)]
        full[starts[0]:ends[0] + 1, starts[1]:ends[1] + 1, starts[2]:ends[2] + 1] = pred
        
        # Post-process
        if hasattr(self.config, 'crop_size'):
            full = crop_and_pad(full, self.config.crop_size)
        
        full = rescale_back(full, data_normalization_factors['min'], data_normalization_factors['max'])
        return full
