"""
# -----------------------------------------
Data Loader
CC-SAG-NPI d.1.1
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
"""

import random
import os
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_swinmr import *
from models.select_mask import define_Mask
from utils.noise_generator import MRINoiseGenerator
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import numpy as np
import torch
import warnings


def safe_collate_fn(batch):
    """
    Safe collate function that handles potential type casting issues.
    """
    try:
        # Filter out None values
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
            
        # Ensure all tensors have consistent types
        for item in batch:
            if 'L' in item and item['L'] is not None:
                item['L'] = item['L'].float()
            if 'H' in item and item['H'] is not None:
                item['H'] = item['H'].float()
                
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Return a minimal valid batch to prevent crash
        dummy_item = {
            'L': torch.zeros(1, 256, 256),
            'H': torch.zeros(1, 256, 256),
            'H_path': 'dummy_path',
            'mask': np.ones((256, 256)),
            'SM': 0,
            'img_info': 'dummy'
        }
        return torch.utils.data.dataloader.default_collate([dummy_item])


def validate_data_files(data_paths, max_check=5):
    """
    Validate a sample of data files to check for common issues.
    """
    issues_found = []
    check_count = min(max_check, len(data_paths))
    
    for i, path in enumerate(data_paths[:check_count]):
        try:
            data = np.load(path)
            
            # Check for NaN/Inf
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                issues_found.append(f"NaN/Inf in {path}")
                
            # Check for extreme values
            if data.max() > 1e6 or data.min() < -1e6:
                issues_found.append(f"Extreme values in {path}: min={data.min()}, max={data.max()}")
                
            # Check for constant data
            if data.max() == data.min():
                issues_found.append(f"Constant values in {path}")
                
        except Exception as e:
            issues_found.append(f"Cannot load {path}: {e}")
    
    if issues_found:
        print("Data validation issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print(f"Validation passed for {check_count} files")
    
    return issues_found


class DatasetCCsagnpi(data.Dataset):
    def __init__(self, opt):
        super(DatasetCCsagnpi, self).__init__()
        print(
            'Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.'
        )
        self.opt = opt
        self.n_channels = self.opt["n_channels"]
        self.patch_size = self.opt["H_size"]
        self.is_noise = self.opt["is_noise"]
        self.noise_level = self.opt["noise_level"]
        self.noise_var = self.opt["noise_var"]
        self.is_mini_dataset = self.opt["is_mini_dataset"]
        self.mini_dataset_prec = self.opt["mini_dataset_prec"]

        # Enhanced noise options
        self.use_enhanced_noise = self.opt.get("use_enhanced_noise", False)
        self.noise_config_path = self.opt.get("noise_config_path", None)
        self.random_mask_selection = self.opt.get("random_mask_selection", False)
        self.available_masks = self.opt.get(
            "available_masks", ["G1D30"]
        )  # Default mask

        # Noise type limitation (1-2 types per image)
        self.max_noise_types = self.opt.get(
            "max_noise_types", 2
        )  # Maximum 1-2 noise types per image
        self.min_noise_types = self.opt.get(
            "min_noise_types", 1
        )  # Minimum 1 noise type per image
        self.include_mask_as_noise = self.opt.get(
            "include_mask_as_noise", True
        )  # Count mask as a noise type

        # Initialize noise generator if enhanced noise is enabled
        if self.use_enhanced_noise:
            self.noise_generator = MRINoiseGenerator(self.noise_config_path)
            print(
                f"Enhanced noise generator initialized with config: {self.noise_config_path}"
            )

        # get data path of image & sensitivity map
        self.paths_raw = util.get_image_paths(opt["dataroot_H"])
        assert self.paths_raw, "Error: Raw path is empty."

        self.paths_H = []
        self.paths_SM = []
        for path in self.paths_raw:
            if "imgGT" in path:
                self.paths_H.append(path)
            elif "SensitivityMaps" in path:
                self.paths_SM.append(path)
            else:
                raise ValueError("Error: Unknown filename is in raw path")

        if self.is_mini_dataset:
            pass

        # get mask (will be randomized if random_mask_selection is True)
        if not self.random_mask_selection:
            self.mask = define_Mask(self.opt)
        else:
            self.mask = None  # Will be selected randomly per sample

    def __getitem__(self, index):
        # Select mask (random or fixed)
        if self.random_mask_selection:
            # Randomly select a mask type
            selected_mask_name = random.choice(self.available_masks)
            temp_opt = self.opt.copy()
            temp_opt["mask"] = selected_mask_name
            mask = define_Mask(temp_opt)
        else:
            mask = self.mask

        is_noise = self.is_noise
        noise_level = self.noise_level
        noise_var = self.noise_var

        # get gt image
        H_path = self.paths_H[index]
        img_H, _ = self.load_images(H_path, 0, isSM=False)

        # Apply enhanced noise with limitation to 1-2 types
        if self.use_enhanced_noise:
            # Get all available noise types
            available_noise_types = [
                "gaussian_noise",
                "rician_noise",
                "ghosting",
                "aliasing",
                "line_noise",
                "zipper_artifact",
            ]

            # Determine how many noise types to apply (1 or 2)
            num_noise_types = random.randint(self.min_noise_types, self.max_noise_types)

            # If mask is considered as noise type, we might reduce other noise types
            if self.include_mask_as_noise and self.random_mask_selection:
                # Mask selection counts as one noise type
                remaining_noise_slots = max(0, num_noise_types - 1)
                if remaining_noise_slots == 0:
                    # Only apply mask, no other noise
                    img_for_undersampling = img_H
                else:
                    # Apply 1 additional noise type + mask
                    selected_noise_types = random.sample(
                        available_noise_types, remaining_noise_slots
                    )
                    img_H_noisy = self.noise_generator.apply_noise(
                        img_H.squeeze(), selected_noise_types
                    )
                    img_H_noisy = np.expand_dims(img_H_noisy, axis=2)
                    img_for_undersampling = img_H_noisy
            else:
                # Apply 1-2 noise types without considering mask as noise
                selected_noise_types = random.sample(
                    available_noise_types, num_noise_types
                )
                img_H_noisy = self.noise_generator.apply_noise(
                    img_H.squeeze(), selected_noise_types
                )
                img_H_noisy = np.expand_dims(img_H_noisy, axis=2)
                img_for_undersampling = img_H_noisy

            # Debug info (can be removed in production)
            if hasattr(self, "debug_mode") and self.debug_mode:
                if self.include_mask_as_noise and self.random_mask_selection:
                    noise_info = f"Mask: {selected_mask_name}, Additional: {selected_noise_types if remaining_noise_slots > 0 else 'None'}"
                else:
                    noise_info = f"Noise types: {selected_noise_types}"
                print(f"Sample {index}: {noise_info}")
        else:
            img_for_undersampling = img_H

        # get zf image (undersampled)
        img_L = self.undersample_kspace(
            img_for_undersampling, mask, is_noise, noise_level, noise_var
        )

        # get image information
        image_name_ext = os.path.basename(H_path)
        img_name, ext = os.path.splitext(image_name_ext)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt["phase"] == "train":
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[
                rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :
            ]
            patch_H = img_H[
                rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :
            ]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------

            mode = random.randint(0, 7)
            patch_L, patch_H = (
                util.augment_img(patch_L, mode=mode),
                util.augment_img(patch_H, mode=mode),
            )
            
            # Ensure consistent data types before tensor conversion
            patch_L = patch_L.astype(np.float32)
            patch_H = patch_H.astype(np.float32)
            
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.float2tensor3(patch_L), util.float2tensor3(patch_H)

        else:
            # Ensure consistent data types before tensor conversion
            img_L = img_L.astype(np.float32)
            img_H = img_H.astype(np.float32)
            
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.float2tensor3(img_L), util.float2tensor3(img_H)

        return {
            "L": img_L,
            "H": img_H,
            "H_path": H_path,
            "mask": mask,
            "SM": _,
            "img_info": img_name,
        }

    def __len__(self):
        return len(self.paths_H)

    def load_images(self, H_path, SM_path, isSM=True):
        # load GT
        gt = np.load(H_path).astype(np.float32)

        gt = np.reshape(gt, (gt.shape[0], gt.shape[1], 1))
        
        # Check for NaN or Inf values and handle them
        if np.any(np.isnan(gt)) or np.any(np.isinf(gt)):
            print(f"Warning: NaN or Inf found in {H_path}, replacing with zeros")
            gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to 0 ~ 1 with safe division
        gt_min, gt_max = gt.min(), gt.max()
        if gt_max > gt_min:
            gt = (gt - gt_min) / (gt_max - gt_min)
        else:
            # Handle case where all values are the same
            print(f"Warning: Constant values in {H_path}, setting to zeros")
            gt = np.zeros_like(gt)

        # load SM
        if isSM:
            sm = np.load(SM_path).astype(np.float32)[:, :, :, 1]

            # sm = np.reshape(sm[:, :, :, 1], (256, 256, 12))

            # Check for NaN or Inf values in SM
            if np.any(np.isnan(sm)) or np.any(np.isinf(sm)):
                print(f"Warning: NaN or Inf found in SM {SM_path}, replacing with zeros")
                sm = np.nan_to_num(sm, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize to 0 ~ 1 with safe division
            sm_min, sm_max = sm.min(), sm.max()
            if sm_max > sm_min:
                sm = (sm - sm_min) / (sm_max - sm_min)
            else:
                # Handle case where all values are the same
                print(f"Warning: Constant values in SM {SM_path}, setting to zeros")
                sm = np.zeros_like(sm)

            return gt, sm
        else:
            return gt, 0

    def undersample_kspace(self, x, mask, is_noise, noise_level, noise_var):
        # Ensure input is valid
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("Warning: NaN or Inf found in input to undersample_kspace, replacing with zeros")
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
        fft = fft2(x[:, :, 0])
        fft = fftshift(fft)

        # Resize mask to match FFT dimensions if needed
        if mask.shape != fft.shape:
            from scipy.ndimage import zoom

            zoom_factors = (fft.shape[0] / mask.shape[0], fft.shape[1] / mask.shape[1])
            mask = zoom(
                mask, zoom_factors, order=0
            )  # Use nearest neighbor interpolation for binary mask

        fft = fft * mask
        if is_noise:
            noise = self.generate_gaussian_noise(fft, noise_level, noise_var)
            # Ensure noise doesn't contain NaN or Inf
            if np.any(np.isnan(noise)) or np.any(np.isinf(noise)):
                print("Warning: NaN or Inf found in generated noise, replacing with zeros")
                noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
            fft = fft + noise
            
        fft = ifftshift(fft)
        xx = ifft2(fft)
        xx = np.abs(xx)
        
        # Final check for NaN or Inf in result
        if np.any(np.isnan(xx)) or np.any(np.isinf(xx)):
            print("Warning: NaN or Inf found in undersample result, replacing with zeros")
            xx = np.nan_to_num(xx, nan=0.0, posinf=0.0, neginf=0.0)

        x = xx[:, :, np.newaxis].astype(np.float32)

        return x

    def generate_gaussian_noise(self, x, noise_level, noise_var):
        # Ensure inputs are valid
        if noise_level <= 0 or noise_level >= 1:
            print(f"Warning: Invalid noise_level {noise_level}, setting to 0.1")
            noise_level = 0.1
            
        if noise_var <= 0:
            print(f"Warning: Invalid noise_var {noise_var}, setting to 0.01")
            noise_var = 0.01
            
        spower = np.sum(np.abs(x)**2) / x.size
        
        # Avoid division by zero
        if spower == 0:
            print("Warning: Signal power is zero, using minimal noise")
            noise = np.random.normal(0, noise_var**0.5, x.shape) * 1e-6
        else:
            npower = noise_level / (1 - noise_level) * spower
            if npower <= 0:
                print("Warning: Calculated noise power is non-positive, using minimal noise")
                noise = np.random.normal(0, noise_var**0.5, x.shape) * 1e-6
            else:
                noise = np.random.normal(0, noise_var**0.5, x.shape) * np.sqrt(npower)
        
        # Ensure noise doesn't contain NaN or Inf
        if np.any(np.isnan(noise)) or np.any(np.isinf(noise)):
            print("Warning: Generated noise contains NaN or Inf, replacing with zeros")
            noise = np.zeros_like(x)
            
        return noise
