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
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.float2tensor3(patch_L), util.float2tensor3(patch_H)

        else:
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
        # # 0 ~ 1
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        # load SM
        if isSM:
            sm = np.load(SM_path).astype(np.float32)[:, :, :, 1]

            # sm = np.reshape(sm[:, :, :, 1], (256, 256, 12))

            # 0 ~ 1
            sm = (sm - sm.min()) / (sm.max() - sm.min())

            return gt, sm
        else:
            return gt, 0

    def undersample_kspace(self, x, mask, is_noise, noise_level, noise_var):
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
            fft = fft + self.generate_gaussian_noise(fft, noise_level, noise_var)
        fft = ifftshift(fft)
        xx = ifft2(fft)
        xx = np.abs(xx)

        x = xx[:, :, np.newaxis]

        return x

    def generate_gaussian_noise(self, x, noise_level, noise_var):
        spower = np.sum(x**2) / x.size
        npower = noise_level / (1 - noise_level) * spower
        noise = np.random.normal(0, noise_var**0.5, x.shape) * np.sqrt(npower)
        return noise
