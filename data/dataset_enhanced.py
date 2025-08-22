"""
# -----------------------------------------
Enhanced Dataset Loader with Noise Support
DICOM/JPG Support without Sensitivity Maps
by SwinMR Enhancement
# -----------------------------------------
"""

import random
import os
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_swinmr import *
from models.select_mask import define_Mask
from utils.noise_generator import MRINoiseGenerator
from pathlib import Path
from typing import Optional, Dict


class DatasetEnhanced(data.Dataset):
    """
    Enhanced dataset class that supports:
    - DICOM and JPG/PNG images
    - Comprehensive noise generation
    - No sensitivity maps required
    - Configurable noise parameters
    """
    
    def __init__(self, opt):
        super(DatasetEnhanced, self).__init__()
        print('Enhanced Dataset: Get L/H for image-to-image mapping.')
        self.opt = opt
        self.n_channels = self.opt['n_channels']
        self.patch_size = self.opt['H_size']
        
        # Legacy noise parameters (for compatibility)
        self.is_noise = self.opt.get('is_noise', False)
        self.noise_level = self.opt.get('noise_level', 0.0)
        self.noise_var = self.opt.get('noise_var', 0.1)
        
        # Enhanced noise parameters
        self.use_enhanced_noise = self.opt.get('use_enhanced_noise', True)
        self.noise_config_path = self.opt.get('noise_config_path', None)
        self.noise_types = self.opt.get('noise_types', None)  # Specific noise types to apply
        
        # Mini dataset for debugging
        self.is_mini_dataset = self.opt.get('is_mini_dataset', False)
        self.mini_dataset_prec = self.opt.get('mini_dataset_prec', 1)
        
        # Initialize noise generator
        if self.use_enhanced_noise:
            self.noise_generator = MRINoiseGenerator(self.noise_config_path)
        else:
            self.noise_generator = None
        
        # Get data paths
        self.paths_H = self._get_image_paths(opt['dataroot_H'])
        assert self.paths_H, 'Error: No valid images found in dataroot_H'
        
        # Apply mini dataset if specified
        if self.is_mini_dataset:
            num_samples = max(1, int(len(self.paths_H) * self.mini_dataset_prec))
            self.paths_H = self.paths_H[:num_samples]
            print(f"Using mini dataset with {num_samples} samples")
        
        # Get mask
        self.mask = define_Mask(self.opt)
        
        print(f"Dataset initialized with {len(self.paths_H)} images")
        if self.use_enhanced_noise:
            print("Enhanced noise generation enabled")
    
    def _get_image_paths(self, dataroot: str):
        """Get all valid image paths from dataroot."""
        dataroot = Path(dataroot)
        
        if not dataroot.exists():
            raise ValueError(f"Data root directory does not exist: {dataroot}")
        
        # Supported extensions
        extensions = {'.npy', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Try to import pydicom for DICOM support
        try:
            import pydicom
            extensions.update({'.dcm', '.dicom'})
        except ImportError:
            pass
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            # Search recursively
            pattern = f"**/*{ext}"
            files = list(dataroot.glob(pattern))
            image_paths.extend(files)
            
            # Also search uppercase
            pattern = f"**/*{ext.upper()}"
            files = list(dataroot.glob(pattern))
            image_paths.extend(files)
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        # Filter for valid files
        valid_paths = []
        for path in image_paths:
            try:
                # Quick validation by trying to load
                if self._load_image(path) is not None:
                    valid_paths.append(str(path))
            except Exception:
                continue
        
        return valid_paths
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from various formats."""
        path = Path(image_path)
        ext = path.suffix.lower()
        
        try:
            if ext == '.npy':
                # Load numpy array
                image = np.load(image_path).astype(np.float32)
            
            elif ext in {'.dcm', '.dicom'}:
                # Load DICOM
                try:
                    import pydicom
                    ds = pydicom.dcmread(image_path)
                    if hasattr(ds, 'pixel_array'):
                        image = ds.pixel_array.astype(np.float32)
                        
                        # Handle multi-dimensional data
                        if len(image.shape) == 3:
                            if image.shape[2] == 3:
                                # RGB to grayscale
                                import cv2
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                            else:
                                # Take middle slice
                                image = image[image.shape[0] // 2]
                    else:
                        return None
                except ImportError:
                    print(f"Warning: pydicom not available, skipping DICOM file: {image_path}")
                    return None
            
            else:
                # Load standard image formats
                import cv2
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    return None
                image = image.astype(np.float32)
            
            # Ensure 2D
            if image.ndim > 2:
                image = image.squeeze()
            
            # Normalize to [0, 1]
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min())
            
            # Add channel dimension
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def __getitem__(self, index):
        mask = self.mask
        
        # Load ground truth image
        H_path = self.paths_H[index]
        img_H = self._load_image(H_path)
        
        if img_H is None:
            # Fallback to next image if current fails
            index = (index + 1) % len(self.paths_H)
            H_path = self.paths_H[index]
            img_H = self._load_image(H_path)
        
        # Get image information
        image_name_ext = os.path.basename(H_path)
        img_name, ext = os.path.splitext(image_name_ext)
        
        # Create low-quality image with undersampling and noise
        img_L = self._undersample_kspace(img_H, mask)
        
        # Apply additional noise if enabled
        if self.use_enhanced_noise and self.noise_generator is not None:
            img_L = self.noise_generator.apply_noise(img_L, self.noise_types)
        
        # Training mode: extract patches
        if self.opt['phase'] == 'train':
            H, W, _ = img_H.shape
            
            # Randomly crop patches
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            
            # Data augmentation
            mode = random.randint(0, 7)
            patch_L = util.augment_img(patch_L, mode=mode)
            patch_H = util.augment_img(patch_H, mode=mode)
            
            # Convert to tensors
            img_L = util.float2tensor3(patch_L)
            img_H = util.float2tensor3(patch_H)
        
        else:
            # Test mode: use full images
            img_L = util.float2tensor3(img_L)
            img_H = util.float2tensor3(img_H)
        
        return {
            'L': img_L,
            'H': img_H,
            'H_path': H_path,
            'mask': mask,
            'SM': None,  # No sensitivity maps
            'img_info': img_name
        }
    
    def __len__(self):
        return len(self.paths_H)
    
    def _undersample_kspace(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply k-space undersampling with optional legacy noise."""
        # Transform to k-space
        fft = fft2(x[:, :, 0])
        fft = fftshift(fft)
        
        # Apply undersampling mask
        fft = fft * mask
        
        # Add legacy Gaussian noise if enabled
        if self.is_noise and not self.use_enhanced_noise:
            fft = fft + self._generate_gaussian_noise(fft, self.noise_level, self.noise_var)
        
        # Transform back to image space
        fft = ifftshift(fft)
        xx = ifft2(fft)
        xx = np.abs(xx)
        
        return xx[:, :, np.newaxis]
    
    def _generate_gaussian_noise(self, x: np.ndarray, noise_level: float, noise_var: float) -> np.ndarray:
        """Generate Gaussian noise (legacy method for compatibility)."""
        spower = np.sum(x ** 2) / x.size
        npower = noise_level / (1 - noise_level) * spower
        noise = np.random.normal(0, noise_var ** 0.5, x.shape) * np.sqrt(npower)
        return noise


class DatasetEnhancedNPI(DatasetEnhanced):
    """
    Enhanced dataset class without parallel imaging (no sensitivity maps).
    Alias for better compatibility with existing code.
    """
    pass


class DatasetDICOMJPG(DatasetEnhanced):
    """
    Dataset class specifically for DICOM and JPG images.
    Alias for better naming clarity.
    """
    pass