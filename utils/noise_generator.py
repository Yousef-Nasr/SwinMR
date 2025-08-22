"""
# -----------------------------------------
Comprehensive Noise Generator for MRI Data
by SwinMR Enhancement
# -----------------------------------------
"""

import numpy as np
import cv2
import json
import random
import os
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from typing import Dict, Tuple, Optional, List


class MRINoiseGenerator:
    """
    Comprehensive noise generator for MRI images to simulate real-world artifacts.
    Supports multiple noise types with configurable parameters.
    """
    
    def __init__(self, noise_config_path: Optional[str] = None):
        """
        Initialize the noise generator with configuration.
        
        Args:
            noise_config_path: Path to JSON configuration file
        """
        self.default_config = {
            "gaussian_noise": {
                "enabled": True,
                "weight": 1.0,
                "variance_range": [0.001, 0.1],
                "snr_range": [20, 50]
            },
            "rician_noise": {
                "enabled": True,
                "weight": 0.8,
                "sigma_range": [0.01, 0.05],
                "amplitude_range": [0.1, 0.3]
            },
            "ghosting": {
                "enabled": True,
                "weight": 0.6,
                "intensity_range": [0.1, 0.4],
                "offset_range": [5, 20],
                "direction": ["horizontal", "vertical", "both"]
            },
            "aliasing": {
                "enabled": True,
                "weight": 0.5,
                "fold_factor_range": [2, 4],
                "intensity_range": [0.2, 0.6]
            },
            "line_noise": {
                "enabled": True,
                "weight": 0.7,
                "coverage_max": 0.25,  # Maximum 25% coverage
                "intensity_range": [0.1, 0.5],
                "direction": ["horizontal", "vertical"],
                "line_width_range": [1, 3]
            },
            "zipper_artifact": {
                "enabled": True,
                "weight": 0.4,
                "intensity_range": [0.2, 0.8],
                "frequency_range": [0.1, 0.3],
                "direction": ["horizontal", "vertical"]
            }
        }
        
        if noise_config_path and os.path.exists(noise_config_path):
            self.config = self.load_config(noise_config_path)
        else:
            self.config = self.default_config
    
    def load_config(self, config_path: str) -> Dict:
        """Load noise configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Merge with default config to ensure all parameters exist
        merged_config = self.default_config.copy()
        for key, value in config.items():
            if key in merged_config:
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        
        return merged_config
    
    def save_config(self, config_path: str):
        """Save current configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def apply_noise(self, image: np.ndarray, noise_types: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply selected noise types to the input image.
        
        Args:
            image: Input MRI image (H, W) or (H, W, 1)
            noise_types: List of noise types to apply. If None, applies all enabled types.
            
        Returns:
            Noisy image with same shape as input
        """
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        
        noisy_image = image.copy().astype(np.float32)
        
        # Apply noise types based on their weights and enabled status
        if noise_types is None:
            noise_types = list(self.config.keys())
        
        for noise_type in noise_types:
            if noise_type in self.config and self.config[noise_type]['enabled']:
                weight = self.config[noise_type]['weight']
                if random.random() < weight:
                    noisy_image = self._apply_specific_noise(noisy_image, noise_type)
        
        return np.expand_dims(noisy_image, axis=2) if len(image.shape) == 3 else noisy_image
    
    def _apply_specific_noise(self, image: np.ndarray, noise_type: str) -> np.ndarray:
        """Apply a specific type of noise to the image."""
        noise_func = getattr(self, f"_apply_{noise_type}", None)
        if noise_func:
            return noise_func(image)
        else:
            print(f"Warning: Unknown noise type '{noise_type}'")
            return image
    
    def _apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise in k-space."""
        config = self.config['gaussian_noise']
        
        # Transform to k-space
        kspace = fftshift(fft2(image))
        
        # Generate noise
        variance = random.uniform(*config['variance_range'])
        noise_real = np.random.normal(0, np.sqrt(variance), kspace.shape)
        noise_imag = np.random.normal(0, np.sqrt(variance), kspace.shape)
        noise = noise_real + 1j * noise_imag
        
        # Add noise to k-space
        noisy_kspace = kspace + noise * np.std(kspace)
        
        # Transform back to image space
        noisy_image = np.abs(ifft2(ifftshift(noisy_kspace)))
        
        return noisy_image
    
    def _apply_rician_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply Rician noise (common in magnitude MRI images)."""
        config = self.config['rician_noise']
        
        sigma = random.uniform(*config['sigma_range'])
        amplitude = random.uniform(*config['amplitude_range'])
        
        # Add complex Gaussian noise
        noise_real = np.random.normal(0, sigma, image.shape)
        noise_imag = np.random.normal(0, sigma, image.shape)
        
        # Apply Rician distribution
        noisy_real = image + amplitude * noise_real
        noisy_imag = amplitude * noise_imag
        
        # Compute magnitude (Rician distributed)
        noisy_image = np.sqrt(noisy_real**2 + noisy_imag**2)
        
        return noisy_image
    
    def _apply_ghosting(self, image: np.ndarray) -> np.ndarray:
        """Apply ghosting artifacts."""
        config = self.config['ghosting']
        
        intensity = random.uniform(*config['intensity_range'])
        offset = random.randint(*config['offset_range'])
        direction = random.choice(config['direction'])
        
        ghosted_image = image.copy()
        
        if direction in ['horizontal', 'both']:
            # Horizontal ghosting
            ghost = np.roll(image, offset, axis=1)
            ghosted_image += intensity * ghost
        
        if direction in ['vertical', 'both']:
            # Vertical ghosting
            ghost = np.roll(image, offset, axis=0)
            ghosted_image += intensity * ghost
        
        return ghosted_image
    
    def _apply_aliasing(self, image: np.ndarray) -> np.ndarray:
        """Apply aliasing artifacts by undersampling in k-space."""
        config = self.config['aliasing']
        
        fold_factor = random.uniform(*config['fold_factor_range'])
        intensity = random.uniform(*config['intensity_range'])
        
        # Transform to k-space
        kspace = fftshift(fft2(image))
        
        # Create undersampling mask
        h, w = kspace.shape
        step = int(fold_factor)
        mask = np.zeros_like(kspace, dtype=bool)
        mask[::step, :] = True  # Keep every 'step' lines
        
        # Apply undersampling
        undersampled_kspace = kspace * mask
        
        # Transform back and combine with original
        aliased_image = np.abs(ifft2(ifftshift(undersampled_kspace)))
        combined_image = (1 - intensity) * image + intensity * aliased_image
        
        return combined_image
    
    def _apply_line_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply line noise artifacts (vertical or horizontal lines)."""
        config = self.config['line_noise']
        
        intensity = random.uniform(*config['intensity_range'])
        direction = random.choice(config['direction'])
        line_width = random.randint(*config['line_width_range'])
        
        h, w = image.shape
        noisy_image = image.copy()
        
        if direction == 'horizontal':
            # Horizontal line noise
            max_lines = int(h * config['coverage_max'])
            num_lines = random.randint(1, max_lines)
            
            for _ in range(num_lines):
                y_pos = random.randint(0, h - line_width)
                noise_value = random.uniform(-intensity, intensity) * np.mean(image)
                noisy_image[y_pos:y_pos + line_width, :] += noise_value
        
        elif direction == 'vertical':
            # Vertical line noise
            max_lines = int(w * config['coverage_max'])
            num_lines = random.randint(1, max_lines)
            
            for _ in range(num_lines):
                x_pos = random.randint(0, w - line_width)
                noise_value = random.uniform(-intensity, intensity) * np.mean(image)
                noisy_image[:, x_pos:x_pos + line_width] += noise_value
        
        return noisy_image
    
    def _apply_zipper_artifact(self, image: np.ndarray) -> np.ndarray:
        """Apply zipper artifacts (periodic noise in k-space)."""
        config = self.config['zipper_artifact']
        
        intensity = random.uniform(*config['intensity_range'])
        frequency = random.uniform(*config['frequency_range'])
        direction = random.choice(config['direction'])
        
        # Transform to k-space
        kspace = fftshift(fft2(image))
        h, w = kspace.shape
        
        # Create zipper pattern
        if direction == 'horizontal':
            # Horizontal zipper (affects vertical k-space)
            zipper_pos = int(h * frequency)
            zipper_mask = np.zeros(h, dtype=bool)
            zipper_mask[zipper_pos::int(1/frequency)] = True
            kspace[zipper_mask, :] *= (1 + intensity)
        
        elif direction == 'vertical':
            # Vertical zipper (affects horizontal k-space)
            zipper_pos = int(w * frequency)
            zipper_mask = np.zeros(w, dtype=bool)
            zipper_mask[zipper_pos::int(1/frequency)] = True
            kspace[:, zipper_mask] *= (1 + intensity)
        
        # Transform back to image space
        zipper_image = np.abs(ifft2(ifftshift(kspace)))
        
        return zipper_image


def create_default_noise_config(output_path: str):
    """Create a default noise configuration file."""
    generator = MRINoiseGenerator()
    generator.save_config(output_path)
    print(f"Default noise configuration saved to: {output_path}")


if __name__ == "__main__":
    # Create default configuration
    create_default_noise_config("/tmp/default_noise_config.json")
    
    # Example usage
    generator = MRINoiseGenerator()
    
    # Create test image
    test_image = np.random.rand(256, 256)
    
    # Apply noise
    noisy_image = generator.apply_noise(test_image)
    
    print("Noise generator test completed successfully")