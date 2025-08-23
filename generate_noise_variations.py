"""
# -----------------------------------------
Comprehensive Noise Testing Tool for MRI Images
This tool takes an input image and generates all noise variations
to demonstrate the enhanced noise generator capabilities
# -----------------------------------------
"""

import os
import cv2
import numpy as np
import argparse
from utils.noise_generator import MRINoiseGenerator
from models.select_mask import define_Mask
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


class NoiseVariationGenerator:
    """
    Generate comprehensive noise variations for MRI images
    to mimic real training conditions and test noise robustness.
    """

    def __init__(self, output_dir="noise_variations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize noise generator with different configurations
        self.noise_configs = {
            "mild": "configs/noise_config_mild.json",
            "default": "configs/noise_config_default.json",
            "aggressive": "configs/noise_config_aggressive.json",
        }

        # Available mask types for undersampling
        self.mask_types = [
            "G1D10",
            "G1D20",
            "G1D30",
            "G1D40",
            "G1D50",
            "G2D10",
            "G2D20",
            "G2D30",
            "G2D40",
            "G2D50",
            "R10",
            "R20",
            "R30",
            "R40",
            "R50",
            "R60",
            "R70",
            "R80",
            "R90",
            "S10",
            "S20",
            "S30",
            "S40",
            "S50",
            "S60",
            "S70",
            "S80",
            "S90",
        ]

        # Specific noise types to test individually
        self.individual_noise_types = [
            "gaussian_noise",
            "rician_noise",
            "ghosting",
            "aliasing",
            "line_noise",
            "zipper_artifact",
        ]

    def load_image(self, image_path):
        """Load and normalize an MRI image."""
        if image_path.endswith(".npy"):
            image = np.load(image_path).astype(np.float32)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # Ensure 2D and normalize to [0, 1]
        if image.ndim == 3:
            image = image[:, :, 0]
        image = (image - image.min()) / (image.max() - image.min())

        return image

    def apply_undersampling_mask(self, image, mask_type):
        """Apply k-space undersampling using specified mask."""
        # Create temporary options for mask loading
        temp_opt = {"mask": mask_type}
        mask = define_Mask(temp_opt)

        # Transform to k-space
        fft_img = fftshift(fft2(image))

        # Resize mask to match image if needed
        if mask.shape != fft_img.shape:
            from scipy.ndimage import zoom

            zoom_factors = (
                fft_img.shape[0] / mask.shape[0],
                fft_img.shape[1] / mask.shape[1],
            )
            mask = zoom(mask, zoom_factors, order=0)

        # Apply mask and transform back
        masked_fft = fft_img * mask
        undersampled_image = np.abs(ifft2(ifftshift(masked_fft)))

        return undersampled_image, mask

    def generate_comprehensive_variations(self, image_path, num_random_combinations=10):
        """
        Generate comprehensive noise variations including:
        1. Individual noise types
        2. Different noise intensities
        3. Various mask combinations
        4. Random combinations mimicking training
        """
        print(f"Loading image: {image_path}")
        original_image = self.load_image(image_path)

        # Save original image
        self.save_image(original_image, "00_original.png")

        # 1. Test individual noise types
        print("Generating individual noise type variations...")
        self._generate_individual_noise_variations(original_image)

        # 2. Test different noise intensities
        print("Generating different noise intensity variations...")
        self._generate_intensity_variations(original_image)

        # 3. Test various undersampling masks
        print("Generating undersampling mask variations...")
        self._generate_mask_variations(original_image)

        # 4. Generate random combinations (mimicking training)
        print("Generating random combination variations...")
        self._generate_random_combinations(original_image, num_random_combinations)

        # 5. Create comparison grid
        print("Creating comparison grids...")
        self._create_comparison_grids()

        print(f"All variations saved to: {self.output_dir}")

    def _generate_individual_noise_variations(self, image):
        """Test each noise type individually."""
        for config_name, config_path in self.noise_configs.items():
            if not os.path.exists(config_path):
                print(f"Config not found: {config_path}, skipping...")
                continue

            noise_gen = MRINoiseGenerator(config_path)

            for noise_type in self.individual_noise_types:
                try:
                    noisy_image = noise_gen.apply_noise(image, [noise_type])
                    filename = f"01_individual_{config_name}_{noise_type}.png"
                    self.save_image(noisy_image, filename)
                except Exception as e:
                    print(f"Error applying {noise_type} with {config_name}: {e}")

    def _generate_intensity_variations(self, image):
        """Test different noise intensities."""
        # Create custom configurations with varying intensities
        intensities = [0.1, 0.3, 0.5, 0.7, 0.9]

        for intensity in intensities:
            # Create temporary config with specific intensity
            custom_config = {
                "gaussian_noise": {
                    "enabled": True,
                    "weight": 1.0,
                    "variance_range": [0.001 * intensity, 0.1 * intensity],
                    "snr_range": [20, 50],
                },
                "rician_noise": {
                    "enabled": True,
                    "weight": 0.8,
                    "sigma_range": [0.01 * intensity, 0.05 * intensity],
                    "amplitude_range": [0.1 * intensity, 0.3 * intensity],
                },
            }

            noise_gen = MRINoiseGenerator()
            noise_gen.config = custom_config

            noisy_image = noise_gen.apply_noise(image)
            filename = f"02_intensity_{intensity:.1f}.png"
            self.save_image(noisy_image, filename)

    def _generate_mask_variations(self, image):
        """Test various undersampling masks."""
        # Test representative masks from each category
        representative_masks = [
            "G1D30",
            "G2D30",  # Gaussian
            "R30",
            "R50",
            "R70",  # Radial
            "S30",
            "S50",
            "S70",  # Spiral
        ]

        for mask_type in representative_masks:
            try:
                undersampled_image, mask = self.apply_undersampling_mask(
                    image, mask_type
                )
                filename = f"03_mask_{mask_type}.png"
                self.save_image(undersampled_image, filename)

                # Also save the mask itself
                mask_filename = f"03_mask_{mask_type}_pattern.png"
                self.save_image(mask, mask_filename)

            except Exception as e:
                print(f"Error applying mask {mask_type}: {e}")

    def _generate_random_combinations(self, image, num_combinations):
        """Generate random combinations mimicking real training."""
        import random

        # Use default noise configuration
        noise_gen = MRINoiseGenerator("configs/noise_config_default.json")

        for i in range(num_combinations):
            # Randomly select mask
            mask_type = random.choice(self.mask_types)

            # Apply noise first
            noisy_image = noise_gen.apply_noise(image)

            # Then apply random undersampling
            try:
                final_image, _ = self.apply_undersampling_mask(noisy_image, mask_type)
                filename = f"04_random_combo_{i + 1:02d}_{mask_type}.png"
                self.save_image(final_image, filename)
            except Exception as e:
                print(f"Error in random combination {i}: {e}")

    def _create_comparison_grids(self):
        """Create comparison grids for easy visualization."""
        # This is a placeholder for creating comparison grids
        # You can implement grid creation logic here if needed
        pass

    def save_image(self, image, filename):
        """Save image to output directory."""
        if image.ndim > 2:
            image = image.squeeze()

        # Ensure values are in [0, 1] range
        image = np.clip(image, 0, 1)

        # Convert to uint8 and save
        image_uint8 = (image * 255).astype(np.uint8)
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, image_uint8)
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive noise variations for MRI images"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input image path (.npy or image file)"
    )
    parser.add_argument(
        "--output", "-o", default="noise_variations", help="Output directory"
    )
    parser.add_argument(
        "--num_random", "-n", type=int, default=10, help="Number of random combinations"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist!")
        return

    # Create noise variation generator
    generator = NoiseVariationGenerator(args.output)

    # Generate all variations
    generator.generate_comprehensive_variations(args.input, args.num_random)

    print("\\n" + "=" * 50)
    print("NOISE VARIATION GENERATION COMPLETE!")
    print(f"Check output directory: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
