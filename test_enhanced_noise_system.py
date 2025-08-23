"""
# -----------------------------------------
Test Enhanced Noise System
This script demonstrates the integrated noise system
# -----------------------------------------
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the SwinMR directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.noise_generator import MRINoiseGenerator
from data.dataset_CCsagnpi import DatasetCCsagnpi
import json


def create_sample_image():
    """Create a sample MRI-like image for testing."""
    # Create a synthetic brain-like image
    size = 256
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))

    # Create circular regions
    brain = np.zeros((size, size))

    # Main brain region
    brain_mask = (x**2 + y**2) < 0.8
    brain[brain_mask] = 0.8

    # Add some structure
    for i in range(3):
        center_x, center_y = np.random.uniform(-0.5, 0.5, 2)
        structure_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < 0.2
        brain[structure_mask] = 0.6 + i * 0.1

    # Add noise for realism
    brain += np.random.normal(0, 0.02, brain.shape)
    brain = np.clip(brain, 0, 1)

    return brain


def test_noise_generator():
    """Test the noise generator with different configurations."""
    print("Testing Noise Generator...")

    # Create sample image
    sample_image = create_sample_image()

    # Test with different noise configurations
    configs = ["mild", "default", "aggressive"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Show original
    axes[0].imshow(sample_image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    idx = 1
    for config_name in configs:
        config_path = f"configs/noise_config_{config_name}.json"

        if os.path.exists(config_path):
            noise_gen = MRINoiseGenerator(config_path)
            noisy_image = noise_gen.apply_noise(sample_image)

            axes[idx].imshow(noisy_image, cmap="gray")
            axes[idx].set_title(f"{config_name.capitalize()} Noise")
            axes[idx].axis("off")
            idx += 1
        else:
            print(f"Config file not found: {config_path}")

    # Test individual noise types
    noise_gen = MRINoiseGenerator()
    individual_types = ["gaussian_noise", "rician_noise", "ghosting", "line_noise"]

    for noise_type in individual_types[:4]:
        if idx < len(axes):
            noisy_image = noise_gen.apply_noise(sample_image, [noise_type])
            axes[idx].imshow(noisy_image, cmap="gray")
            axes[idx].set_title(f"{noise_type.replace('_', ' ').title()}")
            axes[idx].axis("off")
            idx += 1

    plt.tight_layout()
    plt.savefig("noise_test_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Noise generator test completed. Results saved to 'noise_test_results.png'")


def test_dataset_integration():
    """Test the enhanced dataset with noise integration."""
    print("Testing Dataset Integration...")

    # Create a minimal dataset configuration for testing
    test_config = {
        "n_channels": 1,
        "H_size": 96,
        "phase": "train",
        "dataroot_H": "testsets/db_test",  # Use existing test data
        "mask": "G1D30",
        # Enhanced noise options
        "use_enhanced_noise": True,
        "noise_config_path": "configs/noise_config_default.json",
        "random_mask_selection": True,
        "available_masks": ["G1D30", "G1D40", "R30", "S30"],
        # Legacy options
        "is_noise": False,
        "noise_level": 0.0,
        "noise_var": 0.1,
        "is_mini_dataset": False,
        "mini_dataset_prec": 1,
    }

    try:
        # Create dataset instance
        dataset = DatasetCCsagnpi(test_config)

        if len(dataset) > 0:
            # Test loading a few samples
            print(f"Dataset loaded successfully with {len(dataset)} samples")

            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(
                    f"Sample {i}: L shape={sample['L'].shape}, H shape={sample['H'].shape}"
                )
                print(f"Sample {i}: Image info={sample['img_info']}")
        else:
            print("No samples found in dataset. Please check the dataroot_H path.")

    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("This is expected if test data is not available.")


def create_enhanced_config_example():
    """Create an example configuration with enhanced noise options."""
    config = {
        "task": "swinmr_enhanced_noise_example",
        "model": "swinmr_npi",
        "mask": "G1D30",
        "gpu_ids": [0],
        "dist": False,
        "n_channels": 1,
        "manual_seed": 42,
        "datasets": {
            "train": {
                "name": "train_dataset",
                "dataset_type": "ccsagnpi",
                "dataroot_H": "./testsets/db_test",
                "mask": "G1D30",
                "H_size": 96,
                # Enhanced noise configuration
                "use_enhanced_noise": True,
                "noise_config_path": "configs/noise_config_default.json",
                "random_mask_selection": True,
                "available_masks": [
                    "G1D20",
                    "G1D30",
                    "G1D40",
                    "R30",
                    "R40",
                    "R50",
                    "S30",
                    "S40",
                    "S50",
                ],
                # Training options
                "is_mini_dataset": False,
                "dataloader_shuffle": True,
                "dataloader_batch_size": 2,
            }
        },
    }

    output_path = "example_enhanced_config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Example configuration saved to: {output_path}")


def main():
    """Main test function."""
    print("=" * 60)
    print("TESTING ENHANCED NOISE SYSTEM FOR SWINMR")
    print("=" * 60)

    # Test 1: Noise generator
    test_noise_generator()
    print()

    # Test 2: Dataset integration
    test_dataset_integration()
    print()

    # Test 3: Create example config
    create_enhanced_config_example()
    print()

    print("=" * 60)
    print("TESTING COMPLETED!")
    print("=" * 60)
    print("\nTo use the enhanced noise system:")
    print(
        "1. Use the configuration file: options/SwinMR/example/train_swinmr_enhanced_noise.json"
    )
    print(
        "2. Run: python main_train_swinmr.py --opt options/SwinMR/example/train_swinmr_enhanced_noise.json"
    )
    print("3. Use generate_noise_variations.py to test noise on individual images")


if __name__ == "__main__":
    main()
