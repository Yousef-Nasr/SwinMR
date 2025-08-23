"""
# -----------------------------------------
Quick Usage Example for Enhanced Noise System
This script shows how to quickly use the enhanced features
# -----------------------------------------
"""

import os
import sys
import numpy as np

# Add the SwinMR directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.noise_generator import MRINoiseGenerator
from models.select_mask import define_Mask


def quick_noise_demo():
    """Quick demonstration of the noise generator."""
    print("🔧 Enhanced Noise System Demo")
    print("=" * 40)

    # Create a simple test image
    test_image = np.random.rand(256, 256) * 0.8 + 0.1

    # 1. Basic noise generation
    print("1. Applying default noise...")
    noise_gen = MRINoiseGenerator()
    noisy_image = noise_gen.apply_noise(test_image)
    print(f"   Original shape: {test_image.shape}")
    print(f"   Noisy shape: {noisy_image.shape}")
    print(
        f"   Noise difference: {np.mean(np.abs(noisy_image.squeeze() - test_image)):.4f}"
    )

    # 2. Specific noise types
    print("\n2. Testing individual noise types...")
    noise_types = ["gaussian_noise", "rician_noise", "ghosting"]
    for noise_type in noise_types:
        try:
            noisy = noise_gen.apply_noise(test_image, [noise_type])
            diff = np.mean(np.abs(noisy.squeeze() - test_image))
            print(f"   {noise_type}: difference = {diff:.4f}")
        except Exception as e:
            print(f"   {noise_type}: Error - {e}")

    # 3. Different configurations
    print("\n3. Testing different noise configurations...")
    configs = ["mild", "default", "aggressive"]
    for config_name in configs:
        config_path = f"configs/noise_config_{config_name}.json"
        if os.path.exists(config_path):
            config_gen = MRINoiseGenerator(config_path)
            noisy = config_gen.apply_noise(test_image)
            diff = np.mean(np.abs(noisy.squeeze() - test_image))
            print(f"   {config_name}: difference = {diff:.4f}")
        else:
            print(f"   {config_name}: Config file not found")


def quick_mask_demo():
    """Quick demonstration of random mask selection."""
    print("\n🎯 Random Mask Selection Demo")
    print("=" * 40)

    # Test different mask types
    mask_types = ["G1D30", "G2D30", "R30", "S30"]

    for mask_type in mask_types:
        try:
            temp_opt = {"mask": mask_type}
            mask = define_Mask(temp_opt)
            coverage = np.mean(mask)
            print(f"   {mask_type}: shape={mask.shape}, coverage={coverage:.3f}")
        except Exception as e:
            print(f"   {mask_type}: Error - {e}")


def quick_training_demo():
    """Show how to use in training configuration."""
    print("\n🚀 Training Configuration Demo")
    print("=" * 40)

    config_example = {
        "use_enhanced_noise": True,
        "noise_config_path": "configs/noise_config_default.json",
        "random_mask_selection": True,
        "available_masks": ["G1D30", "G2D30", "R30", "S30"],
    }

    print("Add these options to your training JSON:")
    print("```json")
    for key, value in config_example.items():
        if isinstance(value, list):
            print(f'  "{key}": [')
            for item in value:
                print(f'    "{item}",')
            print("  ]")
        else:
            print(f'  "{key}": {str(value).lower()}')
    print("```")


def main():
    """Main demo function."""
    print("🎉 SwinMR Enhanced Noise System - Quick Demo")
    print("=" * 50)

    try:
        quick_noise_demo()
        quick_mask_demo()
        quick_training_demo()

        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run: python test_enhanced_noise_system.py")
        print(
            "2. Train: python main_train_swinmr.py --opt options/SwinMR/example/train_swinmr_enhanced_noise.json"
        )
        print("3. Test: python generate_noise_variations.py --input your_image.npy")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please ensure all required files are in place.")


if __name__ == "__main__":
    main()
