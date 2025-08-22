#!/usr/bin/env python3
"""
# -----------------------------------------
Data Organization Utility Script
by SwinMR Enhancement
# -----------------------------------------
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_preparer import DataPreparer, prepare_data_from_config, create_data_prep_config
from utils.noise_generator import MRINoiseGenerator, create_default_noise_config


def create_folder_structure(base_path: str):
    """Create recommended folder structure for MRI data."""
    base = Path(base_path)
    
    # Main directories
    directories = [
        'raw_data',
        'processed_data/train',
        'processed_data/test',
        'configs',
        'models',
        'results'
    ]
    
    for dir_path in directories:
        (base / dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Created folder structure at: {base}")
    print("Recommended structure:")
    for dir_path in directories:
        print(f"  {dir_path}/")
    
    return base


def setup_configs(base_path: str):
    """Create default configuration files."""
    base = Path(base_path)
    configs_dir = base / 'configs'
    
    # Create noise configurations
    noise_configs = {
        'mild': configs_dir / 'noise_config_mild.json',
        'default': configs_dir / 'noise_config_default.json',
        'aggressive': configs_dir / 'noise_config_aggressive.json'
    }
    
    generator = MRINoiseGenerator()
    
    # Mild noise config
    mild_config = generator.config.copy()
    for noise_type in mild_config:
        mild_config[noise_type]['weight'] *= 0.5
        if 'intensity_range' in mild_config[noise_type]:
            intensity = mild_config[noise_type]['intensity_range']
            mild_config[noise_type]['intensity_range'] = [intensity[0] * 0.5, intensity[1] * 0.7]
    
    with open(noise_configs['mild'], 'w') as f:
        json.dump(mild_config, f, indent=2)
    
    # Default config
    generator.save_config(str(noise_configs['default']))
    
    # Aggressive config
    aggressive_config = generator.config.copy()
    for noise_type in aggressive_config:
        aggressive_config[noise_type]['weight'] = min(1.0, aggressive_config[noise_type]['weight'] * 1.5)
        if 'intensity_range' in aggressive_config[noise_type]:
            intensity = aggressive_config[noise_type]['intensity_range']
            aggressive_config[noise_type]['intensity_range'] = [intensity[0], min(1.0, intensity[1] * 1.3)]
    
    with open(noise_configs['aggressive'], 'w') as f:
        json.dump(aggressive_config, f, indent=2)
    
    # Create data prep config
    data_prep_config = {
        "source_dir": str(base / "raw_data"),
        "target_dir": str(base / "processed_data"),
        "train_ratio": 0.8,
        "organize_by_patient": True,
        "min_size": [64, 64],
        "max_size": [512, 512],
        "normalize": True
    }
    
    with open(configs_dir / 'data_prep_config.json', 'w') as f:
        json.dump(data_prep_config, f, indent=2)
    
    print(f"Configuration files created in: {configs_dir}")
    for name, path in noise_configs.items():
        print(f"  Noise config ({name}): {path.name}")
    print(f"  Data prep config: data_prep_config.json")


def prepare_data_interactive():
    """Interactive data preparation wizard."""
    print("\n=== MRI Data Preparation Wizard ===\n")
    
    # Get source directory
    source_dir = input("Enter source directory path (containing DICOM/JPG files): ").strip()
    if not Path(source_dir).exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Get target directory
    target_dir = input("Enter target directory path (where to save organized data): ").strip()
    
    # Get parameters
    train_ratio = float(input("Enter train/test split ratio (0.8): ") or "0.8")
    organize_by_patient = input("Organize by patient/series? (y/n) [y]: ").strip().lower() != 'n'
    normalize = input("Normalize images to [0,1]? (y/n) [y]: ").strip().lower() != 'n'
    
    # Min/max sizes
    min_size_str = input("Enter minimum image size (height,width) [64,64]: ").strip() or "64,64"
    max_size_str = input("Enter maximum image size (height,width) [512,512]: ").strip() or "512,512"
    
    try:
        min_size = tuple(map(int, min_size_str.split(',')))
        max_size = tuple(map(int, max_size_str.split(',')))
    except:
        print("Invalid size format. Using defaults.")
        min_size = (64, 64)
        max_size = (512, 512)
    
    print(f"\nPreparing data...")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Organize by patient: {organize_by_patient}")
    print(f"  Size range: {min_size} to {max_size}")
    
    # Create preparer and run
    preparer = DataPreparer(source_dir, target_dir, train_ratio)
    
    try:
        metadata = preparer.prepare_data(
            organize_by_patient=organize_by_patient,
            min_size=min_size,
            max_size=max_size,
            normalize=normalize
        )
        
        print(f"\n✓ Data preparation completed successfully!")
        print(f"  Total images processed: {metadata['total_images']}")
        print(f"  Training images: {metadata['train_images']}")
        print(f"  Test images: {metadata['test_images']}")
        
    except Exception as e:
        print(f"\n✗ Error during data preparation: {str(e)}")


def test_noise_generator():
    """Test the noise generator with sample data."""
    print("\n=== Testing Noise Generator ===\n")
    
    # Create test image
    import numpy as np
    test_image = np.random.rand(256, 256) * 0.8 + 0.1
    
    # Test different noise configurations
    configs = ['mild', 'default', 'aggressive']
    
    for config_name in configs:
        config_path = f"./configs/noise_config_{config_name}.json"
        if Path(config_path).exists():
            print(f"Testing {config_name} noise configuration...")
            
            generator = MRINoiseGenerator(config_path)
            noisy_image = generator.apply_noise(test_image.copy())
            
            # Calculate noise metrics
            mse = np.mean((test_image - noisy_image.squeeze()) ** 2)
            snr = 10 * np.log10(np.var(test_image) / mse) if mse > 0 else float('inf')
            
            print(f"  MSE: {mse:.6f}")
            print(f"  SNR: {snr:.2f} dB")
        else:
            print(f"Configuration not found: {config_path}")
    
    print("✓ Noise generator test completed")


def main():
    parser = argparse.ArgumentParser(description='MRI Data Organization Utility')
    parser.add_argument('command', choices=['setup', 'prepare', 'test-noise', 'interactive'],
                       help='Command to execute')
    parser.add_argument('--base-path', type=str, default='./mri_data',
                       help='Base path for data organization')
    parser.add_argument('--source', type=str, help='Source directory for data preparation')
    parser.add_argument('--target', type=str, help='Target directory for data preparation')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        # Setup folder structure and configs
        base_path = create_folder_structure(args.base_path)
        setup_configs(base_path)
        print(f"\n✓ Setup completed at: {base_path}")
        print("\nNext steps:")
        print("1. Copy your DICOM/JPG files to: raw_data/")
        print("2. Run: python organize_data.py prepare")
        print("3. Update configuration files in: configs/")
        
    elif args.command == 'prepare':
        if args.config:
            # Use config file
            prepare_data_from_config(args.config)
        elif args.source and args.target:
            # Use command line arguments
            preparer = DataPreparer(args.source, args.target)
            preparer.prepare_data()
        else:
            # Interactive mode
            prepare_data_interactive()
    
    elif args.command == 'test-noise':
        test_noise_generator()
    
    elif args.command == 'interactive':
        prepare_data_interactive()


if __name__ == "__main__":
    main()