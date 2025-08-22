#!/usr/bin/env python3
"""
# -----------------------------------------
Example Usage of Enhanced MRI Data System
by SwinMR Enhancement
# -----------------------------------------
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_sample_data(output_dir: str, num_images: int = 20):
    """Create sample MRI-like data for demonstration."""
    print(f"Creating {num_images} sample images in {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        # Create realistic MRI-like image
        # Simulate brain-like structure with multiple regions
        image = np.zeros((256, 256), dtype=np.float32)
        
        # Add circular brain outline
        y, x = np.ogrid[:256, :256]
        center_y, center_x = 128, 128
        brain_mask = (x - center_x)**2 + (y - center_y)**2 < 120**2
        
        # Add different tissue intensities
        image[brain_mask] = 0.3 + 0.4 * np.random.rand(np.sum(brain_mask))
        
        # Add some structure (simulate white matter, gray matter)
        inner_mask = (x - center_x)**2 + (y - center_y)**2 < 80**2
        image[inner_mask] *= 1.5
        
        # Add CSF regions (darker)
        csf_mask = ((x - center_x)**2 + (y - center_y)**2 < 40**2) & brain_mask
        image[csf_mask] *= 0.3
        
        # Add some random anatomical variation
        image += 0.1 * np.random.rand(256, 256) * brain_mask
        
        # Normalize to [0, 1]
        image = np.clip(image, 0, 1)
        
        # Save as numpy array
        filename = f"sample_brain_{i:03d}.npy"
        np.save(output_path / filename, image)
    
    print(f"✓ Created {num_images} sample images")
    return output_path


def demonstrate_noise_generation():
    """Demonstrate the noise generation capabilities."""
    print("\n=== Noise Generation Demonstration ===")
    
    try:
        from utils.noise_generator import MRINoiseGenerator
        
        # Create a test image
        test_image = np.random.rand(128, 128) * 0.8 + 0.1
        
        # Test different noise configurations
        configs = {
            'mild': './configs/noise_config_mild.json',
            'default': './configs/noise_config_default.json', 
            'aggressive': './configs/noise_config_aggressive.json'
        }
        
        print("Testing noise generation with different configurations:")
        
        for config_name, config_path in configs.items():
            if Path(config_path).exists():
                generator = MRINoiseGenerator(config_path)
                noisy_image = generator.apply_noise(test_image.copy())
                
                # Calculate metrics
                mse = np.mean((test_image - noisy_image.squeeze()) ** 2)
                snr = 10 * np.log10(np.var(test_image) / mse) if mse > 0 else float('inf')
                
                print(f"  {config_name:12} - MSE: {mse:.6f}, SNR: {snr:.2f} dB")
            else:
                print(f"  {config_name:12} - Configuration not found")
        
        # Demonstrate selective noise application
        print("\nTesting selective noise types:")
        generator = MRINoiseGenerator()
        
        noise_types = ['gaussian_noise', 'rician_noise', 'line_noise', 'ghosting']
        for noise_type in noise_types:
            try:
                noisy_image = generator.apply_noise(test_image.copy(), [noise_type])
                mse = np.mean((test_image - noisy_image.squeeze()) ** 2)
                print(f"  {noise_type:15} - Applied successfully (MSE: {mse:.6f})")
            except Exception as e:
                print(f"  {noise_type:15} - Error: {str(e)}")
        
        print("✓ Noise generation demonstration completed")
        
    except ImportError as e:
        print(f"✗ Cannot demonstrate noise generation: {e}")
        print("Please install required dependencies: numpy, scipy, opencv-python")


def demonstrate_data_preparation():
    """Demonstrate the data preparation workflow."""
    print("\n=== Data Preparation Demonstration ===")
    
    try:
        from utils.data_preparer import DataPreparer
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Create sample source data
            source_dir = Path(temp_dir) / 'source_data'
            target_dir = Path(temp_dir) / 'organized_data'
            
            # Create sample data
            sample_data_path = create_sample_data(str(source_dir), num_images=10)
            
            # Prepare data
            print(f"\nOrganizing data from {source_dir} to {target_dir}...")
            
            preparer = DataPreparer(
                source_dir=str(source_dir),
                target_dir=str(target_dir),
                train_ratio=0.7
            )
            
            metadata = preparer.prepare_data(
                organize_by_patient=False,
                min_size=(128, 128),
                max_size=(512, 512),
                normalize=True
            )
            
            # Display results
            print("\n📊 Data Preparation Results:")
            print(f"  Total images: {metadata['total_images']}")
            print(f"  Training images: {metadata['train_images']}")
            print(f"  Test images: {metadata['test_images']}")
            print(f"  Train ratio: {metadata['train_ratio']:.2f}")
            
            # Show file structure
            print(f"\n📁 Generated file structure:")
            for root, dirs, files in os.walk(target_dir):
                level = root.replace(str(target_dir), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:3]:  # Show first 3 files
                    print(f"{subindent}{file}")
                if len(files) > 3:
                    print(f"{subindent}... and {len(files) - 3} more files")
        
        print("✓ Data preparation demonstration completed")
        
    except ImportError as e:
        print(f"✗ Cannot demonstrate data preparation: {e}")
        print("Please install required dependencies: numpy, opencv-python, pathlib")


def demonstrate_integration():
    """Demonstrate integration with SwinMR training configuration."""
    print("\n=== Integration Demonstration ===")
    
    # Show how to use the enhanced dataset in training
    example_config = {
        "datasets": {
            "train": {
                "name": "enhanced_train_dataset",
                "dataset_type": "enhanced",
                "dataroot_H": "/path/to/your/train/data",
                "mask": "G1D30",
                "H_size": 96,
                
                # Enhanced noise configuration
                "use_enhanced_noise": True,
                "noise_config_path": "./configs/noise_config_default.json",
                "noise_types": None,  # Use all enabled noise types
                
                # Standard parameters
                "dataloader_shuffle": True,
                "dataloader_num_workers": 8,
                "dataloader_batch_size": 4
            },
            "test": {
                "name": "enhanced_test_dataset", 
                "dataset_type": "enhanced",
                "dataroot_H": "/path/to/your/test/data",
                "mask": "G1D30",
                
                # Use milder noise for testing
                "use_enhanced_noise": True,
                "noise_config_path": "./configs/noise_config_mild.json"
            }
        }
    }
    
    print("Enhanced dataset configuration example:")
    print("```json")
    import json
    print(json.dumps(example_config, indent=2))
    print("```")
    
    print("\nKey advantages of the enhanced system:")
    print("  ✓ No sensitivity maps required (simplified data requirements)")
    print("  ✓ Support for DICOM and standard image formats")
    print("  ✓ Realistic noise simulation with 6 different artifact types")
    print("  ✓ Configurable noise parameters via JSON files")
    print("  ✓ Easy data organization with automatic train/test splitting")
    print("  ✓ Backward compatibility with existing SwinMR configurations")
    
    print("\n🚀 Ready to use with existing SwinMR training scripts!")


def main():
    """Main demonstration function."""
    print("🧠 Enhanced MRI Data System Demonstration")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('./utils/noise_generator.py').exists():
        print("⚠️  Please run this script from the SwinMR root directory")
        return
    
    try:
        # Run demonstrations
        demonstrate_noise_generation()
        demonstrate_data_preparation()
        demonstrate_integration()
        
        print("\n" + "=" * 50)
        print("🎉 Demonstration completed successfully!")
        print("\nNext steps to use the enhanced system:")
        print("1. Install dependencies: pip install -r requirements_enhanced.txt")
        print("2. Organize your data: python organize_data.py setup")
        print("3. Copy DICOM/JPG files to: raw_data/")
        print("4. Prepare data: python organize_data.py prepare")
        print("5. Train with enhanced dataset using provided example configs")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        print("Please ensure all required files are present and dependencies are installed.")


if __name__ == "__main__":
    main()