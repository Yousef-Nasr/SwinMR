#!/usr/bin/env python3
"""
Example usage of the simple inference script
"""

import os
import sys
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_inference import run_inference

def create_example_image():
    """Create a simple example image for testing"""
    # Create a simple test image (256x256)
    img = np.zeros((256, 256, 1), dtype=np.float32)
    
    # Add some patterns
    # Circle in center
    center = (128, 128)
    y, x = np.ogrid[:256, :256]
    mask = (x - center[0])**2 + (y - center[1])**2 <= 50**2
    img[mask, 0] = 0.8
    
    # Add some lines
    img[100:110, :, 0] = 0.6
    img[:, 100:110, 0] = 0.6
    
    # Add noise
    img += np.random.normal(0, 0.05, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    
    return img

def main():
    """Example usage"""
    print("SwinMR Inference Example")
    print("=" * 40)
    
    # Create example image
    print("Creating example image...")
    example_img = create_example_image()
    
    # Save example image
    example_path = "example_input.npy"
    np.save(example_path, example_img)
    print(f"Saved example image to: {example_path}")
    
    # Configuration file
    config_path = "options/SwinMR/example/train_swinmr_enhanced_noise.json"
    
    # Try to find a model weights file (look for latest checkpoint)
    import glob
    model_paths = glob.glob("experiments/*/models/*_G.pth")
    if not model_paths:
        print("Error: No model weights found!")
        print("Please train the model first or provide a path to model weights")
        print("Expected location: experiments/*/models/*_G.pth")
        return
    
    # Use the latest model (highest iteration number)
    weights_path = sorted(model_paths, key=lambda x: int(os.path.basename(x).split('_')[0]))[-1]
    print(f"Found model weights: {weights_path}")
    
    # Output directory
    output_dir = "inference_results"
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Please make sure you're running from the SwinMR directory")
        return
    
    # Run inference
    print("\nRunning inference...")
    try:
        psnr, ssim = run_inference(example_path, config_path, weights_path, output_dir)
        if psnr is not None and ssim is not None:
            print(f"\nExample completed!")
            print(f"Results: PSNR = {psnr:.4f} dB, SSIM = {ssim:.6f}")
            print(f"Check the '{output_dir}' folder for output images")
        else:
            print("Example failed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    if os.path.exists(example_path):
        os.remove(example_path)
        print(f"Cleaned up: {example_path}")

if __name__ == "__main__":
    main()