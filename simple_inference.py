#!/usr/bin/env python3
"""
Simple Inference Script for SwinMR
Mimics the exact testing process from training loop
Usage: python simple_inference.py --input path/to/image.npy --output results/
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from utils import utils_option as option
from utils import utils_image as util
from models.select_model import define_Model
from models.select_mask import define_Mask
from utils.noise_generator import MRINoiseGenerator
from utils.utils_swinmr import *


def load_and_preprocess_image(image_path):
    """
    Load and preprocess image exactly like the dataset loader
    """
    if image_path.endswith('.npy'):
        # Load .npy file
        img = np.load(image_path).astype(np.float32)
    else:
        # Load regular image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # Handle different shapes
    if len(img.shape) == 2:
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    elif len(img.shape) == 3 and img.shape[2] != 1:
        img = img[:, :, 0:1]  # Take first channel
    
    # Check for NaN or Inf values and handle them
    if np.any(np.isnan(img)) or np.any(np.isinf(img)):
        print(f"Warning: NaN or Inf found in {image_path}, replacing with zeros")
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize to 0 ~ 1 with safe division (exactly like dataset)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        print(f"Warning: Constant values in {image_path}, setting to zeros")
        img = np.zeros_like(img)
    
    return img


def create_corrupted_image(img, opt):
    """
    Create corrupted image exactly like the dataset does
    """
    # Get mask
    mask_generator = define_Mask(opt)
    
    # Select random mask if multiple available
    if hasattr(opt, 'available_masks') and len(opt['available_masks']) > 1:
        selected_mask = np.random.choice(opt['available_masks'])
        print(f"Using random mask: {selected_mask}")
        # Update mask type temporarily
        original_mask = opt['mask']
        opt['mask'] = selected_mask
        mask_generator = define_Mask(opt)
        opt['mask'] = original_mask  # Restore original
    
    mask = mask_generator
    
    # Initialize noise generator if enhanced noise is enabled
    use_enhanced_noise = opt.get("use_enhanced_noise", False)
    if use_enhanced_noise:
        noise_config_path = opt.get("noise_config_path", "configs/noise_config_default.json")
        noise_generator = MRINoiseGenerator(noise_config_path)
        
        # Apply enhanced noise (exactly like dataset)
        max_noise_types = opt.get("max_noise_types", 2)
        min_noise_types = opt.get("min_noise_types", 1)
        
        img_corrupted = undersample_kspace_enhanced(
            img, mask, noise_generator, 
            max_noise_types, min_noise_types
        )
    else:
        # Apply basic undersampling
        img_corrupted = undersample_kspace_basic(
            img, mask, 
            opt.get("is_noise", False),
            opt.get("noise_level", 0.0),
            opt.get("noise_var", 0.1)
        )
    
    return img_corrupted, mask


def undersample_kspace_basic(img, mask, is_noise=False, noise_level=0.0, noise_var=0.1):
    """
    Basic k-space undersampling (mimics dataset function)
    """
    # Convert to k-space
    kspace = np.fft.fftshift(np.fft.fft2(img[:, :, 0]))
    
    # Apply mask
    kspace_masked = kspace * mask
    
    # Add noise if specified
    if is_noise:
        noise_real = np.random.normal(0, noise_var, kspace_masked.shape)
        noise_imag = np.random.normal(0, noise_var, kspace_masked.shape)
        noise = noise_real + 1j * noise_imag
        kspace_masked = kspace_masked + noise_level * noise
    
    # Convert back to image space
    img_corrupted = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
    img_corrupted = np.expand_dims(img_corrupted, axis=2)
    
    # Normalize
    img_min, img_max = img_corrupted.min(), img_corrupted.max()
    if img_max > img_min:
        img_corrupted = (img_corrupted - img_min) / (img_max - img_min)
    
    return img_corrupted


def undersample_kspace_enhanced(img, mask, noise_generator, max_noise_types, min_noise_types):
    """
    Enhanced k-space undersampling with multiple noise types
    """
    # Start with basic undersampling
    img_corrupted = undersample_kspace_basic(img, mask)
    
    # Apply enhanced noise
    num_noise_types = np.random.randint(min_noise_types, max_noise_types + 1)
    img_corrupted = noise_generator.apply_noise(img_corrupted[:, :, 0])
    img_corrupted = np.expand_dims(img_corrupted, axis=2)
    
    return img_corrupted


def run_inference(image_path, model_config_path, model_weights_path, output_dir, no_corruption=False):
    """
    Run inference exactly like the testing loop
    """
    print("=" * 60)
    print("SwinMR Simple Inference")
    print("=" * 60)
    print(f"Input image: {image_path}")
    print(f"Model config: {model_config_path}")
    print(f"Model weights: {model_weights_path}")
    print(f"Output dir: {output_dir}")
    print(f"Corruption: {'Disabled (clean image)' if no_corruption else 'Enabled (noise + mask)'}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    opt = option.parse(model_config_path, is_train=False)
    
    # Add missing netG fields if not present
    if 'out_chans' not in opt['netG']:
        opt['netG']['out_chans'] = opt['netG']['in_chans']
    if 'init_bn_type' not in opt['netG']:
        opt['netG']['init_bn_type'] = 'uniform'
    if 'init_gain' not in opt['netG']:
        opt['netG']['init_gain'] = 0.02
    
    # Add required path configuration for inference
    if 'path' not in opt:
        opt['path'] = {}
    if 'models' not in opt['path']:
        opt['path']['models'] = os.path.dirname(model_weights_path) if model_weights_path else 'models'
    if 'log' not in opt['path']:
        opt['path']['log'] = 'logs'
    if 'images' not in opt['path']:
        opt['path']['images'] = output_dir
    
    # Ensure datasets section exists
    if 'datasets' not in opt:
        opt['datasets'] = {}
    
    # Create safe test dataset configuration
    train_config = opt['datasets'].get('train', {})
    opt['datasets']['test'] = {
        'dataset_type': 'ccsagnpi',
        'dataroot_H': os.path.dirname(image_path) if image_path else '.',
        'mask': train_config.get('mask', 'G1D30'),
        'use_enhanced_noise': train_config.get('use_enhanced_noise', False),
        'noise_config_path': train_config.get('noise_config_path', None),
        'available_masks': train_config.get('available_masks', ['G1D30']),
        'max_noise_types': train_config.get('max_noise_types', 2),
        'min_noise_types': train_config.get('min_noise_types', 1),
        'is_noise': train_config.get('is_noise', False),
        'noise_level': train_config.get('noise_level', 0.0),
        'noise_var': train_config.get('noise_var', 0.1),
        'H_size': train_config.get('H_size', 96),
        'n_channels': opt.get('n_channels', 1),
    }
    
    # Initialize model
    print("Loading model...")
    model = define_Model(opt)
    
    # Load the specified checkpoint
    if model_weights_path and os.path.exists(model_weights_path):
        print(f"Loading checkpoint: {model_weights_path}")
        model.load_network(model_weights_path, model.netG, strict=True, param_key='params')
    else:
        print(f"Error: Model weights file not found: {model_weights_path}")
        print("Please provide a valid path to model weights (.pth file)")
        return None, None
    
    model.netG.eval()
    
    # Load and preprocess image
    print("Loading and preprocessing image...")
    img_gt = load_and_preprocess_image(image_path)
    print(f"Image shape: {img_gt.shape}")
    
    # Create corrupted version or use clean image
    if no_corruption:
        print("Using clean image (no corruption applied)...")
        img_corrupted = img_gt.copy()
        mask = None
    else:
        print("Creating corrupted image...")
        img_corrupted, mask = create_corrupted_image(img_gt, opt['datasets']['test'])
    
    # Convert to tensors (exactly like dataset)
    img_corrupted_tensor = util.float2tensor3(img_corrupted)
    img_gt_tensor = util.float2tensor3(img_gt)
    
    # Create data dict (exactly like dataset)
    test_data = {
        'L': img_corrupted_tensor.unsqueeze(0),  # Add batch dimension
        'H': img_gt_tensor.unsqueeze(0),         # Add batch dimension
        'H_path': image_path,
        'mask': mask,
        'SM': 0,
        'img_info': os.path.splitext(os.path.basename(image_path))[0]
    }
    
    # Run inference (exactly like testing loop)
    print("Running inference...")
    with torch.no_grad():
        # Feed data to model
        model.feed_data(test_data)
        
        # Adjust window size if needed
        model.check_windowsize()
        
        # Run inference
        model.test()
        
        # Recover window size
        model.recover_windowsize()
        
        # Get results
        results = model.current_results_gpu()
    
    # Extract results
    L_img = results["L"]  # Corrupted input
    E_img = results["E"]  # Predicted output
    H_img = results["H"]  # Ground truth
    
    # Calculate metrics
    print("Calculating metrics...")
    
    # Convert to CPU numpy
    L_img_np = util.tensor2float(L_img)
    E_img_np = util.tensor2float(E_img)
    H_img_np = util.tensor2float(H_img)
    
    # Calculate PSNR and SSIM
    psnr = util.calculate_psnr_single(H_img_np, E_img_np, border=0)
    ssim = util.calculate_ssim_single(H_img_np, E_img_np, border=0)
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.6f}")
    print()
    
    # Save individual images
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_clean" if no_corruption else "_corrupted"
    
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_input{suffix}.png"),
        np.clip(L_img_np, 0, 1) * 255
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_output.png"),
        np.clip(E_img_np, 0, 1) * 255
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_ground_truth.png"),
        np.clip(H_img_np, 0, 1) * 255
    )
    
    # Create merged comparison image (exactly like training)
    h, w = H_img_np.shape[:2]
    label_height = 30
    merged_img = np.ones((h + label_height, w * 3), dtype=np.float32)
    
    # Place images side by side
    merged_img[label_height:, :w] = H_img_np
    merged_img[label_height:, w:2*w] = L_img_np
    merged_img[label_height:, 2*w:3*w] = E_img_np
    
    # Add labels
    try:
        merged_pil = Image.fromarray((np.clip(merged_img, 0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(merged_pil)
        draw.text((w//2-40, 5), "Ground Truth", fill=0)
        input_label = "Clean" if no_corruption else "Corrupted"
        draw.text((w+w//2-20, 5), input_label, fill=0)
        draw.text((2*w+w//2-30, 5), "Reconstructed", fill=0)
        
        merged_pil.save(os.path.join(output_dir, f"{base_name}_comparison{suffix}.png"))
    except:
        # Fallback without labels
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_comparison{suffix}.png"),
            np.clip(merged_img, 0, 1) * 255
        )
    
    print(f"Results saved to: {output_dir}")
    print(f"Files:")
    input_desc = "clean input" if no_corruption else "corrupted input"
    print(f"  - {base_name}_input{suffix}.png ({input_desc})")
    print(f"  - {base_name}_output.png (reconstructed)")
    print(f"  - {base_name}_ground_truth.png (original)")
    print(f"  - {base_name}_comparison{suffix}.png (side-by-side)")
    print("=" * 60)
    
    return psnr, ssim


def main():
    parser = argparse.ArgumentParser(description='Simple SwinMR Inference')
    parser.add_argument('--input', '-i', required=True,
                       help='Input image path (.npy or image file)')
    parser.add_argument('--weights', '-w', default="13000_G.pth",
                       help='Path to model weights (.pth file) - will search in experiments/*/models/ if not found')
    parser.add_argument('--config', '-c', 
                       default='options/SwinMR/example/train_swinmr_enhanced_noise.json',
                       help='Model configuration file')
    parser.add_argument('--output', '-o', default='inference_results',
                       help='Output directory')
    parser.add_argument('--no-corruption', action='store_true',
                       help='Skip noise and mask corruption (process clean image directly)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    # Check if weights path is absolute or relative
    if not os.path.isabs(args.weights):
        # Try to find the weights file in common locations
        possible_paths = [
            args.weights,  # Current directory
            f"experiments/*/models/{args.weights}",  # Standard experiment directory
            f"models/{args.weights}",  # Models directory
        ]
        
        weights_found = False
        for path_pattern in possible_paths:
            if '*' in path_pattern:
                import glob
                matches = glob.glob(path_pattern)
                if matches:
                    args.weights = matches[0]  # Use first match
                    weights_found = True
                    break
            elif os.path.exists(path_pattern):
                args.weights = path_pattern
                weights_found = True
                break
        
        if not weights_found:
            print(f"Error: Model weights file not found: {args.weights}")
            print("Searched in:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease provide the full path to your model weights (.pth file)")
            return
    elif not os.path.exists(args.weights):
        print(f"Error: Model weights file does not exist: {args.weights}")
        return
    
    print(f"Using model weights: {args.weights}")
    
    if not os.path.exists(args.config):
        print(f"Error: Config file does not exist: {args.config}")
        return
    
    # Run inference
    try:
        psnr, ssim = run_inference(args.input, args.config, args.weights, args.output, args.no_corruption)
        if psnr is not None and ssim is not None:
            print(f"\nInference completed successfully!")
            print(f"PSNR: {psnr:.4f} dB, SSIM: {ssim:.6f}")
        else:
            print("Inference failed!")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()