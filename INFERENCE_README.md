# SwinMR Simple Inference

This script provides a simple way to run inference with SwinMR models, mimicking exactly what happens during the testing phase of training.

## Features

✅ **Exact Testing Replication**: Uses the same preprocessing, corruption, and inference pipeline as training  
✅ **Multiple Input Formats**: Supports `.npy` files and regular images  
✅ **Aggressive Noise**: Uses the same enhanced noise configuration as training  
✅ **Complete Output**: Saves individual images + merged comparison  
✅ **Metrics Calculation**: Provides PSNR and SSIM scores  

## Quick Start

### 1. Basic Usage

```bash
# Run inference with specific model weights
python simple_inference.py --input path/to/image.npy --weights experiments/train_swinmr/models/10000_G.pth

# Specify output directory
python simple_inference.py --input image.npy --weights model.pth --output results/

# Use custom config
python simple_inference.py --input image.npy --weights model.pth --config custom_config.json --output results/
```

### 2. Example with Test Data

```bash
# Use an existing test image (replace with your actual model path)
python simple_inference.py --input testsets/db_test/imgGT_1_1.npy --weights experiments/train_swinmr/models/10000_G.pth

# Or create and test with example image (automatically finds latest model)
python inference_example.py
```

## Arguments

- `--input` / `-i`: Input image path (`.npy` or image file) **[Required]**
- `--weights` / `-w`: Path to model weights (.pth file) **[Required]**
- `--config` / `-c`: Model configuration file (default: `train_swinmr_enhanced_noise.json`)
- `--output` / `-o`: Output directory (default: `inference_results`)

## What the Script Does

### 1. **Load & Preprocess**
- Loads image (`.npy` or regular image formats)
- Normalizes to 0-1 range (same as training dataset)
- Handles NaN/Inf values safely

### 2. **Apply Corruption** 
- Uses random mask selection (same as training)
- Applies aggressive noise (multiple types: Gaussian, Rician, ghosting, aliasing, etc.)
- Performs k-space undersampling

### 3. **Run Inference**
- Feeds data to model exactly like testing loop
- Adjusts window size automatically
- Runs model in evaluation mode

### 4. **Generate Results**
- Calculates PSNR and SSIM metrics
- Saves 4 output images:
  - `*_input.png`: Corrupted input
  - `*_output.png`: Model reconstruction
  - `*_ground_truth.png`: Original image  
  - `*_comparison.png`: Side-by-side comparison

## Output Structure

```
inference_results/
├── example_input_input.png          # Corrupted input
├── example_input_output.png         # Model reconstruction  
├── example_input_ground_truth.png   # Original image
└── example_input_comparison.png     # Side-by-side: GT | Corrupted | Reconstructed
```

## Example Output

```
============================================================
SwinMR Simple Inference
============================================================
Input image: testsets/db_test/imgGT_1_1.npy
Model config: options/SwinMR/example/train_swinmr_enhanced_noise.json
Model weights: experiments/train_swinmr/models/15000_G.pth
Output dir: inference_results

Loading model...
Loading checkpoint: experiments/train_swinmr/models/15000_G.pth
Loading and preprocessing image...
Image shape: (256, 256, 1)
Creating corrupted image...
Using random mask: G1D30
Running inference...
Calculating metrics...
============================================================
RESULTS
============================================================
PSNR: 28.4567 dB
SSIM: 0.847234

Results saved to: inference_results
Files:
  - imgGT_1_1_input.png (corrupted input)
  - imgGT_1_1_output.png (reconstructed)
  - imgGT_1_1_ground_truth.png (original)
  - imgGT_1_1_comparison.png (side-by-side)
============================================================
```

## Requirements

- Trained SwinMR model (checkpoint files in `experiments/*/models/`)
- All training dependencies (PyTorch, PIL, cv2, etc.)
- Same environment as used for training

## Notes

- **Automatic Checkpoint Loading**: Finds and loads the latest checkpoint automatically
- **Random Corruption**: Each run applies different noise/masks for variety
- **GPU Support**: Automatically uses GPU if available
- **Memory Efficient**: Processes single images (no batching overhead)
- **Training Compatibility**: Results should match training validation exactly

## Troubleshooting

### No Checkpoint Found
```
Warning: No checkpoint found, using randomly initialized model
```
**Solution**: Make sure you have trained the model and checkpoints exist in `experiments/*/models/`

### CUDA Out of Memory
**Solution**: The script processes single images, so this is rare. Try reducing image size or using CPU.

### Import Errors
**Solution**: Make sure you're running from the SwinMR root directory and all dependencies are installed.

## Advanced Usage

### Custom Noise Configuration
```bash
# Edit the config file to use different noise settings
cp options/SwinMR/example/train_swinmr_enhanced_noise.json my_config.json
# Edit my_config.json to change noise_config_path, max_noise_types, etc.
python simple_inference.py --input image.npy --config my_config.json
```

### Batch Processing
```bash
# Process multiple images
for img in testsets/db_test/*.npy; do
    python simple_inference.py --input "$img" --output "results/$(basename "$img" .npy)/"
done
```

This script gives you the exact same results as you would see during training validation, making it perfect for testing your trained models on new data! 🎉