# Enhanced Noise System for SwinMR

This document describes the enhanced noise generation system integrated into SwinMR for more realistic MRI reconstruction training.

## Overview

The enhanced noise system provides:
1. **Comprehensive noise types**: Gaussian, Rician, ghosting, aliasing, line noise, and zipper artifacts
2. **Random mask selection**: Dynamically choose from different undersampling patterns during training
3. **Configurable noise levels**: Mild, default, and aggressive noise configurations
4. **Noise variation generator**: Tool to create all noise variations from a single image

## Features

### 1. Multiple Noise Types
- **Gaussian Noise**: Applied in k-space domain
- **Rician Noise**: Magnitude-based noise common in MRI
- **Ghosting**: Motion artifacts creating ghost images
- **Aliasing**: Undersampling artifacts
- **Line Noise**: Horizontal/vertical line artifacts
- **Zipper Artifacts**: Periodic noise patterns

### 2. Random Mask Selection
- Supports Gaussian 1D/2D, Radial, and Spiral masks
- Random selection during training for better generalization
- All mask types from the original implementation

### 3. Configurable Noise Levels
- **Mild**: `configs/noise_config_mild.json` - Light noise for sensitive training
- **Default**: `configs/noise_config_default.json` - Balanced noise levels
- **Aggressive**: `configs/noise_config_aggressive.json` - Strong noise for robustness

### 4. Limited Noise Types (NEW!)
- **1-2 noise types per image**: Prevents over-noising during training
- **Smart selection**: Randomly chooses 1-2 noise types from available options
- **Mask as noise**: Can count mask selection as one of the noise types
- **Realistic training**: More closely mimics real-world MRI acquisition

## Usage

### 1. Training with Enhanced Noise

Use the provided configuration file:

```bash
python main_train_swinmr.py --opt options/SwinMR/example/train_swinmr_enhanced_noise.json
```

### 2. Configuration Options

Add these options to your training JSON configuration:

```json
{
  "datasets": {
    "train": {
      "use_enhanced_noise": true,
      "noise_config_path": "configs/noise_config_default.json",
      "random_mask_selection": true,
      "max_noise_types": 2,
      "min_noise_types": 1,
      "include_mask_as_noise": true,
      "available_masks": [
        "G1D10", "G1D20", "G1D30", "G1D40", "G1D50",
        "G2D10", "G2D20", "G2D30", "G2D40", "G2D50",
        "R10", "R20", "R30", "R40", "R50", "R60", "R70", "R80", "R90",
        "S10", "S20", "S30", "S40", "S50", "S60", "S70", "S80", "S90"
      ]
    }
  }
}
```

### 3. Generate Noise Variations

Test noise effects on individual images:

```bash
python generate_noise_variations.py --input path/to/image.npy --output noise_output --num_random 10
```

This creates:
- Individual noise type variations
- Different intensity levels
- Various mask combinations
- Random combinations mimicking training

### 4. Test the System

Run the test script to verify everything works:

```bash
python test_enhanced_noise_system.py
```

## Configuration Parameters

### Enhanced Noise Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_enhanced_noise` | Enable comprehensive noise generator | `false` |
| `noise_config_path` | Path to noise configuration JSON | `null` |
| `random_mask_selection` | Randomly select masks during training | `false` |
| `available_masks` | List of mask types to randomly choose from | `["G1D30"]` |
| `max_noise_types` | Maximum number of noise types per image (1-2) | `2` |
| `min_noise_types` | Minimum number of noise types per image (1-2) | `1` |
| `include_mask_as_noise` | Count mask selection as one noise type | `true` |

### Noise Configuration Structure

```json
{
  "gaussian_noise": {
    "enabled": true,
    "weight": 1.0,
    "variance_range": [0.001, 0.1],
    "snr_range": [20, 50]
  },
  "rician_noise": {
    "enabled": true,
    "weight": 0.8,
    "sigma_range": [0.01, 0.05],
    "amplitude_range": [0.1, 0.3]
  }
}
```

## Available Mask Types

### Gaussian 1D
- `G1D10`, `G1D20`, `G1D30`, `G1D40`, `G1D50` (10%, 20%, 30%, 40%, 50% sampling)

### Gaussian 2D  
- `G2D10`, `G2D20`, `G2D30`, `G2D40`, `G2D50`

### Radial
- `R10` through `R90` (10% to 90% sampling)

### Spiral
- `S10` through `S90` (10% to 90% sampling)

## Training Progress Display

The system now shows real-time training progress:

```
Epoch:  1 | Iter:      1 | LR:1.000e-04 | G_loss:0.8234 | Img:0.4567 | Freq:0.2234 | Perc:0.1433
Epoch:  1 | Iter:      2 | LR:1.000e-04 | G_loss:0.7834 | Img:0.4367 | Freq:0.2134 | Perc:0.1333
...
Epoch:  1 | Iter:  1,000 | LR:9.950e-05 | G_loss:0.4234 | Img:0.2567 | Freq:0.1234 | Perc:0.0433

[Testing at iteration 1000]
Testing: 50/100 (50.0%)
```

## File Structure

```
SwinMR/
├── configs/
│   ├── noise_config_mild.json       # Mild noise settings
│   ├── noise_config_default.json    # Default noise settings
│   └── noise_config_aggressive.json # Aggressive noise settings
├── utils/
│   └── noise_generator.py          # Main noise generator class
├── data/
│   └── dataset_CCsagnpi.py         # Enhanced dataset with noise integration
├── options/SwinMR/example/
│   └── train_swinmr_enhanced_noise.json  # Example training configuration
├── generate_noise_variations.py    # Noise variation generator tool
└── test_enhanced_noise_system.py   # System test script
```

## Benefits

1. **Improved Generalization**: Random noise and mask selection prevent overfitting
2. **Realistic Training**: Multiple noise types simulate real MRI artifacts
3. **Flexible Configuration**: Easy to adjust noise levels and types
4. **Better Robustness**: Models trained with enhanced noise handle real-world data better
5. **Real-time Monitoring**: Progress display helps track training status

## Examples

### Basic Training
```bash
# Train with enhanced noise using default settings
python main_train_swinmr.py --opt options/SwinMR/example/train_swinmr_enhanced_noise.json
```

### Custom Noise Configuration
```bash
# Create custom noise config and use it
cp configs/noise_config_default.json configs/my_custom_noise.json
# Edit my_custom_noise.json as needed
# Update training config to use "noise_config_path": "configs/my_custom_noise.json"
```

### Generate Test Variations
```bash
# Test noise on your own image
python generate_noise_variations.py -i testsets/db_test/imgGT_1_1.npy -o my_noise_test -n 15
```

## Noise Type Selection Patterns

### Pattern 1: Mask as Noise Type (include_mask_as_noise = true)
```
Sample 1: Mask: G1D30 only (1 noise type total)
Sample 2: Mask: R50 + Noise: gaussian_noise (2 noise types total)
Sample 3: Mask: S30 + Noise: rician_noise (2 noise types total)
Sample 4: Mask: G2D40 only (1 noise type total)
```

### Pattern 2: Mask Separate (include_mask_as_noise = false)
```
Sample 1: Mask: G1D30 + Noise: gaussian_noise (mask + 1 noise)
Sample 2: Mask: R50 + Noise: [ghosting, line_noise] (mask + 2 noise)
Sample 3: Mask: S30 + Noise: rician_noise (mask + 1 noise)
Sample 4: Mask: G2D40 + Noise: [aliasing, zipper_artifact] (mask + 2 noise)
```

### Benefits of Limited Noise Types
- **Prevents over-corruption**: Too many noise types can make images unrecoverable
- **Realistic simulation**: Real MRI scans typically have 1-2 dominant artifacts
- **Better training**: Models learn to handle specific artifact combinations
- **Improved generalization**: Each sample focuses on different noise patterns

This enhanced noise system provides a comprehensive solution for training robust MRI reconstruction models that can handle real-world artifacts and variations.
