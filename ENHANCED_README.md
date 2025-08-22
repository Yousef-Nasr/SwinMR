# Enhanced MRI Data Preparation and Noise Generation

This enhancement to SwinMR provides comprehensive data preparation utilities and advanced noise generation capabilities for DICOM and JPG/PNG images, eliminating the need for sensitivity maps (PI training).

## Features

### Data Preparation
- **Multi-format support**: DICOM (.dcm, .dicom), JPG, PNG, BMP, TIFF, and NumPy arrays
- **Automatic organization**: Split data into train/test sets with patient-aware grouping
- **Flexible preprocessing**: Image normalization, size filtering, and format conversion
- **Metadata generation**: Comprehensive statistics and processing information

### Advanced Noise Generation
- **Rician noise**: Realistic magnitude MRI noise distribution
- **Gaussian noise**: K-space and image-space noise simulation
- **Ghosting artifacts**: Motion-induced phase encoding errors
- **Aliasing artifacts**: Undersampling-related wraparound effects
- **Line noise**: Vertical/horizontal line artifacts (max 25% coverage)
- **Zipper artifacts**: Periodic k-space interference patterns

### Configuration-Driven
- **JSON-based**: Easy-to-modify noise and data preparation parameters
- **Weight-based control**: Probabilistic application of each noise type
- **Range specifications**: Customizable intensity and parameter ranges
- **Enable/disable flags**: Fine-grained control over noise components

## Quick Start

### 1. Setup Environment

```bash
# Create folder structure and default configurations
python organize_data.py setup --base-path ./my_mri_project

# This creates:
# my_mri_project/
# ├── raw_data/          # Place your DICOM/JPG files here
# ├── processed_data/    # Organized train/test data
# │   ├── train/
# │   └── test/
# ├── configs/           # Configuration files
# ├── models/            # Trained models
# └── results/           # Output results
```

### 2. Prepare Your Data

```bash
# Interactive data preparation
python organize_data.py interactive

# Or use command line
python organize_data.py prepare --source ./raw_data --target ./processed_data

# Or use configuration file
python organize_data.py prepare --config ./configs/data_prep_config.json
```

### 3. Train with Enhanced Dataset

Update your training configuration:

```json
{
  "datasets": {
    "train": {
      "dataset_type": "enhanced",
      "dataroot_H": "./processed_data/train",
      "use_enhanced_noise": true,
      "noise_config_path": "./configs/noise_config_default.json"
    }
  }
}
```

Then train normally:

```bash
python main_train_swinmr.py --opt ./options/SwinMR/example/train_swinmr_enhanced_dicom_jpg.json
```

## Configuration Files

### Noise Configuration

Three predefined noise levels are available:

- `noise_config_mild.json`: Light noise for high-quality data
- `noise_config_default.json`: Moderate noise for typical scenarios  
- `noise_config_aggressive.json`: Heavy noise for robust training

Example noise configuration:

```json
{
  "gaussian_noise": {
    "enabled": true,
    "weight": 0.8,
    "variance_range": [0.001, 0.05],
    "snr_range": [25, 45]
  },
  "line_noise": {
    "enabled": true,
    "weight": 0.5,
    "coverage_max": 0.25,
    "intensity_range": [0.1, 0.4],
    "direction": ["horizontal", "vertical"],
    "line_width_range": [1, 2]
  }
}
```

### Data Preparation Configuration

```json
{
  "source_dir": "/path/to/raw/data",
  "target_dir": "/path/to/processed/data",
  "train_ratio": 0.8,
  "organize_by_patient": true,
  "min_size": [64, 64],
  "max_size": [512, 512],
  "normalize": true
}
```

## Dataset Types

### Enhanced Dataset (`"dataset_type": "enhanced"`)

The new enhanced dataset class supports:

- **No sensitivity maps required**: Simplifies data requirements
- **Multi-format loading**: Automatic format detection and loading
- **Advanced noise simulation**: Configurable noise generation
- **Backward compatibility**: Works with existing SwinMR configurations

Key parameters:

```json
{
  "dataset_type": "enhanced",
  "use_enhanced_noise": true,
  "noise_config_path": "./configs/noise_config_default.json",
  "noise_types": null  // null = all enabled, or specify: ["gaussian_noise", "line_noise"]
}
```

## Noise Types and Parameters

### 1. Gaussian Noise
Simulates thermal and electronic noise in k-space.

```json
"gaussian_noise": {
  "enabled": true,
  "weight": 0.8,           // Probability of application
  "variance_range": [0.001, 0.05],  // Noise variance range
  "snr_range": [25, 45]    // Signal-to-noise ratio range (dB)
}
```

### 2. Rician Noise
Models the magnitude distribution of complex Gaussian noise (realistic for MRI).

```json
"rician_noise": {
  "enabled": true,
  "weight": 0.6,
  "sigma_range": [0.01, 0.03],      // Noise standard deviation
  "amplitude_range": [0.1, 0.25]    // Noise amplitude scaling
}
```

### 3. Ghosting Artifacts
Simulates motion-induced phase encoding errors.

```json
"ghosting": {
  "enabled": true,
  "weight": 0.4,
  "intensity_range": [0.1, 0.3],    // Ghost intensity relative to signal
  "offset_range": [5, 15],          // Spatial offset in pixels
  "direction": ["horizontal", "vertical", "both"]
}
```

### 4. Aliasing Artifacts
Simulates undersampling-induced wraparound artifacts.

```json
"aliasing": {
  "enabled": true,
  "weight": 0.3,
  "fold_factor_range": [2, 3],      // Undersampling factor
  "intensity_range": [0.2, 0.4]     // Alias intensity
}
```

### 5. Line Noise
Simulates RF interference and hardware artifacts.

```json
"line_noise": {
  "enabled": true,
  "weight": 0.5,
  "coverage_max": 0.25,             // Maximum 25% coverage
  "intensity_range": [0.1, 0.4],
  "direction": ["horizontal", "vertical"],
  "line_width_range": [1, 2]        // Line thickness in pixels
}
```

### 6. Zipper Artifacts
Simulates periodic k-space interference.

```json
"zipper_artifact": {
  "enabled": true,
  "weight": 0.3,
  "intensity_range": [0.2, 0.6],
  "frequency_range": [0.1, 0.25],   // Spatial frequency of artifact
  "direction": ["horizontal", "vertical"]
}
```

## File Organization

### Input Data Structure
Your raw data can be organized in any structure. The system will recursively find all supported files:

```
raw_data/
├── patient001/
│   ├── series001/
│   │   ├── image001.dcm
│   │   ├── image002.dcm
│   │   └── ...
│   └── series002/
├── patient002/
└── standalone_images/
    ├── scan001.jpg
    ├── scan002.png
    └── ...
```

### Output Data Structure
After preparation, data is organized as:

```
processed_data/
├── train/
│   ├── img_train_000001_patient001_series001_image001.npy
│   ├── img_train_000002_patient001_series001_image002.npy
│   └── ...
├── test/
│   ├── img_test_000001_patient002_series001_image001.npy
│   └── ...
└── metadata.json
```

## Integration with Existing Code

The enhanced dataset is designed to be a drop-in replacement for existing datasets:

1. **Minimal configuration changes**: Just change `dataset_type` to `"enhanced"`
2. **Backward compatibility**: Supports all existing parameters
3. **Gradual adoption**: Can be used alongside existing datasets

### Migration Example

Before:
```json
{
  "dataset_type": "ccsagnpi",
  "is_noise": true,
  "noise_level": 0.1
}
```

After:
```json
{
  "dataset_type": "enhanced",
  "use_enhanced_noise": true,
  "noise_config_path": "./configs/noise_config_default.json"
}
```

## Advanced Usage

### Custom Noise Configuration

Create custom noise profiles for specific scenarios:

```python
from utils.noise_generator import MRINoiseGenerator

# Create custom configuration
generator = MRINoiseGenerator()
config = generator.config

# Modify parameters
config['gaussian_noise']['weight'] = 0.9
config['line_noise']['enabled'] = False

# Save custom configuration
generator.save_config('./my_custom_noise.json')
```

### Programmatic Data Preparation

```python
from utils.data_preparer import DataPreparer

preparer = DataPreparer(
    source_dir='./raw_data',
    target_dir='./processed_data',
    train_ratio=0.85
)

metadata = preparer.prepare_data(
    organize_by_patient=True,
    min_size=(128, 128),
    max_size=(256, 256),
    normalize=True
)

print(f"Processed {metadata['total_images']} images")
```

### Selective Noise Application

Apply only specific noise types:

```json
{
  "noise_types": ["gaussian_noise", "line_noise"]  // Only these types
}
```

Or in code:
```python
noisy_image = generator.apply_noise(image, ["rician_noise", "ghosting"])
```

## Testing and Validation

### Test Noise Generator

```bash
python organize_data.py test-noise
```

### Validate Data Preparation

```python
# Check metadata
import json
with open('./processed_data/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Train/test split: {metadata['train_images']}/{metadata['test_images']}")
print(f"Size stats: {metadata['image_stats']}")
```

## Troubleshooting

### Common Issues

1. **DICOM loading fails**: Install pydicom: `pip install pydicom`
2. **Memory issues**: Reduce batch size or use mini dataset mode
3. **Path issues**: Use absolute paths in configuration files
4. **Size mismatch**: Check min/max size constraints in data preparation

### Debug Mode

Enable debugging with mini dataset:

```json
{
  "is_mini_dataset": true,
  "mini_dataset_prec": 0.1  // Use 10% of data
}
```

## Performance Considerations

- **Lazy loading**: Images are loaded on-demand during training
- **Caching**: Consider implementing image caching for repeated access
- **Parallel processing**: Use multiple dataloader workers for faster loading
- **Memory management**: Large datasets benefit from smaller batch sizes

## Requirements

- Python 3.7+
- NumPy
- OpenCV (cv2)
- SciPy
- Optional: pydicom (for DICOM support)
- Optional: Pillow (alternative image loading)