"""
# -----------------------------------------
Data Preparation Utility for DICOM/JPG Images
by SwinMR Enhancement
# -----------------------------------------
"""

import os
import json
import shutil
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import cv2

# Import DICOM support if available
try:
    import pydicom
    DICOM_SUPPORT = True
except ImportError:
    DICOM_SUPPORT = False
    print("Warning: pydicom not available. DICOM support disabled.")


class DataPreparer:
    """
    Utility class to prepare and organize MRI data from various formats
    into standardized train/test folder structures.
    """
    
    def __init__(self, source_dir: str, target_dir: str, train_ratio: float = 0.8):
        """
        Initialize the data preparer.
        
        Args:
            source_dir: Source directory containing raw data
            target_dir: Target directory for organized data
            train_ratio: Ratio of data to use for training (0.0-1.0)
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.train_ratio = train_ratio
        
        # Supported file extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if DICOM_SUPPORT:
            self.supported_extensions.update({'.dcm', '.dicom'})
        
        # Create target directories
        self.train_dir = self.target_dir / 'train'
        self.test_dir = self.target_dir / 'test'
        
    def prepare_data(self, 
                    organize_by_patient: bool = True,
                    min_size: Tuple[int, int] = (64, 64),
                    max_size: Tuple[int, int] = (512, 512),
                    normalize: bool = True) -> dict:
        """
        Prepare and organize data from source to target directories.
        
        Args:
            organize_by_patient: Whether to organize by patient/series
            min_size: Minimum image size (height, width)
            max_size: Maximum image size (height, width)
            normalize: Whether to normalize images to [0, 1]
            
        Returns:
            Dictionary with preparation statistics
        """
        print(f"Preparing data from {self.source_dir} to {self.target_dir}")
        
        # Create target directories
        self._create_directories()
        
        # Find all supported files
        image_files = self._find_image_files()
        print(f"Found {len(image_files)} supported image files")
        
        if len(image_files) == 0:
            raise ValueError("No supported image files found in source directory")
        
        # Process and filter images
        valid_images = []
        processed_count = 0
        
        for file_path in image_files:
            try:
                image_data = self._load_image(file_path)
                if image_data is not None:
                    h, w = image_data.shape[:2]
                    
                    # Check size constraints
                    if (h >= min_size[0] and w >= min_size[1] and 
                        h <= max_size[0] and w <= max_size[1]):
                        
                        # Process image
                        if normalize:
                            image_data = self._normalize_image(image_data)
                        
                        valid_images.append({
                            'path': file_path,
                            'data': image_data,
                            'patient_id': self._extract_patient_id(file_path) if organize_by_patient else None
                        })
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} images...")
                            
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(valid_images)} valid images")
        
        # Split into train/test
        train_images, test_images = self._split_data(valid_images, organize_by_patient)
        
        # Save images
        train_stats = self._save_images(train_images, self.train_dir, 'train')
        test_stats = self._save_images(test_images, self.test_dir, 'test')
        
        # Save metadata
        metadata = {
            'total_images': len(valid_images),
            'train_images': len(train_images),
            'test_images': len(test_images),
            'train_ratio': len(train_images) / len(valid_images),
            'image_stats': {
                'train': train_stats,
                'test': test_stats
            },
            'source_dir': str(self.source_dir),
            'target_dir': str(self.target_dir),
            'parameters': {
                'organize_by_patient': organize_by_patient,
                'min_size': min_size,
                'max_size': max_size,
                'normalize': normalize
            }
        }
        
        self._save_metadata(metadata)
        
        print(f"Data preparation completed:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Test: {len(test_images)} images")
        print(f"  Metadata saved to: {self.target_dir / 'metadata.json'}")
        
        return metadata
    
    def _create_directories(self):
        """Create target directory structure."""
        for dir_path in [self.train_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _find_image_files(self) -> List[Path]:
        """Find all supported image files in source directory."""
        image_files = []
        
        for ext in self.supported_extensions:
            # Search recursively for files with supported extensions
            pattern = f"**/*{ext}"
            files = list(self.source_dir.glob(pattern))
            image_files.extend(files)
            
            # Also search for uppercase extensions
            pattern = f"**/*{ext.upper()}"
            files = list(self.source_dir.glob(pattern))
            image_files.extend(files)
        
        return sorted(list(set(image_files)))  # Remove duplicates and sort
    
    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        """Load image from file (DICOM or standard image format)."""
        ext = file_path.suffix.lower()
        
        try:
            if ext in {'.dcm', '.dicom'} and DICOM_SUPPORT:
                return self._load_dicom(file_path)
            else:
                return self._load_standard_image(file_path)
        except Exception as e:
            print(f"Failed to load {file_path}: {str(e)}")
            return None
    
    def _load_dicom(self, file_path: Path) -> Optional[np.ndarray]:
        """Load DICOM image."""
        if not DICOM_SUPPORT:
            return None
        
        try:
            ds = pydicom.dcmread(str(file_path))
            
            # Get pixel data
            if hasattr(ds, 'pixel_array'):
                image = ds.pixel_array.astype(np.float32)
                
                # Handle different DICOM image types
                if len(image.shape) == 3:
                    # Multi-slice or RGB
                    if image.shape[2] == 3:
                        # RGB - convert to grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    else:
                        # Multi-slice - take middle slice
                        image = image[image.shape[0] // 2]
                
                return image
            else:
                print(f"No pixel data found in DICOM file: {file_path}")
                return None
                
        except Exception as e:
            print(f"Error reading DICOM file {file_path}: {str(e)}")
            return None
    
    def _load_standard_image(self, file_path: Path) -> Optional[np.ndarray]:
        """Load standard image format (JPG, PNG, etc.)."""
        try:
            # Load image
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                # Try with PIL as fallback
                from PIL import Image
                pil_image = Image.open(file_path).convert('L')
                image = np.array(pil_image)
            
            return image.astype(np.float32) if image is not None else None
            
        except Exception as e:
            print(f"Error reading image file {file_path}: {str(e)}")
            return None
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        if image.max() > image.min():
            return (image - image.min()) / (image.max() - image.min())
        else:
            return np.zeros_like(image)
    
    def _extract_patient_id(self, file_path: Path) -> str:
        """Extract patient/series ID from file path."""
        # Simple heuristic: use parent directory name or file prefix
        parent_name = file_path.parent.name
        
        # Look for patient ID patterns
        if 'patient' in parent_name.lower() or 'subj' in parent_name.lower():
            return parent_name
        
        # Use file name prefix (before first underscore or number)
        filename = file_path.stem
        for i, char in enumerate(filename):
            if char.isdigit() or char == '_':
                if i > 0:
                    return filename[:i]
                break
        
        return parent_name
    
    def _split_data(self, images: List[dict], organize_by_patient: bool) -> Tuple[List[dict], List[dict]]:
        """Split data into train and test sets."""
        if organize_by_patient:
            # Group by patient ID
            patient_groups = {}
            for img in images:
                patient_id = img['patient_id'] or 'unknown'
                if patient_id not in patient_groups:
                    patient_groups[patient_id] = []
                patient_groups[patient_id].append(img)
            
            # Split patients
            patient_ids = list(patient_groups.keys())
            random.shuffle(patient_ids)
            
            split_idx = int(len(patient_ids) * self.train_ratio)
            train_patients = patient_ids[:split_idx]
            test_patients = patient_ids[split_idx:]
            
            # Collect images
            train_images = []
            test_images = []
            
            for patient_id in train_patients:
                train_images.extend(patient_groups[patient_id])
            
            for patient_id in test_patients:
                test_images.extend(patient_groups[patient_id])
                
        else:
            # Simple random split
            random.shuffle(images)
            split_idx = int(len(images) * self.train_ratio)
            train_images = images[:split_idx]
            test_images = images[split_idx:]
        
        return train_images, test_images
    
    def _save_images(self, images: List[dict], target_dir: Path, split_name: str) -> dict:
        """Save images to target directory."""
        stats = {
            'count': len(images),
            'sizes': [],
            'mean_intensity': [],
            'std_intensity': []
        }
        
        for i, img_info in enumerate(images):
            # Generate filename
            original_name = img_info['path'].stem
            filename = f"img_{split_name}_{i:06d}_{original_name}.npy"
            target_path = target_dir / filename
            
            # Save as numpy array
            image_data = img_info['data']
            np.save(target_path, image_data)
            
            # Collect stats
            stats['sizes'].append(image_data.shape)
            stats['mean_intensity'].append(float(np.mean(image_data)))
            stats['std_intensity'].append(float(np.std(image_data)))
        
        # Compute summary statistics
        if stats['sizes']:
            heights = [s[0] for s in stats['sizes']]
            widths = [s[1] for s in stats['sizes']]
            
            stats['size_stats'] = {
                'height': {'min': min(heights), 'max': max(heights), 'mean': np.mean(heights)},
                'width': {'min': min(widths), 'max': max(widths), 'mean': np.mean(widths)}
            }
            stats['intensity_stats'] = {
                'mean': {'min': min(stats['mean_intensity']), 'max': max(stats['mean_intensity']), 
                        'mean': np.mean(stats['mean_intensity'])},
                'std': {'min': min(stats['std_intensity']), 'max': max(stats['std_intensity']), 
                       'mean': np.mean(stats['std_intensity'])}
            }
        
        return stats
    
    def _save_metadata(self, metadata: dict):
        """Save metadata to JSON file."""
        metadata_path = self.target_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def prepare_data_from_config(config_path: str):
    """Prepare data using configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    preparer = DataPreparer(
        source_dir=config['source_dir'],
        target_dir=config['target_dir'],
        train_ratio=config.get('train_ratio', 0.8)
    )
    
    return preparer.prepare_data(
        organize_by_patient=config.get('organize_by_patient', True),
        min_size=tuple(config.get('min_size', [64, 64])),
        max_size=tuple(config.get('max_size', [512, 512])),
        normalize=config.get('normalize', True)
    )


def create_data_prep_config(output_path: str):
    """Create a default data preparation configuration file."""
    config = {
        "source_dir": "/path/to/source/data",
        "target_dir": "/path/to/target/data", 
        "train_ratio": 0.8,
        "organize_by_patient": True,
        "min_size": [64, 64],
        "max_size": [512, 512],
        "normalize": True
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Data preparation configuration template saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare MRI data for training')
    parser.add_argument('--source', type=str, required=True, help='Source directory')
    parser.add_argument('--target', type=str, required=True, help='Target directory')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.config:
        prepare_data_from_config(args.config)
    else:
        preparer = DataPreparer(args.source, args.target, args.train_ratio)
        preparer.prepare_data()