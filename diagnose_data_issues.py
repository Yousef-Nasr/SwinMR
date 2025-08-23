#!/usr/bin/env python3
"""
Data Loading Diagnostic Script for SwinMR
This script helps diagnose NaN/Inf and DataLoader worker issues.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_CCsagnpi import DatasetCCsagnpi, safe_collate_fn, validate_data_files
from utils import utils_option as option


def test_data_loading(config_path, max_samples=10):
    """
    Test data loading with various worker configurations.
    """
    print("=" * 60)
    print("SwinMR Data Loading Diagnostic Tool")
    print("=" * 60)
    
    # Load configuration
    opt = option.parse(config_path, is_train=True)
    dataset_opt = opt["datasets"]["train"]
    
    print(f"\nDataset configuration:")
    print(f"  dataroot_H: {dataset_opt['dataroot_H']}")
    print(f"  mask: {dataset_opt['mask']}")
    print(f"  batch_size: {dataset_opt['dataloader_batch_size']}")
    print(f"  num_workers: {dataset_opt['dataloader_num_workers']}")
    
    # Test dataset creation
    print(f"\n1. Testing dataset creation...")
    try:
        dataset = DatasetCCsagnpi(dataset_opt)
        print(f"   ✓ Dataset created successfully")
        print(f"   ✓ Found {len(dataset)} samples")
        
        # Validate some data files
        print(f"\n2. Validating data files...")
        if hasattr(dataset, 'paths_H') and len(dataset.paths_H) > 0:
            issues = validate_data_files(dataset.paths_H, max_check=min(5, len(dataset.paths_H)))
            if not issues:
                print(f"   ✓ Data validation passed")
            else:
                print(f"   ⚠ Found {len(issues)} issues in data files")
        
    except Exception as e:
        print(f"   ✗ Dataset creation failed: {e}")
        return False
    
    # Test different worker configurations
    worker_configs = [0, 1, 2, 4] if os.name != 'nt' else [0]  # Windows only supports 0 workers reliably
    
    for num_workers in worker_configs:
        print(f"\n3. Testing DataLoader with {num_workers} workers...")
        
        try:
            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=min(2, dataset_opt['dataloader_batch_size']),
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=False,
                collate_fn=safe_collate_fn,
                persistent_workers=num_workers > 0,
            )
            
            # Test loading first few batches
            success_count = 0
            error_count = 0
            
            for i, batch in enumerate(dataloader):
                if i >= max_samples:
                    break
                    
                if batch is None:
                    error_count += 1
                    print(f"   ⚠ Batch {i}: None returned")
                    continue
                
                # Check for NaN/Inf in batch
                nan_inf_found = False
                if 'L' in batch and batch['L'] is not None:
                    if torch.any(torch.isnan(batch['L'])) or torch.any(torch.isinf(batch['L'])):
                        nan_inf_found = True
                        print(f"   ⚠ Batch {i}: NaN/Inf in L tensor")
                
                if 'H' in batch and batch['H'] is not None:
                    if torch.any(torch.isnan(batch['H'])) or torch.any(torch.isinf(batch['H'])):
                        nan_inf_found = True
                        print(f"   ⚠ Batch {i}: NaN/Inf in H tensor")
                
                if not nan_inf_found:
                    success_count += 1
                else:
                    error_count += 1
            
            print(f"   ✓ Workers={num_workers}: {success_count} good batches, {error_count} problematic batches")
            
        except Exception as e:
            print(f"   ✗ Workers={num_workers}: Failed with error: {e}")
    
    print(f"\n4. Testing single sample loading...")
    try:
        # Test loading individual samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            
            if sample is None:
                print(f"   ⚠ Sample {i}: None returned")
                continue
            
            # Check sample contents
            issues = []
            if 'L' in sample:
                if torch.any(torch.isnan(sample['L'])) or torch.any(torch.isinf(sample['L'])):
                    issues.append("NaN/Inf in L")
            if 'H' in sample:
                if torch.any(torch.isnan(sample['H'])) or torch.any(torch.isinf(sample['H'])):
                    issues.append("NaN/Inf in H")
            
            if issues:
                print(f"   ⚠ Sample {i}: {', '.join(issues)}")
            else:
                print(f"   ✓ Sample {i}: OK")
        
    except Exception as e:
        print(f"   ✗ Single sample loading failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("Diagnostic complete!")
    print("\nRecommendations:")
    print("1. If you see NaN/Inf warnings, check your data files for corruption")
    print("2. If DataLoader fails with workers > 0, set 'dataloader_num_workers': 0 in your config")
    print("3. On Windows, always use 'dataloader_num_workers': 0")
    print("4. Use the safe_collate_fn in your training script")
    
    return True


def create_safe_config(base_config_path, output_path):
    """
    Create a safe configuration file with recommended settings.
    """
    print(f"\nCreating safe configuration...")
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Apply safe settings
    if 'datasets' in config and 'train' in config['datasets']:
        config['datasets']['train']['dataloader_num_workers'] = 0
        print(f"   ✓ Set dataloader_num_workers to 0")
    
    if 'datasets' in config and 'test' in config['datasets']:
        config['datasets']['test']['dataloader_num_workers'] = 0
    
    # Save safe config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"   ✓ Safe configuration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose SwinMR data loading issues')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file')
    parser.add_argument('--max-samples', type=int, default=10,
                        help='Maximum number of samples to test')
    parser.add_argument('--create-safe-config', action='store_true',
                        help='Create a safe configuration file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    # Test data loading
    success = test_data_loading(args.config, args.max_samples)
    
    # Create safe config if requested
    if args.create_safe_config:
        safe_config_path = args.config.replace('.json', '_safe.json')
        create_safe_config(args.config, safe_config_path)


if __name__ == "__main__":
    main()
