#!/usr/bin/env python3
"""
Test script to verify optimal worker configuration for different platforms.
"""

import platform
import multiprocessing
import os
import sys
sys.path.append('.')

def get_optimal_workers(requested_workers, distributed=False, world_size=1):
    """
    Determine optimal number of workers based on platform and configuration.
    """
    # Get system info
    max_cpu_workers = multiprocessing.cpu_count()
    platform_name = platform.system()
    
    if distributed:
        # For distributed training, divide workers among processes
        base_workers = min(requested_workers // max(world_size, 1), 4)
    else:
        base_workers = min(requested_workers, 4)
    
    # Platform-specific adjustments
    if platform_name == "Windows":
        # Windows has issues with multiprocessing in PyTorch
        optimal_workers = min(base_workers, 2)
        if optimal_workers > 0:
            print(f"[INFO] Windows detected - using {optimal_workers} workers (max recommended: 2)")
            print(f"[INFO] If you encounter multiprocessing errors, set dataloader_num_workers to 0")
    elif platform_name == "Darwin":  # macOS
        # macOS generally handles multiprocessing well but be conservative
        optimal_workers = min(base_workers, max_cpu_workers // 2)
    else:  # Linux and others
        optimal_workers = min(base_workers, max_cpu_workers)
    
    # Ensure we don't exceed system capabilities
    optimal_workers = max(0, min(optimal_workers, max_cpu_workers))
    
    return optimal_workers


def test_worker_configurations():
    """Test different worker configurations."""
    print("=" * 60)
    print("WORKER CONFIGURATION TEST")
    print("=" * 60)
    
    # System info
    print(f"Platform         : {platform.system()}")
    print(f"Platform Release : {platform.release()}")
    print(f"CPU Count        : {multiprocessing.cpu_count()}")
    print(f"Python Version   : {platform.python_version()}")
    
    print("\n" + "-" * 60)
    print("TESTING DIFFERENT CONFIGURATIONS")
    print("-" * 60)
    
    test_cases = [
        (0, False, 1, "No workers (safest)"),
        (1, False, 1, "Single worker"),
        (2, False, 1, "Two workers"),
        (4, False, 1, "Four workers"),
        (8, False, 1, "Eight workers (high)"),
        (4, True, 2, "Distributed (2 GPUs)"),
        (8, True, 4, "Distributed (4 GPUs)"),
    ]
    
    for requested, distributed, world_size, description in test_cases:
        optimal = get_optimal_workers(requested, distributed, world_size)
        print(f"{description:25} : {requested:2d} → {optimal:2d} workers")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    platform_name = platform.system()
    if platform_name == "Windows":
        print("• Windows: Use 0-2 workers maximum")
        print("• If you get multiprocessing errors, set workers to 0")
        print("• Consider upgrading to WSL2 for better performance")
    elif platform_name == "Darwin":
        print("• macOS: Use up to half of your CPU cores")
        print("• Generally stable with multiprocessing")
    else:
        print("• Linux: Can use more workers efficiently")
        print("• Monitor memory usage with high worker counts")
    
    print(f"\nFor your system: Recommended max workers = {get_optimal_workers(16, False)}")


if __name__ == "__main__":
    test_worker_configurations()