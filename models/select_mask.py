"""
# -----------------------------------------
Define Undersampling Mask
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
"""

import os
import scipy
import scipy.fftpack
from scipy.io import loadmat
import cv2
import numpy as np


def define_Mask(opt):
    mask_name = opt["mask"]

    # 256 * 256 Gaussian 1D
    if mask_name == "G1D10":
        mask = loadmat(
            os.path.join("mask", "Gaussian1D", "GaussianDistribution1DMask_10.mat")
        )["maskRS1"]
    elif mask_name == "G1D20":
        mask = loadmat(
            os.path.join("mask", "Gaussian1D", "GaussianDistribution1DMask_20.mat")
        )["maskRS1"]
    elif mask_name == "G1D30":
        mask = loadmat(
            os.path.join("mask", "Gaussian1D", "GaussianDistribution1DMask_30.mat")
        )["maskRS1"]
    elif mask_name == "G1D40":
        mask = loadmat(
            os.path.join("mask", "Gaussian1D", "GaussianDistribution1DMask_40.mat")
        )["maskRS1"]
    elif mask_name == "G1D50":
        mask = loadmat(
            os.path.join("mask", "Gaussian1D", "GaussianDistribution1DMask_50.mat")
        )["maskRS1"]

    # 256 * 256 Gaussian 2D
    elif mask_name == "G2D10":
        mask = loadmat(
            os.path.join("mask", "Gaussian2D", "GaussianDistribution2DMask_10.mat")
        )["maskRS2"]
    elif mask_name == "G2D20":
        mask = loadmat(
            os.path.join("mask", "Gaussian2D", "GaussianDistribution2DMask_20.mat")
        )["maskRS2"]
    elif mask_name == "G2D30":
        mask = loadmat(
            os.path.join("mask", "Gaussian2D", "GaussianDistribution2DMask_30.mat")
        )["maskRS2"]
    elif mask_name == "G2D40":
        mask = loadmat(
            os.path.join("mask", "Gaussian2D", "GaussianDistribution2DMask_40.mat")
        )["maskRS2"]
    elif mask_name == "G2D50":
        mask = loadmat(
            os.path.join("mask", "Gaussian2D", "GaussianDistribution2DMask_50.mat")
        )["maskRS2"]

    # 256 * 256 poisson 2D
    elif mask_name == "P2D10":
        mask = loadmat(
            os.path.join("mask", "Poisson2D", "PoissonDistributionMask_10.mat")
        )["population_matrix"]
    elif mask_name == "P2D20":
        mask = loadmat(
            os.path.join("mask", "Poisson2D", "PoissonDistributionMask_20.mat")
        )["population_matrix"]
    elif mask_name == "P2D30":
        mask = loadmat(
            os.path.join("mask", "Poisson2D", "PoissonDistributionMask_30.mat")
        )["population_matrix"]
    elif mask_name == "P2D40":
        mask = loadmat(
            os.path.join("mask", "Poisson2D", "PoissonDistributionMask_40.mat")
        )["population_matrix"]
    elif mask_name == "P2D50":
        mask = loadmat(
            os.path.join("mask", "Poisson2D", "PoissonDistributionMask_50.mat")
        )["population_matrix"]

    # 256 * 256 radial
    elif mask_name == "R10":
        mask_shift = (
            cv2.imread(os.path.join("mask", "radial", "radial_10.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "R20":
        mask_shift = (
            cv2.imread(os.path.join("mask", "radial", "radial_20.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "R30":
        mask_shift = (
            cv2.imread(os.path.join("mask", "radial", "radial_30.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "R40":
        mask_shift = (
            cv2.imread(os.path.join("mask", "radial", "radial_40.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "R50":
        mask_shift = (
            cv2.imread(os.path.join("mask", "radial", "radial_50.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "R60":
        mask_shift = (
            cv2.imread(os.path.join("mask", "radial", "radial_60.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "R70":
        mask_shift = (
            cv2.imread(os.path.join("mask", "radial", "radial_70.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "R80":
        mask_shift = (
            cv2.imread(os.path.join("mask", "radial", "radial_80.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "R90":
        mask_shift = (
            cv2.imread(os.path.join("mask", "radial", "radial_90.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)

    # 256 * 256 spiral
    elif mask_name == "S10":
        mask_shift = (
            cv2.imread(os.path.join("mask", "spiral", "spiral_10.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "S20":
        mask_shift = (
            cv2.imread(os.path.join("mask", "spiral", "spiral_20.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "S30":
        mask_shift = (
            cv2.imread(os.path.join("mask", "spiral", "spiral_30.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "S40":
        mask_shift = (
            cv2.imread(os.path.join("mask", "spiral", "spiral_40.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "S50":
        mask_shift = (
            cv2.imread(os.path.join("mask", "spiral", "spiral_50.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "S60":
        mask_shift = (
            cv2.imread(os.path.join("mask", "spiral", "spiral_60.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "S70":
        mask_shift = (
            cv2.imread(os.path.join("mask", "spiral", "spiral_70.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "S80":
        mask_shift = (
            cv2.imread(os.path.join("mask", "spiral", "spiral_80.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)
    elif mask_name == "S90":
        mask_shift = (
            cv2.imread(os.path.join("mask", "spiral", "spiral_90.tif"), 0) / 255
        )
        mask = scipy.fftpack.fftshift(mask_shift)

    else:
        raise NotImplementedError("Model [{:s}] is not defined.".format(mask_name))

    # Removed verbose print statement to reduce console spam during training
    # print("Training model [{:s}] is created.".format(mask_name))

    return mask


def get_all_available_masks():
    """Return list of all available mask types."""
    return [
        # Gaussian 1D
        "G1D10",
        "G1D20",
        "G1D30",
        "G1D40",
        "G1D50",
        # Gaussian 2D
        "G2D10",
        "G2D20",
        "G2D30",
        "G2D40",
        "G2D50",
        # Radial
        "R10",
        "R20",
        "R30",
        "R40",
        "R50",
        "R60",
        "R70",
        "R80",
        "R90",
        # Spiral
        "S10",
        "S20",
        "S30",
        "S40",
        "S50",
        "S60",
        "S70",
        "S80",
        "S90",
    ]


def get_random_mask(mask_list=None):
    """
    Get a random mask from available masks.

    Args:
        mask_list: List of mask names to choose from. If None, uses all available masks.

    Returns:
        Loaded mask array
    """
    import random

    if mask_list is None:
        mask_list = get_all_available_masks()

    # Randomly select mask name
    selected_mask_name = random.choice(mask_list)

    # Create temporary opt dict and load mask
    temp_opt = {"mask": selected_mask_name}
    mask = define_Mask(temp_opt)

    print(f"Randomly selected mask: {selected_mask_name}")

    return mask, selected_mask_name


def define_Mask_with_random(opt):
    """
    Enhanced mask definition that supports random selection.

    Args:
        opt: Options dict that may contain 'random_mask_selection' and 'available_masks'

    Returns:
        Loaded mask array
    """
    if opt.get("random_mask_selection", False):
        available_masks = opt.get("available_masks", get_all_available_masks())
        mask, mask_name = get_random_mask(available_masks)
        return mask
    else:
        return define_Mask(opt)
