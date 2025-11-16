"""Utility functions for image loading and path handling."""

import cv2
import glob as glob_module
from typing import List, Optional
import numpy as np


def load_image_grayscale(path: str) -> Optional[np.ndarray]:
    """
    Load an image in grayscale mode.

    Args:
        path: Path to the image file

    Returns:
        The grayscale image as a numpy array, or None if loading failed
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read {path}. Skipping.")
    return img


def collect_image_paths(glob_pattern: str) -> List[str]:
    """
    Collect image paths matching a glob pattern.

    Args:
        glob_pattern: Glob-style pattern (e.g., "images/*.jpg")

    Returns:
        Sorted list of image paths

    Raises:
        ValueError: If no images are found matching the pattern
    """
    image_paths = glob_module.glob(glob_pattern)
    if not image_paths:
        raise ValueError(f"No images found at path: {glob_pattern}")

    image_paths.sort()
    return image_paths
