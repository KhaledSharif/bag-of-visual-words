"""Feature extraction using SIFT (Scale-Invariant Feature Transform)."""

import cv2
import numpy as np
from typing import Tuple, Optional, List


def create_sift_detector() -> cv2.SIFT:
    """
    Create a SIFT feature detector.

    Returns:
        Configured SIFT detector instance
    """
    return cv2.SIFT_create()


def create_matcher() -> cv2.BFMatcher:
    """
    Create a Brute-Force matcher for SIFT descriptors.

    Returns:
        BFMatcher configured with L2 norm (Euclidean distance)
    """
    return cv2.BFMatcher(cv2.NORM_L2)


def extract_keypoints(image: np.ndarray, sift: cv2.SIFT) -> List:
    """
    Detect SIFT keypoints in an image.

    Args:
        image: Grayscale image as numpy array
        sift: SIFT detector instance

    Returns:
        List of detected keypoints
    """
    keypoints = sift.detect(image, None)
    return list(keypoints)


def extract_descriptors(image: np.ndarray, sift: cv2.SIFT) -> Tuple[List, Optional[np.ndarray]]:
    """
    Detect SIFT keypoints and compute their descriptors.

    Args:
        image: Grayscale image as numpy array
        sift: SIFT detector instance

    Returns:
        Tuple of (keypoints, descriptors). Descriptors will be None if no keypoints found.
    """
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return list(keypoints), descriptors
