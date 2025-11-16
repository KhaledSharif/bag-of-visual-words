"""Tests for feature extraction functions."""

import cv2
import numpy as np
from src.features import (
    create_sift_detector,
    create_matcher,
    extract_keypoints,
    extract_descriptors
)


def test_create_sift_detector():
    """Test SIFT detector creation."""
    sift = create_sift_detector()
    assert isinstance(sift, cv2.SIFT)


def test_create_matcher():
    """Test BFMatcher creation."""
    matcher = create_matcher()
    assert isinstance(matcher, cv2.BFMatcher)


def test_extract_keypoints(sample_image_checkerboard, sift_detector):
    """Test keypoint extraction returns expected format."""
    keypoints = extract_keypoints(sample_image_checkerboard, sift_detector)

    assert isinstance(keypoints, list)
    assert len(keypoints) > 0
    assert all(isinstance(kp, cv2.KeyPoint) for kp in keypoints)


def test_extract_descriptors(sample_image_checkerboard, sift_detector):
    """Test descriptor extraction."""
    keypoints, descriptors = extract_descriptors(sample_image_checkerboard, sift_detector)

    assert isinstance(keypoints, list)
    assert len(keypoints) > 0

    # Descriptors should be a numpy array with shape (n_keypoints, 128)
    assert descriptors is not None
    assert isinstance(descriptors, np.ndarray)
    assert descriptors.shape[0] == len(keypoints)
    assert descriptors.shape[1] == 128  # SIFT descriptors are 128-dimensional


def test_extract_keypoints_empty_image(sift_detector):
    """Test handling of empty/blank images."""
    # Create a completely black image (no features)
    empty_img = np.zeros((100, 100), dtype=np.uint8)
    keypoints = extract_keypoints(empty_img, sift_detector)

    # Black images typically have no keypoints
    assert isinstance(keypoints, list)
