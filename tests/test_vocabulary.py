"""Tests for vocabulary building functions."""

import cv2
import numpy as np
import pytest
import tempfile
import os

from src.vocabulary import build_vocabulary, create_bow_extractor
from src.features import create_sift_detector, create_matcher


def test_create_bow_extractor(small_vocabulary, sift_detector, matcher):
    """Test BOW extractor creation."""
    bow_extractor = create_bow_extractor(small_vocabulary, sift_detector, matcher)

    assert isinstance(bow_extractor, cv2.BOWImgDescriptorExtractor)


def test_build_vocabulary_small_dataset(temp_dir, sift_detector):
    """Test vocabulary building with a small synthetic dataset."""
    # Create 3 test images in temp directory
    image_paths = []
    for i in range(3):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # Add some structure to ensure features are detected
        cv2.rectangle(img, (20 + i*5, 20), (80, 80), 255, 2)

        path = os.path.join(temp_dir, f"test_image_{i}.png")
        cv2.imwrite(path, img)
        image_paths.append(path)

    # Build vocabulary with k=5
    vocabulary = build_vocabulary(image_paths, k=5, sift=sift_detector, limit=None)

    # Vocabulary should be k x 128 (SIFT descriptor dimension)
    assert vocabulary.shape == (5, 128)
    assert vocabulary.dtype == np.float32


def test_build_vocabulary_with_limit(temp_dir, sift_detector):
    """Test vocabulary building with a limit on training images."""
    # Create 5 test images
    image_paths = []
    for i in range(5):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20 + i*5, 255, 2)

        path = os.path.join(temp_dir, f"test_image_{i}.png")
        cv2.imwrite(path, img)
        image_paths.append(path)

    # Build vocabulary with limit=3 (only use first 3 images)
    vocabulary = build_vocabulary(image_paths, k=5, sift=sift_detector, limit=3)

    assert vocabulary.shape == (5, 128)


def test_build_vocabulary_no_features_raises_error(temp_dir, sift_detector):
    """Test that building vocabulary with no features raises an error."""
    # Create completely black images (no features)
    image_paths = []
    for i in range(2):
        img = np.zeros((100, 100), dtype=np.uint8)
        path = os.path.join(temp_dir, f"black_image_{i}.png")
        cv2.imwrite(path, img)
        image_paths.append(path)

    # Should raise ValueError when no descriptors are found
    with pytest.raises(ValueError, match="No descriptors found"):
        build_vocabulary(image_paths, k=5, sift=sift_detector)


def test_build_vocabulary_with_invalid_images(temp_dir, sift_detector):
    """Test vocabulary building with a mix of valid and invalid images."""
    # Create valid images
    image_paths = []
    for i in range(3):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        cv2.rectangle(img, (20 + i*5, 20), (80, 80), 255, 2)
        path = os.path.join(temp_dir, f"valid_image_{i}.png")
        cv2.imwrite(path, img)
        image_paths.append(path)

    # Create invalid image file
    invalid_path = os.path.join(temp_dir, "invalid.jpg")
    with open(invalid_path, "w") as f:
        f.write("not a valid image")
    image_paths.insert(1, invalid_path)  # Insert in the middle

    # Build vocabulary - should skip invalid image and succeed
    vocabulary = build_vocabulary(image_paths, k=5, sift=sift_detector, limit=None)

    # Vocabulary should still be built from valid images
    assert vocabulary.shape == (5, 128)
    assert vocabulary.dtype == np.float32
