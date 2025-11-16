"""Tests for utility functions."""

import pytest
import cv2
import os
import numpy as np
from src.utils import load_image_grayscale, collect_image_paths


def test_load_image_grayscale_success(temp_dir, sample_image_gradient):
    """Test successfully loading a valid grayscale image."""
    # Save a test image
    image_path = os.path.join(temp_dir, "test_image.jpg")
    cv2.imwrite(image_path, sample_image_gradient)

    # Load the image
    img = load_image_grayscale(image_path)

    # Verify it was loaded correctly
    assert img is not None
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 2  # Grayscale image has 2 dimensions
    assert img.shape == sample_image_gradient.shape


def test_load_image_grayscale_invalid_path(capsys):
    """Test loading an image from an invalid/non-existent path."""
    invalid_path = "/nonexistent/path/to/image.jpg"

    # Should return None and print a warning
    img = load_image_grayscale(invalid_path)

    assert img is None

    # Verify warning message was printed
    captured = capsys.readouterr()
    assert "Warning: Could not read" in captured.out
    assert invalid_path in captured.out


def test_load_image_grayscale_corrupted_file(temp_dir, capsys):
    """Test loading a corrupted/invalid image file."""
    # Create a file that's not a valid image
    corrupted_path = os.path.join(temp_dir, "corrupted.jpg")
    with open(corrupted_path, "w") as f:
        f.write("This is not a valid image file")

    # Should return None and print a warning
    img = load_image_grayscale(corrupted_path)

    assert img is None

    # Verify warning message was printed
    captured = capsys.readouterr()
    assert "Warning: Could not read" in captured.out
    assert corrupted_path in captured.out


def test_collect_image_paths_success(temp_dir, sample_image_gradient, sample_image_checkerboard):
    """Test successfully collecting image paths with a valid pattern."""
    # Create multiple test images
    image1_path = os.path.join(temp_dir, "image_1.jpg")
    image2_path = os.path.join(temp_dir, "image_2.jpg")
    image3_path = os.path.join(temp_dir, "image_3.jpg")

    cv2.imwrite(image1_path, sample_image_gradient)
    cv2.imwrite(image2_path, sample_image_checkerboard)
    cv2.imwrite(image3_path, sample_image_gradient)

    # Collect images using glob pattern
    pattern = os.path.join(temp_dir, "*.jpg")
    paths = collect_image_paths(pattern)

    # Verify all images were found
    assert len(paths) == 3
    assert image1_path in paths
    assert image2_path in paths
    assert image3_path in paths


def test_collect_image_paths_sorting(temp_dir, sample_image_gradient):
    """Test that collected image paths are sorted."""
    # Create images with names that would sort differently
    image_c = os.path.join(temp_dir, "c_image.jpg")
    image_a = os.path.join(temp_dir, "a_image.jpg")
    image_b = os.path.join(temp_dir, "b_image.jpg")

    cv2.imwrite(image_c, sample_image_gradient)
    cv2.imwrite(image_a, sample_image_gradient)
    cv2.imwrite(image_b, sample_image_gradient)

    # Collect images
    pattern = os.path.join(temp_dir, "*_image.jpg")
    paths = collect_image_paths(pattern)

    # Verify they are sorted
    assert len(paths) == 3
    assert paths[0] == image_a
    assert paths[1] == image_b
    assert paths[2] == image_c


def test_collect_image_paths_no_matches_raises_error(temp_dir):
    """Test that ValueError is raised when no images match the pattern."""
    # Use a pattern that won't match any files
    pattern = os.path.join(temp_dir, "nonexistent_*.jpg")

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        collect_image_paths(pattern)

    assert "No images found at path" in str(exc_info.value)
    assert pattern in str(exc_info.value)


def test_collect_image_paths_single_image(temp_dir, sample_image_gradient):
    """Test collecting a single image."""
    image_path = os.path.join(temp_dir, "single.jpg")
    cv2.imwrite(image_path, sample_image_gradient)

    # Collect with specific pattern
    pattern = os.path.join(temp_dir, "single.jpg")
    paths = collect_image_paths(pattern)

    # Verify single image was found
    assert len(paths) == 1
    assert paths[0] == image_path
