"""Pytest fixtures for BoVW tests."""

import pytest
import numpy as np
import tempfile
import os
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse

from src.features import create_sift_detector, create_matcher
from src.database import ImageDatabase


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sift_detector():
    """Create a SIFT detector instance."""
    return create_sift_detector()


@pytest.fixture
def matcher():
    """Create a BFMatcher instance."""
    return create_matcher()


@pytest.fixture
def sample_image_gradient():
    """Create a simple synthetic test image (horizontal gradient)."""
    # Create a 100x100 grayscale image with a horizontal gradient
    img = np.linspace(0, 255, 10000, dtype=np.uint8).reshape(100, 100)
    return img


@pytest.fixture
def sample_image_checkerboard():
    """Create a checkerboard pattern test image."""
    # Create a 100x100 checkerboard pattern
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i // 10 + j // 10) % 2 == 0:
                img[i:i+10, j:j+10] = 255
    return img


@pytest.fixture
def sample_image_circle():
    """Create a test image with a white circle on black background."""
    import cv2
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, 255, -1)
    return img


@pytest.fixture
def small_vocabulary():
    """Create a small mock vocabulary for testing (k=10)."""
    # Create 10 random 128-dimensional vectors (SIFT descriptor size)
    np.random.seed(42)
    vocabulary = np.random.rand(10, 128).astype(np.float32)
    return vocabulary


@pytest.fixture
def mock_database(small_vocabulary):
    """Create a mock ImageDatabase for testing."""
    np.random.seed(42)

    # Create mock TF-IDF vectors for 5 images
    # Each image has a histogram over 10 visual words
    tf_vectors = np.random.rand(5, 10)

    # Fit a TF-IDF transformer
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(tf_vectors)
    tfidf_vectors = tfidf_transformer.transform(tf_vectors)

    # Create mock image paths
    indexed_paths = [f"image_{i}.jpg" for i in range(5)]

    database = ImageDatabase(
        vocabulary=small_vocabulary,
        tfidf_vectors=tfidf_vectors,
        tfidf_transformer=tfidf_transformer,
        indexed_paths=indexed_paths,
        k=10
    )

    return database
