"""Tests for TF-IDF transformation and similarity computation."""

import pytest
import numpy as np
import scipy.sparse
import cv2
import os
from sklearn.feature_extraction.text import TfidfTransformer

from src.tfidf import transform_query_vector, compute_similarities, build_tfidf_vectors
from src.vocabulary import build_vocabulary, create_bow_extractor
from src.features import create_sift_detector, create_matcher


def test_transform_query_vector():
    """Test query vector transformation."""
    # Create a simple TF vector
    tf_vector = np.array([[1, 2, 0, 3, 0, 1, 0, 0, 2, 1]])

    # Create and fit a transformer on some training data
    train_data = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    transformer.fit(train_data)

    # Transform the query vector
    tfidf_vector = transform_query_vector(tf_vector, transformer)

    # Result should be a sparse matrix
    assert scipy.sparse.issparse(tfidf_vector)
    assert tfidf_vector.shape == (1, 10)


def test_compute_similarities():
    """Test cosine similarity computation."""
    # Create a query vector and database vectors
    query_vector = scipy.sparse.csr_matrix([[1, 0, 1, 0, 1]])
    database_vectors = scipy.sparse.csr_matrix([
        [1, 0, 1, 0, 1],  # Identical to query (similarity should be 1.0)
        [0, 1, 0, 1, 0],  # Orthogonal to query (similarity should be 0.0)
        [2, 0, 2, 0, 2],  # Scaled version of query (similarity should be 1.0)
        [1, 1, 1, 1, 1],  # Partially similar
    ])

    similarities = compute_similarities(query_vector, database_vectors)

    # Check shape
    assert similarities.shape == (4,)

    # Check specific values
    assert similarities[0] == pytest.approx(1.0, abs=1e-5)  # Identical
    assert similarities[1] == pytest.approx(0.0, abs=1e-5)  # Orthogonal
    assert similarities[2] == pytest.approx(1.0, abs=1e-5)  # Scaled

    # All similarities should be in range [0, 1] (with small tolerance for floating point errors)
    assert all(0 - 1e-10 <= s <= 1 + 1e-10 for s in similarities)


def test_compute_similarities_returns_array():
    """Test that compute_similarities returns a 1D numpy array."""
    query = scipy.sparse.csr_matrix([[1, 2, 3]])
    database = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]])

    similarities = compute_similarities(query, database)

    assert isinstance(similarities, np.ndarray)
    assert similarities.ndim == 1
    assert len(similarities) == 2


def test_build_tfidf_vectors_success(temp_dir):
    """Test successfully building TF-IDF vectors from multiple images."""
    # Create images with rich features using random patterns
    import numpy as np

    image_paths = []
    for i in range(3):
        # Create random image with structure
        img = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
        # Add geometric shapes to ensure features
        cv2.rectangle(img, (20 + i*10, 20), (80 + i*10, 80), 255, 2)
        cv2.circle(img, (150, 150), 30 + i*10, 255, 2)

        path = os.path.join(temp_dir, f"img{i}.jpg")
        cv2.imwrite(path, img)
        image_paths.append(path)

    # Build vocabulary
    sift = create_sift_detector()
    matcher = create_matcher()
    vocabulary = build_vocabulary(image_paths, k=10, sift=sift, limit=None)
    bow_extractor = create_bow_extractor(vocabulary, sift, matcher)

    # Build TF-IDF vectors
    tfidf_vectors, tfidf_transformer, indexed_paths = build_tfidf_vectors(
        image_paths, bow_extractor, sift
    )

    # Verify outputs
    assert scipy.sparse.issparse(tfidf_vectors), "TF-IDF vectors should be sparse matrix"
    assert tfidf_vectors.shape[0] == 3, "Should have 3 image vectors"
    assert tfidf_vectors.shape[1] == 10, "Should have 10 dimensions (k=10)"
    assert isinstance(tfidf_transformer, TfidfTransformer), "Should return a TfidfTransformer"
    assert len(indexed_paths) == 3, "Should have 3 indexed paths"
    assert indexed_paths == image_paths, "Indexed paths should match input paths"


def test_build_tfidf_vectors_no_images_raises_error(temp_dir):
    """Test that ValueError is raised when no images can be indexed."""
    # Create an invalid image file
    invalid_image = os.path.join(temp_dir, "invalid.jpg")
    with open(invalid_image, "w") as f:
        f.write("not an image")

    sift = create_sift_detector()
    matcher = create_matcher()
    # Create a dummy vocabulary
    vocabulary = np.random.rand(10, 128).astype(np.float32)
    bow_extractor = create_bow_extractor(vocabulary, sift, matcher)

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        build_tfidf_vectors([invalid_image], bow_extractor, sift)

    assert "No images could be indexed" in str(exc_info.value)


def test_build_tfidf_vectors_mixed_valid_invalid(temp_dir):
    """Test building vectors with a mix of valid and invalid images."""
    import numpy as np

    # Create valid images with rich features
    valid_paths = []
    for i in range(2):
        img = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
        cv2.rectangle(img, (20 + i*10, 20), (80 + i*10, 80), 255, 2)
        cv2.circle(img, (150, 150), 30 + i*10, 255, 2)
        path = os.path.join(temp_dir, f"valid{i}.jpg")
        cv2.imwrite(path, img)
        valid_paths.append(path)

    # Create invalid image
    invalid_image = os.path.join(temp_dir, "invalid.jpg")
    with open(invalid_image, "w") as f:
        f.write("not an image")

    # Mix valid and invalid
    image_paths = [valid_paths[0], invalid_image, valid_paths[1]]

    # Build vocabulary from valid images only
    sift = create_sift_detector()
    matcher = create_matcher()
    vocabulary = build_vocabulary(valid_paths, k=10, sift=sift, limit=None)
    bow_extractor = create_bow_extractor(vocabulary, sift, matcher)

    # Build TF-IDF vectors
    tfidf_vectors, _, indexed_paths = build_tfidf_vectors(
        image_paths, bow_extractor, sift
    )

    # Should only index the 2 valid images
    assert tfidf_vectors.shape[0] == 2, "Should have 2 image vectors (invalid skipped)"
    assert len(indexed_paths) == 2, "Should have 2 indexed paths"
    assert valid_paths[0] in indexed_paths
    assert valid_paths[1] in indexed_paths
    assert invalid_image not in indexed_paths


def test_build_tfidf_vectors_single_image(temp_dir):
    """Test building TF-IDF vectors from a single image."""
    import numpy as np

    # Create image with rich features
    img = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), 255, 2)
    cv2.circle(img, (150, 150), 30, 255, 2)
    cv2.line(img, (50, 150), (150, 50), 255, 2)

    image_path = os.path.join(temp_dir, "single.jpg")
    cv2.imwrite(image_path, img)

    # Build vocabulary and TF-IDF
    sift = create_sift_detector()
    matcher = create_matcher()
    vocabulary = build_vocabulary([image_path], k=10, sift=sift, limit=None)
    bow_extractor = create_bow_extractor(vocabulary, sift, matcher)
    tfidf_vectors, _, indexed_paths = build_tfidf_vectors(
        [image_path], bow_extractor, sift
    )

    # Verify single image was indexed
    assert tfidf_vectors.shape[0] == 1
    assert len(indexed_paths) == 1
    assert indexed_paths[0] == image_path


def test_build_tfidf_vectors_output_format(temp_dir):
    """Test that build_tfidf_vectors returns correctly formatted outputs."""
    import numpy as np

    # Create test images with rich features
    image_paths = []
    for i in range(2):
        img = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
        cv2.rectangle(img, (20 + i*10, 20), (80 + i*10, 80), 255, 2)
        cv2.circle(img, (150, 150), 30 + i*10, 255, 2)
        path = os.path.join(temp_dir, f"img{i}.jpg")
        cv2.imwrite(path, img)
        image_paths.append(path)

    # Build vocabulary and TF-IDF
    sift = create_sift_detector()
    matcher = create_matcher()
    vocabulary = build_vocabulary(image_paths, k=10, sift=sift, limit=None)
    bow_extractor = create_bow_extractor(vocabulary, sift, matcher)
    tfidf_vectors, tfidf_transformer, indexed_paths = build_tfidf_vectors(
        image_paths, bow_extractor, sift
    )

    # Verify return types
    assert scipy.sparse.issparse(tfidf_vectors), "Should return sparse matrix"
    assert isinstance(tfidf_vectors, scipy.sparse.csr_matrix), "Should be CSR format"
    assert isinstance(tfidf_transformer, TfidfTransformer), "Should return TfidfTransformer"
    assert isinstance(indexed_paths, list), "Should return list of paths"
    assert all(isinstance(p, str) for p in indexed_paths), "All paths should be strings"

    # Verify transformer is fitted
    assert hasattr(tfidf_transformer, 'idf_'), "Transformer should be fitted (has idf_ attribute)"
