"""Tests for TF-IDF transformation and similarity computation."""

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfTransformer

from src.tfidf import transform_query_vector, compute_similarities


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

    # All similarities should be in range [0, 1]
    assert all(0 <= s <= 1 for s in similarities)


def test_compute_similarities_returns_array():
    """Test that compute_similarities returns a 1D numpy array."""
    query = scipy.sparse.csr_matrix([[1, 2, 3]])
    database = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]])

    similarities = compute_similarities(query, database)

    assert isinstance(similarities, np.ndarray)
    assert similarities.ndim == 1
    assert len(similarities) == 2
