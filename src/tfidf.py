"""TF-IDF (Term Frequency - Inverse Document Frequency) transformation and similarity computation."""

import cv2
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import scipy.sparse

from .utils import load_image_grayscale
from .features import extract_keypoints


def build_tfidf_vectors(
    image_paths: List[str],
    bow_extractor: cv2.BOWImgDescriptorExtractor,
    sift: cv2.SIFT
) -> Tuple[scipy.sparse.csr_matrix, TfidfTransformer, List[str]]:
    """
    Build TF-IDF vectors for a collection of images.

    This function:
    1. Computes TF (Term Frequency) histograms for each image
    2. Fits a TfidfTransformer to learn IDF (Inverse Document Frequency) weights
    3. Transforms all TF histograms to TF-IDF vectors

    Args:
        image_paths: List of paths to images
        bow_extractor: BOWImgDescriptorExtractor with vocabulary set
        sift: SIFT detector instance

    Returns:
        Tuple of:
        - TF-IDF vectors matrix (sparse, shape: num_images x k)
        - Fitted TfidfTransformer (needed to transform query images)
        - List of successfully indexed image paths

    Raises:
        ValueError: If no images could be indexed
    """
    print("Building TF-IDF database...")

    tf_vectors = []
    indexed_paths = []

    for img_path in tqdm(image_paths, desc="2/3: Creating histograms (TF)"):
        img = load_image_grayscale(img_path)
        if img is None:
            continue

        # Detect keypoints
        keypoints = extract_keypoints(img, sift)

        if keypoints:
            # Compute the BoVW histogram for the image
            histogram = bow_extractor.compute(img, keypoints)

            if histogram is not None:
                tf_vectors.append(histogram)
                indexed_paths.append(img_path)

    if not tf_vectors:
        raise ValueError("No images could be indexed. All histograms were None.")

    # Stack all TF histograms into a single (num_images, k) numpy array
    tf_vectors_matrix = np.concatenate(tf_vectors, axis=0)

    # Fit the IDF model
    print("\nFitting TF-IDF transformer...")
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(tf_vectors_matrix)

    # Transform TF vectors to TF-IDF vectors
    print("Applying TF-IDF transformation...")
    tfidf_vectors_matrix = tfidf_transformer.transform(tf_vectors_matrix)

    print("Database indexing complete.")

    return scipy.sparse.csr_matrix(tfidf_vectors_matrix), tfidf_transformer, indexed_paths


def transform_query_vector(
    tf_vector: np.ndarray,
    tfidf_transformer: TfidfTransformer
) -> scipy.sparse.csr_matrix:
    """
    Transform a query TF vector using a pre-fitted TF-IDF transformer.

    IMPORTANT: This function uses the saved transformer to apply the SAME
    IDF weights that were learned from the database. Do NOT fit a new transformer.

    Args:
        tf_vector: Raw TF histogram from query image (1 x k)
        tfidf_transformer: Pre-fitted TfidfTransformer from the database

    Returns:
        TF-IDF weighted query vector (sparse matrix, 1 x k)
    """
    return scipy.sparse.csr_matrix(tfidf_transformer.transform(tf_vector))


def compute_similarities(
    query_vector: scipy.sparse.csr_matrix,
    database_vectors: scipy.sparse.csr_matrix
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and database vectors.

    Args:
        query_vector: TF-IDF vector for query image (1 x k)
        database_vectors: TF-IDF vectors for all database images (n x k)

    Returns:
        Array of similarity scores (shape: n,) sorted from highest to lowest
    """
    # Compute cosine similarity: returns shape (1, n)
    similarities = cosine_similarity(query_vector, database_vectors)

    # Flatten to 1D array
    return similarities[0]
