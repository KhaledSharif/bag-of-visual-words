"""Visual vocabulary building using K-Means clustering."""

import cv2
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from .utils import load_image_grayscale
from .features import extract_descriptors


def build_vocabulary(
    image_paths: List[str],
    k: int,
    sift: cv2.SIFT,
    limit: Optional[int] = None
) -> np.ndarray:
    """
    Build a visual vocabulary using K-Means clustering of SIFT descriptors.

    This function:
    1. Extracts SIFT descriptors from all images
    2. Collects all descriptors into one pool
    3. Clusters them into k groups using K-Means
    4. Returns the cluster centers as the "visual vocabulary"

    Args:
        image_paths: List of paths to images
        k: Number of visual words (clusters)
        sift: SIFT detector instance
        limit: Optional limit on number of images to use for training

    Returns:
        Vocabulary matrix of shape (k, 128) where each row is a visual word

    Raises:
        ValueError: If no descriptors are found in any images
    """
    print(f"Building vocabulary with k={k} from {len(image_paths)} images...")

    # K-Means termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    bow_trainer = cv2.BOWKMeansTrainer(k, criteria, 1, cv2.KMEANS_PP_CENTERS)

    # Limit processing for quick tests if 'limit' is set
    paths_to_process = image_paths[:limit] if limit else image_paths

    for img_path in tqdm(paths_to_process, desc="1/3: Extracting descriptors"):
        img = load_image_grayscale(img_path)
        if img is None:
            continue

        _, descriptors = extract_descriptors(img, sift)

        if descriptors is not None:
            bow_trainer.add(descriptors)

    if bow_trainer.descriptorsCount() == 0:
        raise ValueError("No descriptors found. Check image paths and image content.")

    print("\nClustering descriptors... (This may take a while)")
    vocabulary = bow_trainer.cluster()

    print("Vocabulary built.")
    return vocabulary


def create_bow_extractor(
    vocabulary: np.ndarray,
    sift: cv2.SIFT,
    matcher: cv2.BFMatcher
) -> cv2.BOWImgDescriptorExtractor:
    """
    Create a Bag-of-Visual-Words descriptor extractor.

    This extractor converts SIFT features to histograms by assigning
    each descriptor to the nearest visual word in the vocabulary.

    Args:
        vocabulary: Visual vocabulary matrix (k x 128)
        sift: SIFT detector instance
        matcher: BFMatcher instance for finding nearest visual words

    Returns:
        Configured BOWImgDescriptorExtractor
    """
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, matcher)
    bow_extractor.setVocabulary(vocabulary)
    return bow_extractor
