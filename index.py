#!/usr/bin/env python3
"""Build a BoVW visual index from a directory of images."""

import argparse

from src.utils import collect_image_paths
from src.features import create_sift_detector, create_matcher
from src.vocabulary import build_vocabulary, create_bow_extractor
from src.tfidf import build_tfidf_vectors
from src.database import ImageDatabase, save_database


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build a BoVW visual index from a directory of images.")
    parser.add_argument("--images", type=str, required=True,
                        help="Path to images (glob-style). Example: 'images/*.jpg'")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output database file. Example: 'index.db'")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of visual words (clusters) for K-Means. (Default: 100)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit vocabulary training to the first N images. (Default: all)")

    args = parser.parse_args()

    # Collect image paths
    try:
        image_paths = collect_image_paths(args.images)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Found {len(image_paths)} images to index.")

    # Initialize SIFT and matcher
    sift = create_sift_detector()
    matcher = create_matcher()

    # Phase 1: Build the visual vocabulary
    try:
        vocabulary = build_vocabulary(image_paths, args.k, sift, args.limit)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Phase 2: Build TF-IDF weighted database
    bow_extractor = create_bow_extractor(vocabulary, sift, matcher)
    try:
        tfidf_vectors, tfidf_transformer, indexed_paths = build_tfidf_vectors(
            image_paths, bow_extractor, sift
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Phase 3: Save the database
    database = ImageDatabase(
        vocabulary=vocabulary,
        tfidf_vectors=tfidf_vectors,
        tfidf_transformer=tfidf_transformer,
        indexed_paths=indexed_paths,
        k=args.k
    )

    save_database(database, args.output)


if __name__ == "__main__":
    main()
