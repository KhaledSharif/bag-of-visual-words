#!/usr/bin/env python3
"""Query a BoVW database for similar images."""

import argparse
import os
import numpy as np

from src.database import load_database
from src.features import create_sift_detector, create_matcher
from src.vocabulary import create_bow_extractor
from src.tfidf import transform_query_vector, compute_similarities
from src.utils import load_image_grayscale


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query a BoVW database for similar images.")
    parser.add_argument("--database", type=str, required=True,
                        help="Path to the .db database file created by index.py")
    parser.add_argument("--query", type=str, required=True,
                        help="Path to the new query image")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of top results to display. (Default: 5)")

    args = parser.parse_args()

    # Validate query image exists
    if not os.path.exists(args.query):
        print(f"Error: Query image not found at {args.query}")
        return

    # Load the database
    try:
        database = load_database(args.database)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # Initialize SIFT and matcher (must match index.py configuration)
    sift = create_sift_detector()
    matcher = create_matcher()

    # Setup BoVW descriptor extractor with the database vocabulary
    bow_extractor = create_bow_extractor(database.vocabulary, sift, matcher)

    # Process the query image
    print(f"Processing query image: {args.query}")
    query_img = load_image_grayscale(args.query)
    if query_img is None:
        print(f"Error: Could not read query image at {args.query}")
        return

    # Detect keypoints
    keypoints = sift.detect(query_img, None)
    if not keypoints:
        print("Error: No keypoints (features) found in query image. Cannot proceed.")
        return

    # Compute the TF histogram
    query_tf_vector = bow_extractor.compute(query_img, keypoints)

    if query_tf_vector is None:
        print("Error: Could not compute BoVW histogram. Image may be too different from the database.")
        return

    # Transform to TF-IDF vector using the database transformer
    query_tfidf_vector = transform_query_vector(query_tf_vector, database.tfidf_transformer)

    # Compute similarities
    print("Finding matches...")
    similarities = compute_similarities(query_tfidf_vector, database.tfidf_vectors)

    # Rank and get top N results
    top_n_indices = np.argsort(similarities)[::-1][:args.n]

    # Display results
    print(f"\n--- Top {args.n} Matches for '{os.path.basename(args.query)}' ---")

    if not top_n_indices.any():
        print("No matches found.")

    for i, idx in enumerate(top_n_indices):
        path = database.indexed_paths[idx]
        score = similarities[idx]

        # Don't show results with 0 similarity
        if score > 0:
            print(f"  {i+1}. {path} (Similarity Score: {score:.4f})")
        else:
            if i == 0:
                print("No matches found (all similarity scores are 0).")
            break


if __name__ == "__main__":
    main()
