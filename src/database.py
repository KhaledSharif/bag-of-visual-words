"""Database serialization and deserialization for the BoVW system."""

import pickle
import os
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse


class ImageDatabase:
    """
    Encapsulates a BoVW image database.

    Attributes:
        vocabulary: Visual vocabulary (k x 128 matrix)
        tfidf_vectors: TF-IDF vectors for indexed images (sparse matrix)
        tfidf_transformer: Fitted TfidfTransformer for query transformation
        indexed_paths: List of image file paths in the database
        k: Number of visual words in vocabulary
    """

    def __init__(
        self,
        vocabulary: np.ndarray,
        tfidf_vectors: scipy.sparse.csr_matrix,
        tfidf_transformer: TfidfTransformer,
        indexed_paths: List[str],
        k: int
    ):
        self.vocabulary = vocabulary
        self.tfidf_vectors = tfidf_vectors
        self.tfidf_transformer = tfidf_transformer
        self.indexed_paths = indexed_paths
        self.k = k

    def to_dict(self) -> dict:
        """Convert database to dictionary for serialization."""
        return {
            'vocabulary': self.vocabulary,
            'tfidf_vectors': self.tfidf_vectors,
            'tfidf_transformer': self.tfidf_transformer,
            'indexed_paths': self.indexed_paths,
            'k': self.k
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ImageDatabase':
        """Create database from dictionary after deserialization."""
        return cls(
            vocabulary=data['vocabulary'],
            tfidf_vectors=data['tfidf_vectors'],
            tfidf_transformer=data['tfidf_transformer'],
            indexed_paths=data['indexed_paths'],
            k=data['k']
        )

    def __len__(self) -> int:
        """Return number of indexed images."""
        return len(self.indexed_paths)


def save_database(database: ImageDatabase, filepath: str) -> None:
    """
    Save an ImageDatabase to a file using pickle.

    Args:
        database: ImageDatabase instance to save
        filepath: Path where the database file will be saved

    Raises:
        IOError: If the file cannot be written
    """
    print(f"\n3/3: Saving database to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(database.to_dict(), f)

    print("---")
    print("✅ Indexing complete! Your database is ready.")
    print(f"  - Total images indexed: {len(database)}")
    print(f"  - Vocabulary size (k):  {database.k}")
    print(f"  - Database file:        {filepath}")


def load_database(filepath: str) -> ImageDatabase:
    """
    Load an ImageDatabase from a file.

    Args:
        filepath: Path to the database file

    Returns:
        ImageDatabase instance

    Raises:
        FileNotFoundError: If the database file doesn't exist
        ValueError: If the database file is malformed
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Database file not found at {filepath}")

    print(f"Loading database from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Could not load database file: {e}")

    # Validate required keys
    required_keys = ['vocabulary', 'tfidf_vectors', 'tfidf_transformer', 'indexed_paths', 'k']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Database file is malformed. Missing key: {key}")

    database = ImageDatabase.from_dict(data)
    print(f"Database loaded. (k={database.k}, {len(database)} indexed images)")

    return database
