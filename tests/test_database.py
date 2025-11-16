"""Tests for database save/load operations."""

import os
import pytest
from src.database import ImageDatabase, save_database, load_database


def test_database_creation(mock_database):
    """Test ImageDatabase creation."""
    assert isinstance(mock_database, ImageDatabase)
    assert mock_database.k == 10
    assert len(mock_database) == 5
    assert len(mock_database.indexed_paths) == 5


def test_database_save_and_load(mock_database, temp_dir):
    """Test database save/load roundtrip preserves data."""
    db_path = os.path.join(temp_dir, "test_database.db")

    # Save the database
    save_database(mock_database, db_path)

    # Verify file was created
    assert os.path.exists(db_path)

    # Load the database
    loaded_db = load_database(db_path)

    # Verify data is preserved
    assert loaded_db.k == mock_database.k
    assert len(loaded_db) == len(mock_database)
    assert loaded_db.indexed_paths == mock_database.indexed_paths
    assert loaded_db.vocabulary.shape == mock_database.vocabulary.shape
    assert loaded_db.tfidf_vectors.shape == mock_database.tfidf_vectors.shape


def test_load_nonexistent_database_raises_error():
    """Test loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_database("nonexistent_database.db")


def test_database_to_dict(mock_database):
    """Test database to_dict method."""
    data = mock_database.to_dict()

    assert isinstance(data, dict)
    assert 'vocabulary' in data
    assert 'tfidf_vectors' in data
    assert 'tfidf_transformer' in data
    assert 'indexed_paths' in data
    assert 'k' in data


def test_database_from_dict(mock_database):
    """Test database from_dict method."""
    data = mock_database.to_dict()
    new_db = ImageDatabase.from_dict(data)

    assert new_db.k == mock_database.k
    assert len(new_db) == len(mock_database)
    assert new_db.indexed_paths == mock_database.indexed_paths


def test_load_corrupted_database_raises_error(temp_dir):
    """Test loading a corrupted (non-pickle) file raises ValueError."""
    # Create a corrupted database file (not a valid pickle)
    corrupted_path = os.path.join(temp_dir, "corrupted.db")
    with open(corrupted_path, "w") as f:
        f.write("This is not a valid pickle file")

    # Should raise ValueError with descriptive message
    with pytest.raises(ValueError) as exc_info:
        load_database(corrupted_path)

    assert "Could not load database file" in str(exc_info.value)


def test_load_malformed_database_missing_keys(temp_dir):
    """Test loading a database with missing required keys raises ValueError."""
    import pickle

    # Create a database file with incomplete data (missing keys)
    malformed_path = os.path.join(temp_dir, "malformed.db")
    incomplete_data = {
        'vocabulary': None,
        # Missing 'tfidf_vectors', 'tfidf_transformer', 'indexed_paths', 'k'
    }

    with open(malformed_path, "wb") as f:
        pickle.dump(incomplete_data, f)

    # Should raise ValueError about missing key
    with pytest.raises(ValueError) as exc_info:
        load_database(malformed_path)

    assert "Database file is malformed" in str(exc_info.value)
    assert "Missing key" in str(exc_info.value)
