# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Content-Based Image Retrieval (CBIR) system using the Bag of Visual Words (BoVW) model. The system consists of two main scripts and a refactored modular architecture:

1. **index.py** - Builds a searchable database from image collections
2. **query.py** - Searches the database for visually similar images
3. **src/** - Shared module library containing core functionality
4. **tests/** - Pytest test suite covering core functions

## Core Architecture

### Directory Structure

```
bows/
├── src/                    # Shared module library
│   ├── features.py         # SIFT extraction & keypoint detection
│   ├── vocabulary.py       # Vocabulary building & BOW setup
│   ├── tfidf.py           # TF-IDF transformation & similarity
│   ├── database.py        # Serialization & ImageDatabase class
│   └── utils.py           # Image loading & path handling
├── tests/                  # Pytest test suite
│   ├── conftest.py        # Fixtures (sample images, mock database)
│   ├── test_features.py
│   ├── test_vocabulary.py
│   ├── test_database.py
│   └── test_tfidf.py
├── index.py               # CLI entry point for indexing
├── query.py               # CLI entry point for querying
├── requirements.txt
└── pyproject.toml
```

### Pipeline Overview

The system implements a classic BoVW pipeline with TF-IDF weighting:

1. **Feature Extraction** (`src/features.py`): SIFT features are extracted from all images
2. **Vocabulary Building** (`src/vocabulary.py`): K-Means clustering creates "visual words" from descriptors
3. **Image Representation** (`src/tfidf.py`): Each image is represented as a TF-IDF weighted histogram
4. **Querying** (`src/tfidf.py`): Cosine similarity finds the most similar images

### Module Breakdown

**src/features.py:**
- `create_sift_detector()` - Returns configured SIFT instance
- `create_matcher()` - Returns BFMatcher with L2 norm
- `extract_keypoints()` - Detect keypoints only
- `extract_descriptors()` - Detect keypoints and compute descriptors

**src/vocabulary.py:**
- `build_vocabulary()` - Clusters SIFT descriptors into k visual words using K-Means
- `create_bow_extractor()` - Creates BOWImgDescriptorExtractor with vocabulary

**src/tfidf.py:**
- `build_tfidf_vectors()` - Computes TF histograms and fits IDF transformer
- `transform_query_vector()` - Applies saved IDF weights to query
- `compute_similarities()` - Calculates cosine similarity

**src/database.py:**
- `ImageDatabase` class - Encapsulates vocabulary, vectors, transformer, paths
- `save_database()` - Serializes database to pickle file
- `load_database()` - Deserializes database with validation

**src/utils.py:**
- `load_image_grayscale()` - Safe image loading with error handling
- `collect_image_paths()` - Collects images from glob pattern

### Critical Implementation Details

**Consistency Requirements:**
- SIFT detector and BFMatcher must be created with identical parameters in both scripts
- Both use `create_sift_detector()` and `create_matcher()` from `src/features.py`
- Query processing must use the saved `tfidf_transformer.transform()`, never `.fit_transform()`

**Database Structure:**
The `ImageDatabase` class encapsulates all components needed for querying. The `.db` file contains:
- `vocabulary`: numpy array (k x 128) of visual word cluster centers
- `tfidf_vectors`: scipy sparse matrix of TF-IDF vectors for indexed images
- `tfidf_transformer`: fitted TfidfTransformer instance
- `indexed_paths`: list of image file paths
- `k`: number of visual words

## Development Setup

### Installation
```bash
# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_features.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src tests/
```

## Common Commands

### Building a Database
```bash
# Index images with default vocabulary size (k=100)
python index.py --images "path/to/images/*.jpg" --output "database.db"

# Index with custom vocabulary size
python index.py --images "images/*.jpg" --output "index.db" --k 200

# Quick test with limited training set
python index.py --images "images/*.jpg" --output "test.db" --k 50 --limit 100
```

### Querying
```bash
# Find top 5 similar images (default)
python query.py --database "database.db" --query "query_image.jpg"

# Find top 10 similar images
python query.py --database "database.db" --query "query_image.jpg" --n 10
```

## Dependencies

Requires `opencv-contrib-python` (not regular `opencv-python`) because SIFT is not in the main package.

See `requirements.txt` or `pyproject.toml` for complete dependency list.

## Performance Considerations

- Vocabulary building (k-means clustering) is the slowest phase
- Larger `k` values provide more descriptive vocabularies but increase processing time and database size
- The `--limit` parameter in index.py can be used for quick testing with a subset of images
- Images without readable features or matching vocabulary will be skipped gracefully

## Code Organization Principles

- **Entry points stay simple**: index.py and query.py are ~50-70 lines, orchestrating calls to src/ modules
- **Pure functions**: Most functions in src/ modules avoid side effects for better testability
- **Error handling**: All modules raise descriptive exceptions that are caught in CLI scripts
- **No algorithmic changes**: The refactoring only reorganized code; the BoVW algorithm is unchanged
