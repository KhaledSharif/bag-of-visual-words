# Bag-of-Visual-Words Image Search

This project implements a basic Content-Based Image Retrieval (CBIR) system, also known as "visual search," using Python, OpenCV, and scikit-learn.

It allows you to create a searchable database from a collection of images and then find the most visually similar images for a new query. The system is built on the **Bag of Visual Words (BoVW)** model, a common technique in computer vision for tasks like image classification and visual place recognition in SLAM.

The pipeline is:

1.  **Feature Extraction** (SIFT)
2.  **Vocabulary Building** (K-Means Clustering)
3.  **Image Representation** (TF-IDF Histograms)
4.  **Querying** (Cosine Similarity)

## 📁 Project Structure

```
bows/
├── src/                    # Shared module library
│   ├── features.py         # SIFT extraction
│   ├── vocabulary.py       # Vocabulary building
│   ├── tfidf.py           # TF-IDF transformation
│   ├── database.py        # Database operations
│   └── utils.py           # Utilities
├── tests/                  # Pytest test suite
├── index.py               # Build database from images
├── query.py               # Query database with image
└── requirements.txt
```

  * `index.py`: Scans a directory of images, builds the visual vocabulary, and creates a single `*.db` database file.
  * `query.py`: Loads the database file, processes a new query image, and returns the top N most similar images from the database.
  * `src/`: Contains the refactored core functionality shared by both scripts.
  * `tests/`: Contains pytest tests for the core modules.

## 🛠️ Installation

You must have the `opencv-contrib-python` package, as SIFT is not included in the main `opencv-python` package.

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install in development mode (recommended for contributors)
pip install -e .
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_features.py
```

## 🚀 Usage

The process is a simple two-step workflow: first you **index** your image collection, then you **query** it.

### Step 1: Index Your Images

First, collect all your images into a single directory (e.g., `my_photos`).

Run `index.py` to process them. You must specify the path to your images (using a glob pattern) and a name for your output database file.

```bash
# Example: Index all JPGs in 'my_photos' and create 'travel.db'
# We will use a vocabulary of k=100 visual words.

python index.py --images "my_photos/*.jpg" --output "travel.db" --k 100
```

This script will:

1.  Find all images matching the glob pattern.
2.  Detect SIFT features in all of them.
3.  Cluster all features into `k=100` groups to create the "visual vocabulary."
4.  Re-scan every image and convert it into a TF-IDF vector based on this vocabulary.
5.  Save the vocabulary, TF-IDF vectors, and file paths into `travel.db`.

This step can take a long time if you have many images or a large `k`.

### Step 2: Query the Database

Once your `travel.db` file is created, you can query it with new images using `query.py`.

```bash
# Example: Find the 5 images in 'travel.db' that are
# most similar to 'new_image_01.jpg'.

python query.py --database "travel.db" --query "new_image_01.jpg" --n 5
```

This will produce output similar to this:

```
Loading database from travel.db...
Database loaded. (k=100, 1250 indexed images)
Processing query image: new_image_01.jpg
Finding matches...

--- Top 5 Matches for 'new_image_01.jpg' ---
  1. my_photos/DSC_0142.jpg (Similarity Score: 0.9104)
  2. my_photos/DSC_0141.jpg (Similarity Score: 0.8722)
  3. my_photos/DSC_0458.jpg (Similarity Score: 0.7651)
  4. my_photos/DSC_0139.jpg (Similarity Score: 0.7209)
  5. my_photos/DSC_0459.jpg (Similarity Score: 0.6920)
```

### CLI Arguments

#### `index.py`

  * `--images` (Required): Path to your images (e.g., `"photos/*.jpg"`).
  * `--output` (Required): File path to save the database (e.g., `"index.db"`).
  * `--k` (Optional): Number of visual words (clusters) for K-Means. A larger `k` is more descriptive but slower. Default is `100`.
  * `--limit` (Optional): Limit vocabulary training to the first N images (for a quick test).

#### `query.py`

  * `--database` (Required): Path to the `.db` file created by `index.py`.
  * `--query` (Required): Path to the new image you want to find matches for.
  * `--n` (Optional): The number of top results to show. Default is `5`.