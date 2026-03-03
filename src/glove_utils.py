"""
GloVe embedding utilities.

Provides functions to:
  1. Download GloVe 6B vectors (if not already cached)
  2. Build an embedding matrix aligned with a Vocabulary object

Usage:
    from glove_utils import build_glove_matrix
    matrix = build_glove_matrix(vocab, glove_dim=300, glove_path="data/glove")
    # matrix: (vocab_size, glove_dim) numpy array, rows aligned with vocab.word2idx
"""

import os
import sys
import zipfile
import numpy as np

GLOVE_URL  = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR  = "data/glove"


def download_glove(glove_dir=GLOVE_DIR, dim=300):
    """
    Download and extract GloVe 6B vectors if not already present.
    Returns the path to the extracted .txt file.
    """
    os.makedirs(glove_dir, exist_ok=True)
    txt_path = os.path.join(glove_dir, f"glove.6B.{dim}d.txt")

    if os.path.exists(txt_path):
        return txt_path

    zip_path = os.path.join(glove_dir, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading GloVe 6B from {GLOVE_URL} ...")
        print(f"  (822 MB — this may take a few minutes)")
        import urllib.request
        urllib.request.urlretrieve(GLOVE_URL, zip_path)
        print(f"  Downloaded to {zip_path}")

    print(f"Extracting glove.6B.{dim}d.txt ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        target = f"glove.6B.{dim}d.txt"
        z.extract(target, glove_dir)
    print(f"  Extracted to {txt_path}")

    return txt_path


def load_glove_vectors(glove_path, dim=300):
    """
    Load GloVe vectors from a .txt file into a dict {word: np.array}.
    """
    vectors = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            if len(vec) == dim:
                vectors[word] = vec
    print(f"  Loaded {len(vectors):,} GloVe vectors ({dim}d)")
    return vectors


def build_glove_matrix(vocab, glove_dim=300, glove_dir=GLOVE_DIR):
    """
    Build an embedding matrix of shape (vocab_size, glove_dim).

    - Words found in GloVe → use pretrained vector
    - Words NOT found (OOV) → random init from N(0, 0.1)
    - <pad> (index 0) → zeros

    Args:
        vocab     : Vocabulary object with word2idx dict
        glove_dim : 50, 100, 200, or 300
        glove_dir : directory to cache GloVe files

    Returns:
        matrix    : np.ndarray of shape (len(vocab), glove_dim)
        coverage  : float — fraction of vocab words found in GloVe
    """
    glove_path = download_glove(glove_dir, glove_dim)
    vectors    = load_glove_vectors(glove_path, glove_dim)

    vocab_size = len(vocab)
    matrix     = np.random.normal(0, 0.1, (vocab_size, glove_dim)).astype(np.float32)
    matrix[0]  = 0.0  # <pad> should be zeros

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in vectors:
            matrix[idx] = vectors[word]
            found += 1

    coverage = found / vocab_size
    print(f"  GloVe coverage: {found}/{vocab_size} = {coverage:.1%}")
    return matrix, coverage


if __name__ == "__main__":
    # Quick test — download + build matrix for a dummy vocab
    sys.path.append(os.path.dirname(__file__))
    from vocab import Vocabulary

    vocab = Vocabulary()
    vocab.load("data/processed/vocab_questions.json")

    matrix, cov = build_glove_matrix(vocab, glove_dim=300)
    print(f"Matrix shape: {matrix.shape}")   # (4546, 300)
    print(f"Coverage    : {cov:.1%}")
