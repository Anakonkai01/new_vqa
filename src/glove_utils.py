"""GloVe embedding utilities.

Downloads GloVe 6B pre-trained vectors (if not cached locally) and builds
an embedding weight matrix aligned with a ``Vocabulary`` object.

Typical usage::

    from glove_utils import build_glove_matrix

    matrix, coverage = build_glove_matrix(vocab, glove_dim=300)
    # matrix   : np.ndarray of shape (vocab_size, 300)
    # coverage : float in [0, 1] — fraction of vocab words found in GloVe
"""

from __future__ import annotations

import os
import zipfile
from typing import Dict, Tuple

import numpy as np

from vocab import Vocabulary

_GLOVE_URL:     str = "https://nlp.stanford.edu/data/glove.6B.zip"
_DEFAULT_DIR:   str = "data/glove"


def _download_glove(glove_dir: str, dim: int) -> str:
    """Download and extract GloVe 6B vectors if not already cached.

    Args:
        glove_dir: Directory in which to store the downloaded files.
        dim: Embedding dimension — one of 50, 100, 200, 300.

    Returns:
        Absolute path to the extracted ``.txt`` file.
    """
    os.makedirs(glove_dir, exist_ok=True)
    txt_path = os.path.join(glove_dir, f"glove.6B.{dim}d.txt")

    if os.path.exists(txt_path):
        return txt_path

    zip_path = os.path.join(glove_dir, "glove.6B.zip")
    if not os.path.exists(zip_path):
        import urllib.request
        print(f"Downloading GloVe 6B (~822 MB) from {_GLOVE_URL} ...")
        urllib.request.urlretrieve(_GLOVE_URL, zip_path)
        print(f"  Saved to {zip_path}")

    print(f"Extracting glove.6B.{dim}d.txt ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(f"glove.6B.{dim}d.txt", glove_dir)
    print(f"  Extracted to {txt_path}")
    return txt_path


def _load_glove_vectors(glove_path: str, dim: int) -> Dict[str, np.ndarray]:
    """Load GloVe vectors from a ``.txt`` file into a word → vector mapping.

    Args:
        glove_path: Path to the GloVe ``.txt`` file.
        dim: Expected embedding dimension (used for row validation).

    Returns:
        Dict mapping each word string to its ``np.ndarray`` of shape ``(dim,)``.
    """
    vectors: Dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            if len(vec) == dim:
                vectors[word] = vec
    print(f"  Loaded {len(vectors):,} GloVe vectors ({dim}d)")
    return vectors


def build_glove_matrix(
    vocab: Vocabulary,
    glove_dim: int = 300,
    glove_dir: str = _DEFAULT_DIR,
) -> Tuple[np.ndarray, float]:
    """Build a pre-trained embedding weight matrix aligned with *vocab*.

    Initialisation rules:

    - Words found in GloVe → use the pre-trained vector.
    - OOV words → random ``N(0, 0.1)`` (matches scale of GloVe vectors).
    - ``<pad>`` (index 0) → all-zeros. This is consistent with
      ``nn.Embedding(padding_idx=0)`` where the pad embedding is never
      updated by gradients regardless, but explicit zeros prevent any
      accidental contribution in the first forward pass.

    Args:
        vocab: ``Vocabulary`` object whose ``word2idx`` defines the row order.
        glove_dim: Embedding dimension — one of 50, 100, 200, 300.
        glove_dir: Local directory to cache the GloVe zip and txt files.

    Returns:
        Tuple of:
            matrix   – ``np.ndarray`` of shape ``(vocab_size, glove_dim)``.
            coverage – Fraction of vocab words found in GloVe (float in 0–1).
    """
    glove_path = _download_glove(glove_dir, glove_dim)
    vectors    = _load_glove_vectors(glove_path, glove_dim)

    vocab_size = len(vocab)
    matrix     = np.random.normal(0.0, 0.1, (vocab_size, glove_dim)).astype(np.float32)
    matrix[vocab.pad_idx] = 0.0  # <pad> is always explicitly zeroed

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in vectors:
            matrix[idx] = vectors[word]
            found += 1

    coverage = found / vocab_size
    print(f"  GloVe coverage: {found}/{vocab_size} = {coverage:.1%}")
    return matrix, coverage


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))

    test_vocab = Vocabulary()
    test_vocab.load("data/processed/vocab_questions.json")
    print(f"Vocab: {test_vocab}")

    mat, cov = build_glove_matrix(test_vocab, glove_dim=300)
    print(f"Matrix shape : {mat.shape}")
    print(f"Coverage     : {cov:.1%}")
