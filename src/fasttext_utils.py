"""
fasttext_utils.py — Semantic Representation Upgrade (H2)
=========================================================
Replaces GloVe with FastText to solve the 3.8% OOV rate.
Because FastText uses character n-grams, it can generate decent vectors
even for completely unseen words (e.g. rare brands, misspellings), achieving
effectively 100% vocabulary coverage.
"""

import os
import torch
import numpy as np

try:
    import fasttext
    import fasttext.util
except ImportError:
    fasttext = None

def get_fasttext_model_path(download_dir="data/embeddings"):
    """Returns the path to the English FastText bin file."""
    os.makedirs(download_dir, exist_ok=True)
    bin_path = os.path.join(download_dir, 'cc.en.300.bin')
    # Use fasttext utility to download if missing, but it downloads to CWD by default
    if not os.path.exists(bin_path):
        print(f"[INFO] FastText cc.en.300.bin not found at {bin_path}.")
        print("[INFO] Please download it manually or let the script download it to current dir.")
        fasttext.util.download_model('en', if_exists='ignore')
        # Rename/Move to the target directory
        if os.path.exists('cc.en.300.bin'):
            os.rename('cc.en.300.bin', bin_path)
            if os.path.exists('cc.en.300.bin.gz'):
                os.remove('cc.en.300.bin.gz')
    return bin_path

def build_fasttext_matrix(vocab, fasttext_dim=300):
    """
    Builds the embedding matrix for the given vocabulary utilizing FastText.
    Because FastText composes OOV vectors from subwords, coverage is practically 100%.
    
    Args:
        vocab (dict): Mapping from word -> index.
        fasttext_dim (int): Dimensionality (usually 300).
        
    Returns:
        matrix (torch.Tensor): The (V, 300) embedding matrix.
        coverage (float): Always 1.0 for FastText.
    """
    if fasttext is None:
        raise ImportError("FastText is not installed. Run: pip install fasttext-wheel")
        
    bin_path = get_fasttext_model_path()
    print(f"[INFO] Loading FastText model from {bin_path} (this takes ~10 seconds)...")
    ft = fasttext.load_model(bin_path)
    
    vocab_size = len(vocab)
    matrix = np.zeros((vocab_size, fasttext_dim), dtype=np.float32)
    
    hit_count = 0
    total_count = 0
    
    # We iterate over vocab and probe FastText
    for word, idx in vocab.items():
        if word == '<pad>':
            matrix[idx] = np.zeros(fasttext_dim)
            total_count += 1
            continue
            
        vector = ft.get_word_vector(word)
        matrix[idx] = vector
        
        # Since get_word_vector always returns a vector, we technically have 100% coverage
        # But we can check if it was exactly in the model's vocabulary vs composed from subwords
        # Usually checking word in ft.words is the exact hit.
        if word in ft.words:
            hit_count += 1
        total_count += 1

    print(f"[INFO] FastText Built: Covered {total_count}/{vocab_size} words (100% effectively).")
    print(f"[INFO] Exact FastText Vocab Hits: {hit_count}/{vocab_size} ({hit_count/max(1, vocab_size)*100:.2f}%)")
    print(f"[INFO] OOV Handled via Sub-word n-grams: {total_count - hit_count - 1}") # -1 for pad
    
    return torch.tensor(matrix, dtype=torch.float32), 1.0

if __name__ == '__main__':
    # Simple test
    test_vocab = {'<pad>': 0, 'dog': 1, 'cat': 2, 'unknowableword123': 3}
    mat, cov = build_fasttext_matrix(test_vocab)
    print("Matrix shape:", mat.shape)
    print("Vector for 'unknowableword123':", mat[3][:5], "...")
