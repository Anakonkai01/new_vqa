"""
concept_gnn.py — Tier 9: ConceptNet Knowledge Graph Integration
===============================================================
Enriches question word representations with commonsense knowledge from
ConceptNet via a lightweight Graph Convolutional Network (GCN).

ARCHITECTURE:
  1. Build a word-level knowledge graph from the vocabulary:
     - Nodes: vocabulary words
     - Edges: ConceptNet relations (IsA, PartOf, HasA, UsedFor, AtLocation)
       loaded from ConceptNet Assertions CSV if available, otherwise
       falls back to a word co-occurrence graph built from training data.

  2. Apply a 2-layer GCN to propagate knowledge:
     GCN_0: H_0 = ReLU(A_hat @ E @ W_0)    (A_hat = normalized adjacency)
     GCN_1: H_1 = A_hat @ H_0 @ W_1
     Output: (vocab_size, gcn_out_dim) — enriched word embeddings

  3. Optional integration: replace/augment the QuestionEncoder's embedding
     layer with the GCN-enriched embeddings.

DEPENDENCIES: torch_geometric (optional)
  pip install torch_geometric
  If not installed: ConceptGNN falls back to a plain Linear projection
  (equivalent to a 0-layer GCN with no graph structure).

USAGE:
  from models.concept_gnn import ConceptGNN

  gnn = ConceptGNN(vocab_size=5000, embed_dim=512, gcn_dim=256,
                   concept_csv='data/conceptnet_assertions.csv')  # optional
  enriched = gnn(word_ids)   # (B, S, gcn_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


# ── Graph construction ────────────────────────────────────────────────────────

_CONCEPT_RELATIONS = {
    '/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor', '/r/AtLocation',
    '/r/CapableOf', '/r/RelatedTo', '/r/SimilarTo',
}


def build_cooccurrence_graph(vocab, annotations, window=5):
    """
    Fallback: build a word co-occurrence graph from training annotations
    when ConceptNet CSV is not available.

    Returns edge_index: (2, num_edges) LongTensor — bidirectional edges
    """
    from collections import defaultdict
    import re

    co_counts = defaultdict(int)
    pattern   = re.compile(r"[a-z']+")

    for ann in annotations:
        text  = (ann.get('question', '') + ' ' +
                 ann.get('multiple_choice_answer', '') + ' ' +
                 ' '.join(ann.get('explanation', [])))
        words = pattern.findall(text.lower())
        ids   = [vocab.word2idx.get(w, 3) for w in words]  # 3 = <unk>
        for i in range(len(ids)):
            for j in range(i+1, min(i+window+1, len(ids))):
                if ids[i] != ids[j] and ids[i] > 3 and ids[j] > 3:
                    pair = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                    co_counts[pair] += 1

    # Keep top-50K edges by co-occurrence count
    top_edges = sorted(co_counts.items(), key=lambda x: -x[1])[:50000]
    if not top_edges:
        return torch.zeros(2, 0, dtype=torch.long)

    src = [e[0] for e, _ in top_edges] + [e[1] for e, _ in top_edges]
    dst = [e[1] for e, _ in top_edges] + [e[0] for e, _ in top_edges]
    return torch.tensor([src, dst], dtype=torch.long)


def build_conceptnet_graph(vocab, concept_csv_path):
    """
    Build a knowledge graph from ConceptNet Assertions CSV.

    ConceptNet CSV format (tab-separated):
      uri  relation  subject  object  metadata_json
    e.g.: /a/[...] /r/IsA /c/en/dog /c/en/animal {...}

    Only keeps English concepts that appear in our vocabulary.
    Returns edge_index: (2, num_edges) LongTensor
    """
    import csv

    w2i = vocab.word2idx
    edges = set()

    with open(concept_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 4:
                continue
            relation = row[1]
            if relation not in _CONCEPT_RELATIONS:
                continue
            # Extract word from /c/en/word or /c/en/word/POS
            def extract_word(uri):
                parts = uri.split('/')
                if len(parts) >= 4 and parts[2] == 'en':
                    return parts[3].replace('_', ' ').lower()
                return None
            w1 = extract_word(row[2])
            w2 = extract_word(row[3])
            if w1 and w2 and w1 in w2i and w2 in w2i:
                i1, i2 = w2i[w1], w2i[w2]
                edges.add((i1, i2))
                edges.add((i2, i1))

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)

    src, dst = zip(*edges)
    return torch.tensor([list(src), list(dst)], dtype=torch.long)


# ── GCN module ────────────────────────────────────────────────────────────────

class ConceptGNN(nn.Module):
    """
    ConceptNet-aware word embedding enrichment via GCN.

    When torch_geometric is available: applies a 2-layer GCN over the
    knowledge graph to propagate commonsense knowledge between related words.

    When torch_geometric is NOT available: falls back to a 2-layer MLP
    (same dimensionality, no graph structure — pure parameter overhead is
    minimal, and the module is still a drop-in replacement).

    Args:
        vocab_size  : number of tokens in the vocabulary
        embed_dim   : input word embedding dimension (matches QuestionEncoder)
        gcn_dim     : hidden and output GCN dimension
        concept_csv : path to ConceptNet Assertions CSV (optional)
        annotations : list of training annotations (for co-occurrence graph fallback)
        vocab       : Vocabulary object (needed for word→id lookup in graph building)
    """

    def __init__(self, vocab_size: int, embed_dim: int, gcn_dim: int = 256,
                 concept_csv: str = None, annotations: list = None, vocab=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.gcn_dim    = gcn_dim

        # Base embedding (shared with or separate from QuestionEncoder)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self._edge_index = None   # built lazily

        if _PYG_AVAILABLE:
            self.gcn1 = GCNConv(embed_dim, gcn_dim)
            self.gcn2 = GCNConv(gcn_dim, gcn_dim)
            self._use_gcn = True
            print("ConceptGNN       : torch_geometric available — GCN mode")
        else:
            # Fallback: 2-layer MLP (same interface, no graph)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, gcn_dim),
                nn.ReLU(inplace=True),
                nn.Linear(gcn_dim, gcn_dim),
            )
            self._use_gcn = False
            print("ConceptGNN       : torch_geometric NOT available — MLP fallback")
            print("  Install: pip install torch_geometric")

        # Build graph (lazily stored — built during first forward call)
        self._concept_csv   = concept_csv
        self._annotations   = annotations
        self._vocab         = vocab
        self._graph_built   = False

    def _build_graph(self, device):
        """Build and cache the knowledge graph edge_index."""
        if self._vocab is None:
            self._edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            self._graph_built = True
            return

        if self._concept_csv and os.path.exists(self._concept_csv):
            print("ConceptGNN       : building graph from ConceptNet CSV ...")
            ei = build_conceptnet_graph(self._vocab, self._concept_csv)
        elif self._annotations:
            print("ConceptGNN       : building co-occurrence graph from annotations ...")
            ei = build_cooccurrence_graph(self._vocab, self._annotations)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)

        self._edge_index  = ei.to(device)
        self._graph_built = True
        print(f"ConceptGNN graph : {self._edge_index.shape[1]:,} edges")

    def _gcn_forward(self, device):
        """
        Run GCN over all vocabulary embeddings once per forward call.
        Returns: (vocab_size, gcn_dim) enriched embeddings.
        """
        if not self._graph_built:
            self._build_graph(device)

        all_ids   = torch.arange(self.vocab_size, device=device)
        node_feat = self.embedding(all_ids)   # (V, embed_dim)

        if self._use_gcn and self._edge_index.shape[1] > 0:
            h = F.relu(self.gcn1(node_feat, self._edge_index))   # (V, gcn_dim)
            h = self.gcn2(h, self._edge_index)                   # (V, gcn_dim)
        elif self._use_gcn:
            # No edges available — run without message passing
            h = F.relu(self.gcn1(node_feat,
                                 torch.zeros(2, 1, dtype=torch.long, device=device)))
            h = self.gcn2(h, torch.zeros(2, 1, dtype=torch.long, device=device))
        else:
            h = self.mlp(node_feat)   # (V, gcn_dim)

        return h   # (vocab_size, gcn_dim)

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            word_ids : (B, S) — token ids

        Returns:
            (B, S, gcn_dim) — GCN-enriched embeddings
        """
        enriched_vocab = self._gcn_forward(word_ids.device)   # (V, gcn_dim)
        # Index into enriched vocab
        return enriched_vocab[word_ids]   # (B, S, gcn_dim)
