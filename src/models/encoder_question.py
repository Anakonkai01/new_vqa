import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from decoder_lstm import HighwayLayer


"""
QuestionEncoder — BiLSTM with optional Tier-7 upgrades:
  - Highway connections between BiLSTM layers (Tier 7B)
  - Char-CNN character embedding pre-pended to word embeddings (Tier 7C)

input: (batch, max_len)
embedding: (batch, max_len, embed_size)
LSTM: (batch, max_len, hidden_size)
output: hidden[-1] (batch, hidden_size)
"""


# ── Tier 7C: Character CNN ───────────────────────────────────────────────────
class CharCNNEmbedding(nn.Module):
    """
    Character-level CNN embedding.

    Handles OOV words (proper nouns, rare words) that GloVe misses.
    Pre-pended to word embeddings before the BiLSTM.

    Architecture: char_embed → Conv1d(k) × N → MaxPool → concat → (B, S, output_size)

    Char IDs are looked up from a precomputed table (vocab_size × MAX_WORD_LEN)
    built at model init time using the Vocabulary object — NO dataset changes needed.

    Output dimension: num_filters * len(kernel_sizes)  (default: 100 * 3 = 300)
    """
    MAX_WORD_LEN = 20
    # 95 printable ASCII chars (0x20–0x7E) + index 0 for padding
    ALPHABET_SIZE = 96

    def __init__(self, embed_dim: int = 50, num_filters: int = 100,
                 kernel_sizes: tuple = (3, 4, 5)):
        super().__init__()
        self.char_embed = nn.Embedding(self.ALPHABET_SIZE, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes
        ])
        self.output_size = num_filters * len(kernel_sizes)   # 300 by default
        # char_table registered as a buffer (not a parameter) — travels with the model
        # It is populated by build_char_table() after __init__
        self.register_buffer('char_table', None)

    def build_char_table(self, vocab):
        """
        Build char_table: (vocab_size, MAX_WORD_LEN) int tensor.
        vocab: Vocabulary object with a .word2idx dict and .idx2word list/dict.
        Called once during model construction.
        """
        V = len(vocab)
        table = torch.zeros(V, self.MAX_WORD_LEN, dtype=torch.long)
        for idx in range(V):
            word = vocab.idx2word.get(idx, '') if hasattr(vocab, 'idx2word') else ''
            if not word and hasattr(vocab, 'idx_to_word'):
                word = vocab.idx_to_word.get(idx, '')
            for j, ch in enumerate(word[:self.MAX_WORD_LEN]):
                code = ord(ch) - 0x1F  # 0x20=' ' → 1, printable range → 1..95
                table[idx, j] = max(1, min(code, self.ALPHABET_SIZE - 1))
        self.char_table = table

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        """
        word_ids: (B, S) — word vocabulary indices
        returns:  (B, S, output_size)
        """
        B, S = word_ids.shape
        # Look up char IDs: (B, S, MAX_WORD_LEN)
        # char_table is built on CPU after model.to(device) — move to match word_ids
        char_ids = self.char_table.to(word_ids.device)[word_ids.view(-1)].view(B, S, self.MAX_WORD_LEN)

        # Embed chars and apply CNN per word
        char_ids_flat = char_ids.view(B * S, self.MAX_WORD_LEN)          # (B*S, L)
        emb = self.char_embed(char_ids_flat).transpose(1, 2)             # (B*S, emb, L)

        feats = []
        for conv in self.convs:
            c = F.relu(conv(emb))                                         # (B*S, F, L')
            c = F.max_pool1d(c, c.size(-1)).squeeze(-1)                  # (B*S, F)
            feats.append(c)

        out = torch.cat(feats, dim=-1)   # (B*S, output_size)
        return out.view(B, S, -1)        # (B, S, output_size)


# ── QuestionEncoder ──────────────────────────────────────────────────────────
class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout=0.5, pretrained_embeddings=None,
                 use_highway=False, use_char_cnn=False, vocab=None):
        """
        vocab          : Vocabulary object — required when use_char_cnn=True
        use_highway    : Tier 7B — highway skip connections between BiLSTM layers
        use_char_cnn   : Tier 7C — char-CNN embedding concatenated to word embed
        """
        super().__init__()
        self.use_highway  = use_highway
        self.use_char_cnn = use_char_cnn
        self.num_layers   = num_layers
        self.hidden_size  = hidden_size

        # Word embedding: supports GloVe pretrained vectors
        if pretrained_embeddings is not None:
            glove_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
            if glove_dim != embed_size:
                self.embed_proj = nn.Linear(glove_dim, embed_size)
            else:
                self.embed_proj = None
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0
            )
            self.embed_proj = None

        # Char-CNN (optional)
        bilstm_input = embed_size
        if use_char_cnn:
            self.char_cnn = CharCNNEmbedding(embed_dim=50, num_filters=100,
                                              kernel_sizes=(3, 4, 5))
            bilstm_input = embed_size + self.char_cnn.output_size  # word + char
            if vocab is not None:
                self.char_cnn.build_char_table(vocab)

        self.dropout = nn.Dropout(dropout)

        # BiLSTM layers — manual ModuleList when use_highway, else single nn.LSTM
        if use_highway and num_layers > 1:
            # One single-layer BiLSTM per level + highway between them
            self.bilstm_layers = nn.ModuleList()
            in_size = bilstm_input
            for _ in range(num_layers):
                self.bilstm_layers.append(nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_size // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                ))
                in_size = hidden_size   # next layer input = concat(fwd, bwd)
            self.highways = nn.ModuleList([
                HighwayLayer(hidden_size) for _ in range(num_layers - 1)
            ])
        else:
            self.lstm = nn.LSTM(
                input_size=bilstm_input,
                hidden_size=hidden_size // 2,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )

    def forward(self, questions):
        # 1. Word embedding with dropout
        embeds = self.dropout(self.embedding(questions))
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)
        # embeds: (B, L, embed_size)

        # 2. Char-CNN features (optional)
        if self.use_char_cnn:
            char_feats = self.char_cnn(questions)          # (B, L, 300)
            embeds = torch.cat([embeds, char_feats], dim=-1)  # (B, L, embed+300)

        # 3. BiLSTM (with or without highway)
        if self.use_highway and self.num_layers > 1:
            x = embeds
            all_hidden = []
            prev_out = None
            for i, bilstm in enumerate(self.bilstm_layers):
                output, (h, _) = bilstm(x)    # output: (B, L, hidden_size)
                all_hidden.append(h)           # h: (2, B, hidden_size//2)
                # Highway skip: blend new output with previous layer output
                if i < self.num_layers - 1:
                    if prev_out is not None and prev_out.shape == output.shape:
                        x = self.highways[i](output, prev_out)
                    else:
                        x = output   # first layer: no skip (shapes differ)
                    prev_out = output
                else:
                    x = output

            # q_feature: concat last-layer fwd + bwd hidden states
            h_last = all_hidden[-1]            # (2, B, hidden_size//2)
            q_feature = torch.cat([h_last[0], h_last[1]], dim=1)  # (B, hidden_size)
            q_hidden_states = x                # (B, L, hidden_size)
        else:
            output, (hidden, _) = self.lstm(embeds)
            q_feature = torch.cat([hidden[-2], hidden[-1]], dim=1)
            q_hidden_states = output

        return q_feature, q_hidden_states


# TESTING
if __name__ == "__main__":
    # Base test
    model = QuestionEncoder(vocab_size=7000, embed_size=512,
                            hidden_size=1024, num_layers=2, dropout=0.5)
    q = torch.randint(0, 7000, (4, 20))
    q_feat, q_hidden = model(q)
    print(f"Base BiLSTM: q_feat {q_feat.shape}, q_hidden {q_hidden.shape}")

    # Highway test
    model_hw = QuestionEncoder(vocab_size=7000, embed_size=512,
                               hidden_size=1024, num_layers=3,
                               dropout=0.5, use_highway=True)
    q_feat, q_hidden = model_hw(q)
    print(f"Highway BiLSTM (3L): q_feat {q_feat.shape}, q_hidden {q_hidden.shape}")

    # Char-CNN test (needs a vocab-like object)
    class MockVocab:
        def __init__(self, n):
            self.idx2word = {i: f"word{i}" for i in range(n)}
        def __len__(self): return len(self.idx2word)
    vocab = MockVocab(7000)

    model_char = QuestionEncoder(vocab_size=7000, embed_size=512,
                                 hidden_size=1024, num_layers=2,
                                 dropout=0.5, use_char_cnn=True, vocab=vocab)
    q_feat, q_hidden = model_char(q)
    print(f"CharCNN BiLSTM: q_feat {q_feat.shape}, q_hidden {q_hidden.shape}")
    print("All encoder_question tests PASSED")
