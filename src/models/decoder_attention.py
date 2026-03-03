"""
LSTM Decoder with Bahdanau (Additive) Attention — used for Model C and D

──────────────────────────────────────────────────────
PROBLEM WITH MODEL A/B (No Attention):
  - CNN compresses the entire image into 1 vector (batch, 1024)
  - That vector is only used to initialize h_0 — the decoder never looks at the image again
  - When generating token 5, the decoder has no way to know which image region to focus on

SOLUTION — ATTENTION:
  - CNN keeps spatial info: (batch, 49, 1024) — 49 image regions from a 7x7 grid
  - At each decode step, compute "which regions matter" based on the current hidden state
  - Build a context vector = weighted sum of the 49 regions
  - Concatenate context with the token embedding as LSTM input

──────────────────────────────────────────────────────
BAHDANAU ATTENTION (one decode step):

  1. energy = tanh( W_h(hidden) + W_img(image_regions) )
             shape: (batch, 49, attn_dim)

  2. alpha  = softmax( v(energy) )     <- attention weights, sum = 1
              shape: (batch, 49)

  3. context = sum(alpha * image_regions, dim=1)
               shape: (batch, hidden_size)   <- weighted sum of image regions

  4. lstm input = concat(embed, context)
                  shape: (batch, 1, embed_size + hidden_size)

  5. output, hidden = lstm(lstm_input, hidden)

  6. logit = fc(output)

──────────────────────────────────────────────────────
IN VQA TERMS:
  - Query  = decoder hidden state (what the decoder is currently "thinking about")
  - Key    = image regions (49 spatial regions)
  - Value  = image regions (Key == Value in Bahdanau)
  - Output = context vector (aggregated info from the relevant image regions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Computes attention weights and context vector for one decode step.
    Supports optional Coverage Mechanism (See et al. 2017, "Get To The Point").

    Formula:
        energy = tanh(W_h(hidden) + W_img(img_features) [+ W_cov(coverage)])  # (batch, N, attn_dim)
        alpha  = softmax(v(energy))                        # (batch, N)
        context = (alpha.unsqueeze(2) * img_features).sum(1)  # (batch, hidden_size)
    """

    def __init__(self, hidden_size, attn_dim=512, use_coverage=False):
        super().__init__()

        self.use_coverage = use_coverage

        # W_h: project decoder hidden -> attn_dim
        self.W_h = nn.Linear(hidden_size, attn_dim)

        # W_img: project each image region -> attn_dim
        self.W_img = nn.Linear(hidden_size, attn_dim)

        # Coverage: project cumulative attention -> attn_dim (1-D per position)
        if use_coverage:
            self.W_cov = nn.Linear(1, attn_dim, bias=False)

        # v: project attn_dim -> scalar score per region
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden, img_features, coverage=None):
        """
        hidden      : (batch, hidden_size)
        img_features: (batch, N, hidden_size)  -- N spatial regions (49 for images)
        coverage    : (batch, N) or None -- cumulative attention from previous steps

        returns:
          context  : (batch, hidden_size)
          alpha    : (batch, N)
        """

        h_proj   = self.W_h(hidden).unsqueeze(1)       # (batch, 1, attn_dim)
        img_proj = self.W_img(img_features)             # (batch, N, attn_dim)

        energy = h_proj + img_proj                      # (batch, N, attn_dim)

        # Add coverage signal if enabled
        if self.use_coverage and coverage is not None:
            # coverage: (batch, N) -> (batch, N, 1) -> (batch, N, attn_dim)
            energy = energy + self.W_cov(coverage.unsqueeze(-1))

        energy = torch.tanh(energy)                     # (batch, N, attn_dim)
        scores = self.v(energy).squeeze(-1)             # (batch, N)
        alpha  = F.softmax(scores, dim=1)               # (batch, N)

        context = (alpha.unsqueeze(2) * img_features).sum(dim=1)  # (batch, hidden_size)

        return context, alpha


class LSTMDecoderWithAttention(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention.

    Differences from LSTMDecoder (no attention):
      - LSTM input = concat(embed, context) instead of just embed
        -> input_size = embed_size + hidden_size
      - At each decode step, BahdanauAttention is called to produce a new context vector
      - Added decode_step() method for autoregressive inference
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 attn_dim=512, dropout=0.5, pretrained_embeddings=None,
                 use_coverage=False):
        super().__init__()

        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.use_coverage = use_coverage

        # Embedding: supports GloVe pretrained vectors
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
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.embed_proj = None

        # Attention modules — dual attention over image AND question
        # Coverage only on image attention (spatial repetition is the main issue)
        self.img_attention = BahdanauAttention(hidden_size, attn_dim, use_coverage=use_coverage)
        self.q_attention   = BahdanauAttention(hidden_size, attn_dim)

        # LSTM: input_size = embed_size + hidden_size * 2
        # (token embedding + image context + question context)
        self.lstm = nn.LSTM(
            input_size=embed_size + hidden_size * 2,  # <- dual attention
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Weight Tying: hidden → embed_dim (projection) → vocab (tied with embedding)
        actual_embed_dim = self.embedding.embedding_dim
        self.out_proj = nn.Linear(hidden_size, actual_embed_dim)
        self.fc       = nn.Linear(actual_embed_dim, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight  # tie weights

        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_hidden, img_features, q_hidden_states, target_seq):
        """
        Training mode — Teacher Forcing with dual attention (image + question).

        encoder_hidden : (num_layers, batch, hidden_size) -- from fusion
        img_features   : (batch, 49, hidden_size) -- spatial CNN output
        q_hidden_states: (batch, q_len, hidden_size) -- question encoder all timesteps
        target_seq     : (batch, max_len) -- [<start>, w1, w2, ...]

        returns:
          logits        : (batch, max_len, vocab_size)
          coverage_loss : scalar tensor (0.0 if coverage disabled)
        """
        # Embed the full sequence upfront
        batch   = target_seq.size(0)
        max_len = target_seq.size(1)
        embeds = self.dropout(self.embedding(target_seq))  # (batch, max_len, glove_dim or embed_size)
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)                # (batch, max_len, embed_size)

        # Initialize hidden from encoder (same as Model A/B)
        hidden = encoder_hidden  # tuple (h, c), each (num_layers, batch, hidden_size)

        # Coverage: cumulative attention over image regions
        num_regions = img_features.size(1)  # 49
        coverage = img_features.new_zeros(batch, num_regions) if self.use_coverage else None
        cov_loss = img_features.new_zeros(1)

        logits_list = []

        # Step-by-step loop required because attention depends on the previous hidden state
        for t in range(max_len):
            embed_t = embeds[:, t, :]             # (batch, embed_size)

            # Take the last layer's hidden state to compute attention
            h_top = hidden[0][-1]                 # (batch, hidden_size)

            # Dual attention: attend over both image regions AND question tokens
            img_context, img_alpha = self.img_attention(h_top, img_features, coverage)  # (batch, hidden_size)
            q_context, _           = self.q_attention(h_top, q_hidden_states)           # (batch, hidden_size)

            # Accumulate coverage loss: sum_i min(alpha_i, coverage_i)
            # Penalises attending to already-attended regions (See et al. 2017)
            if self.use_coverage:
                cov_loss = cov_loss + torch.min(img_alpha, coverage).sum(dim=1).mean()
                coverage = coverage + img_alpha  # update cumulative attention

            # Concatenate token embedding with both context vectors
            lstm_input = torch.cat([embed_t, img_context, q_context], dim=1)  # (batch, embed+hidden*2)
            lstm_input = lstm_input.unsqueeze(1)               # (batch, 1, embed+hidden*2)

            # Single LSTM step
            output, hidden = self.lstm(lstm_input, hidden)     # output: (batch, 1, hidden_size)

            logit = self.fc(self.out_proj(output.squeeze(1)))    # (batch, vocab_size)
            logits_list.append(logit)

        # Stack all steps
        logits = torch.stack(logits_list, dim=1)  # (batch, max_len, vocab_size)

        # Normalize coverage loss by number of decode steps
        coverage_loss = cov_loss / max_len if self.use_coverage else cov_loss.squeeze()

        return logits, coverage_loss

    def decode_step(self, token, hidden, img_features, q_hidden_states, coverage=None):
        """
        Inference mode — one autoregressive step (used in inference.py)

        token           : (batch, 1) -- current token
        hidden          : tuple (h, c), h shape (num_layers, batch, hidden_size)
        img_features    : (batch, 49, hidden_size)
        q_hidden_states : (batch, q_len, hidden_size)
        coverage        : (batch, 49) or None -- cumulative attention (for coverage)

        returns:
          logit    : (batch, vocab_size)
          hidden   : updated tuple after this step
          alpha    : (batch, 49) -- image attention weights (for visualization)
          coverage : (batch, 49) -- updated cumulative attention (None if coverage disabled)
        """
        embed  = self.dropout(self.embedding(token))  # (batch, 1, glove_dim or embed_size)
        if self.embed_proj is not None:
            embed = self.embed_proj(embed)              # (batch, 1, embed_size)
        h_top  = hidden[0][-1]                         # (batch, hidden_size)

        img_context, img_alpha = self.img_attention(h_top, img_features, coverage)  # (batch, hidden_size)
        q_context, _           = self.q_attention(h_top, q_hidden_states)           # (batch, hidden_size)

        # Update coverage: initialize on first step, accumulate thereafter
        if self.use_coverage:
            if coverage is None:
                coverage = torch.zeros_like(img_alpha)
            coverage = coverage + img_alpha

        lstm_input = torch.cat([embed.squeeze(1), img_context, q_context], dim=1).unsqueeze(1)
        # (batch, 1, embed_size + hidden_size * 2)

        output, hidden = self.lstm(lstm_input, hidden)
        logit = self.fc(self.out_proj(output.squeeze(1)))  # (batch, vocab_size)

        return logit, hidden, img_alpha, coverage


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    BATCH       = 4
    VOCAB_SIZE  = 3000
    EMBED_SIZE  = 512
    HIDDEN_SIZE = 1024
    NUM_LAYERS  = 2
    MAX_LEN     = 10
    NUM_REGIONS = 49  # 7×7

    decoder = LSTMDecoderWithAttention(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    )

    # Simulate encoder_hidden (h, c) from fusion
    h = torch.zeros(NUM_LAYERS, BATCH, HIDDEN_SIZE)
    c = torch.zeros(NUM_LAYERS, BATCH, HIDDEN_SIZE)

    # Simulate spatial image features from CNN
    img_features = torch.randn(BATCH, NUM_REGIONS, HIDDEN_SIZE)

    # Simulate question hidden states from BiLSTM encoder
    Q_LEN = 15
    q_hidden = torch.randn(BATCH, Q_LEN, HIDDEN_SIZE)

    # Simulate target sequence (teacher forcing)
    target = torch.randint(0, VOCAB_SIZE, (BATCH, MAX_LEN))

    # Forward (returns logits + coverage_loss)
    logits, cov_loss = decoder((h, c), img_features, q_hidden, target)
    print(f"logits shape       : {logits.shape}")       # expect (4, 10, 3000)
    print(f"coverage_loss      : {cov_loss.item():.4f}")

    # Test decode_step (returns logit, hidden, alpha, coverage)
    token = torch.randint(0, VOCAB_SIZE, (BATCH, 1))
    logit, hidden_new, alpha, cov = decoder.decode_step(token, (h, c), img_features, q_hidden)
    print(f"decode_step logit  : {logit.shape}")         # expect (4, 3000)
    print(f"attention alpha    : {alpha.shape}")          # expect (4, 49)
    print(f"coverage           : {cov}")                 # None (coverage disabled in this test)
