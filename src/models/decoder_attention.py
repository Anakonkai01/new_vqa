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

    Formula:
        energy = tanh(W_h(hidden) + W_img(img_features))  # (batch, 49, attn_dim)
        alpha  = softmax(v(energy))                        # (batch, 49)
        context = (alpha.unsqueeze(2) * img_features).sum(1)  # (batch, hidden_size)
    """

    def __init__(self, hidden_size, attn_dim=512):
        super().__init__()

        # W_h: project decoder hidden -> attn_dim
        # We take only the last layer's hidden state -> (batch, hidden_size)
        self.W_h = nn.Linear(hidden_size, attn_dim)

        # W_img: project each image region -> attn_dim
        self.W_img = nn.Linear(hidden_size, attn_dim)

        # v: project attn_dim -> scalar score per region
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden, img_features):
        """
        hidden      : (batch, hidden_size) -- last-layer hidden state of the decoder
        img_features: (batch, 49, hidden_size) -- spatial features from CNN

        returns:
          context : (batch, hidden_size)
          alpha   : (batch, 49) -- attention weights (useful for visualization)
        """

        # Project hidden: (batch, hidden_size) -> (batch, attn_dim)
        # unsqueeze(1) to broadcast over img_features (batch, 49, attn_dim)
        h_proj = self.W_h(hidden).unsqueeze(1)        # (batch, 1, attn_dim)

        # Project each image region: (batch, 49, hidden_size) -> (batch, 49, attn_dim)
        img_proj = self.W_img(img_features)            # (batch, 49, attn_dim)

        # Sum the two projections via broadcast: (batch, 1, attn_dim) + (batch, 49, attn_dim)
        # -> (batch, 49, attn_dim)
        energy = torch.tanh(h_proj + img_proj)         # (batch, 49, attn_dim)

        # v produces one scalar per region -> squeeze last dim
        scores = self.v(energy).squeeze(-1)            # (batch, 49)

        # Softmax -> attention weights (sum to 1)
        alpha = F.softmax(scores, dim=1)               # (batch, 49)

        # Context = weighted sum: multiply alpha with img_features and sum over 49 regions
        # alpha.unsqueeze(2): (batch, 49, 1) to broadcast with (batch, 49, hidden_size)
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
                 attn_dim=512, dropout=0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Embedding (same as LSTMDecoder)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # Attention module
        self.attention = BahdanauAttention(hidden_size, attn_dim)

        # LSTM: input_size = embed_size + hidden_size
        # (token embedding concatenated with context vector)
        self.lstm = nn.LSTM(
            input_size=embed_size + hidden_size,  # <- key difference from LSTMDecoder
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc      = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_hidden, img_features, target_seq):
        """
        Training mode — Teacher Forcing: runs the full sequence in one pass
        (still requires a loop because attention depends on the previous hidden state)

        encoder_hidden: (num_layers, batch, hidden_size) -- from fusion (same as Model A/B)
        img_features  : (batch, 49, hidden_size) -- spatial CNN output
        target_seq    : (batch, max_len) -- [<start>, w1, w2, ...]

        returns: logits (batch, max_len, vocab_size)
        """
        # Embed the full sequence upfront
        max_len = target_seq.size(1)
        embeds = self.dropout(self.embedding(target_seq))  # (batch, max_len, embed_size)

        # Initialize hidden from encoder (same as Model A/B)
        hidden = encoder_hidden  # tuple (h, c), each (num_layers, batch, hidden_size)

        logits_list = []

        # Step-by-step loop required because attention depends on the previous hidden state
        for t in range(max_len):
            embed_t = embeds[:, t, :]             # (batch, embed_size)

            # Take the last layer's hidden state to compute attention
            h_top = hidden[0][-1]                 # (batch, hidden_size)

            # Compute context vector via attention
            context, _ = self.attention(h_top, img_features)  # (batch, hidden_size)

            # Concatenate token embedding with context vector
            lstm_input = torch.cat([embed_t, context], dim=1)  # (batch, embed+hidden)
            lstm_input = lstm_input.unsqueeze(1)               # (batch, 1, embed+hidden)

            # Single LSTM step
            output, hidden = self.lstm(lstm_input, hidden)     # output: (batch, 1, hidden_size)

            logit = self.fc(output.squeeze(1))    # (batch, vocab_size)
            logits_list.append(logit)

        # Stack all steps
        logits = torch.stack(logits_list, dim=1)  # (batch, max_len, vocab_size)
        return logits

    def decode_step(self, token, hidden, img_features):
        """
        Inference mode — one autoregressive step (used in inference.py)

        token       : (batch, 1) -- current token
        hidden      : tuple (h, c), h shape (num_layers, batch, hidden_size)
        img_features: (batch, 49, hidden_size)

        returns:
          logit  : (batch, vocab_size)
          hidden : updated tuple after this step
          alpha  : (batch, 49) -- attention weights (useful for visualization)
        """
        embed  = self.dropout(self.embedding(token))  # (batch, 1, embed_size)
        h_top  = hidden[0][-1]                         # (batch, hidden_size)

        context, alpha = self.attention(h_top, img_features)  # (batch, hidden_size)

        lstm_input = torch.cat([embed.squeeze(1), context], dim=1).unsqueeze(1)
        # (batch, 1, embed_size + hidden_size)

        output, hidden = self.lstm(lstm_input, hidden)
        logit = self.fc(output.squeeze(1))  # (batch, vocab_size)

        return logit, hidden, alpha


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

    # Simulate target sequence (teacher forcing)
    target = torch.randint(0, VOCAB_SIZE, (BATCH, MAX_LEN))

    # Forward
    logits = decoder((h, c), img_features, target)
    print(f"logits shape: {logits.shape}")  # expect (4, 10, 3000)

    # Test decode_step
    token = torch.randint(0, VOCAB_SIZE, (BATCH, 1))
    logit, hidden_new, alpha = decoder.decode_step(token, (h, c), img_features)
    print(f"decode_step logit: {logit.shape}")   # expect (4, 3000)
    print(f"attention alpha  : {alpha.shape}")   # expect (4, 49)
