"""
LSTM Decoder (Models A/B — no attention)

flow:
  initial_hidden (num_layers, batch, hidden_size) from fusion
  input token: (batch, 1): 1 token each step
  embedding: (batch, 1, embed_size)
  lstm output: (batch, 1, hidden_size)
  fc (linear) (batch, 1, vocab_size): logits for each token

Tier-1 additions (use_layer_norm=True):
  - LayerNormLSTMCell: LayerNorm on each gate's pre-activation
  - WeightDrop (DropConnect): zeros hidden-to-hidden weights during training
  - HighwayLayer: gated skip connections between LSTM layers
  All three default to False for backward compatibility with Models A/B.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Tier 1 helpers ─────────────────────────────────────────────────────────────

class LayerNormLSTMCell(nn.Module):
    """
    LSTM cell with LayerNorm on each gate's pre-activation.
    Stabilizes training on variable-length sequences (better than BatchNorm).
    Drop-in per-step replacement for nn.LSTMCell.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.ln_i   = nn.LayerNorm(hidden_size)
        self.ln_f   = nn.LayerNorm(hidden_size)
        self.ln_g   = nn.LayerNorm(hidden_size)
        self.ln_o   = nn.LayerNorm(hidden_size)
        self.ln_c   = nn.LayerNorm(hidden_size)

    def forward(self, x, hx):
        h, c = hx
        gates   = self.linear(torch.cat([x, h], dim=-1))        # (B, 4H)
        i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=-1)
        i       = torch.sigmoid(self.ln_i(i_raw))
        f       = torch.sigmoid(self.ln_f(f_raw))
        g       = torch.tanh(self.ln_g(g_raw))
        o       = torch.sigmoid(self.ln_o(o_raw))
        c_new   = f * c + i * g
        h_new   = o * torch.tanh(self.ln_c(c_new))
        return h_new, c_new


class WeightDrop(nn.Module):
    """
    AWD-LSTM DropConnect: zeros weights in hidden-to-hidden matrices during training.
    Regularizes without breaking LSTM's temporal memory flow (unlike activation dropout).
    """
    def __init__(self, module, weights, dropout=0.5):
        super().__init__()
        self.module  = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        for w_name in self.weights:
            raw = getattr(self.module, w_name)
            del self.module._parameters[w_name]
            self.module.register_parameter(w_name + '_raw', nn.Parameter(raw.data))

    def _setweights(self):
        for w_name in self.weights:
            raw = getattr(self.module, w_name + '_raw')
            w   = F.dropout(raw, p=self.dropout, training=self.training)
            setattr(self.module, w_name, w)

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module.forward(*args, **kwargs)


class HighwayLayer(nn.Module):
    """
    Gated highway skip connection between LSTM layers.
    h_out = gate * lstm_out + (1 - gate) * h_in
    Allows gradients to bypass layers, enabling deeper LSTMs.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_out, h_in):
        g = torch.sigmoid(self.gate(h_in))
        return g * lstm_out + (1.0 - g) * h_in


class LayerNormLSTMStack(nn.Module):
    """
    Multi-layer LSTM using LayerNormLSTMCells with optional highway connections.
    Replaces nn.LSTM when use_layer_norm=True.
    Processes a full sequence, returns (output_seq, (h_n, c_n)) matching nn.LSTM API.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, use_highway=True):
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size
        self.use_highway = use_highway
        self.drop        = nn.Dropout(dropout)

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size
            self.cells.append(LayerNormLSTMCell(in_sz, hidden_size))

        if use_highway and num_layers > 1:
            # One highway per inter-layer transition
            self.highways = nn.ModuleList(
                [HighwayLayer(hidden_size) for _ in range(num_layers - 1)]
            )

    def forward(self, x, hx=None):
        """
        x:  (batch, seq_len, input_size)  — batch_first=True
        hx: tuple (h_0, c_0), each (num_layers, batch, hidden_size) or None
        returns: output (batch, seq_len, hidden_size), (h_n, c_n)
        """
        B, T, _ = x.shape
        if hx is None:
            h = x.new_zeros(self.num_layers, B, self.hidden_size)
            c = x.new_zeros(self.num_layers, B, self.hidden_size)
        else:
            h, c = hx[0], hx[1]

        # Unpack layer hidden states
        h_list = [h[i] for i in range(self.num_layers)]
        c_list = [c[i] for i in range(self.num_layers)]

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                inp = x_t if layer_idx == 0 else h_out
                h_new, c_new = cell(inp, (h_list[layer_idx], c_list[layer_idx]))
                h_out = self.drop(h_new) if layer_idx < self.num_layers - 1 else h_new
                # Highway: skip connection from layer input to layer output
                if self.use_highway and layer_idx > 0:
                    h_out = self.highways[layer_idx - 1](h_out, inp)
                h_list[layer_idx] = h_new
                c_list[layer_idx] = c_new
            outputs.append(h_out)

        output = torch.stack(outputs, dim=1)                          # (B, T, hidden)
        h_n    = torch.stack(h_list, dim=0)                           # (num_layers, B, hidden)
        c_n    = torch.stack(c_list, dim=0)
        return output, (h_n, c_n)


# ── Main decoder ───────────────────────────────────────────────────────────────

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout=0.5, pretrained_embeddings=None,
                 use_layer_norm=False, use_dropconnect=False):
        """
        use_layer_norm  : replace nn.LSTM with LayerNormLSTMStack (Tier 1A + 1C)
        use_dropconnect : apply WeightDrop to nn.LSTM hidden-to-hidden weights (Tier 1B)
                          Only applies when use_layer_norm=False (standard nn.LSTM path).
        """
        super().__init__()
        self.use_layer_norm = use_layer_norm

        # Embedding
        if pretrained_embeddings is not None:
            glove_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0)
            self.embed_proj = nn.Linear(glove_dim, embed_size) if glove_dim != embed_size else None
        else:
            self.embedding  = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.embed_proj = None

        self.dropout = nn.Dropout(dropout)

        if use_layer_norm:
            self.lstm = LayerNormLSTMStack(
                input_size=embed_size, hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout, use_highway=True,
            )
        else:
            _lstm = nn.LSTM(
                input_size=embed_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            if use_dropconnect and num_layers >= 1:
                self.lstm = WeightDrop(_lstm, ['weight_hh_l0'], dropout=0.3)
            else:
                self.lstm = _lstm

        # Output projection + weight tying
        if pretrained_embeddings is not None:
            self.out_proj = nn.Linear(hidden_size, embed_size)
            self.fc       = nn.Linear(embed_size, vocab_size, bias=False)
        else:
            actual_embed_dim = self.embedding.embedding_dim
            self.out_proj    = nn.Linear(hidden_size, actual_embed_dim)
            self.fc          = nn.Linear(actual_embed_dim, vocab_size, bias=False)
            self.fc.weight   = self.embedding.weight   # weight tying

    def forward(self, encoder_hidden, target_seq):
        """
        encoder_hidden: tuple (h, c), each (num_layers, batch, hidden_size)
        target_seq:     (batch, max_len)
        returns: logits (batch, max_len, vocab_size)
        """
        embeds = self.dropout(self.embedding(target_seq))
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)

        outputs, _ = self.lstm(embeds, encoder_hidden)
        logits = self.fc(self.out_proj(outputs))
        return logits


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Standard LSTM ===")
    decoder = LSTMDecoder(vocab_size=3000, embed_size=512, hidden_size=1024, num_layers=2)
    h = torch.zeros(2, 4, 1024)
    c = torch.zeros(2, 4, 1024)
    target = torch.randint(0, 3000, (4, 10))
    logits = decoder((h, c), target)
    print(f"logits: {logits.shape}")   # (4, 10, 3000)

    print("=== LayerNorm LSTM (Tier 1) ===")
    decoder_ln = LSTMDecoder(vocab_size=3000, embed_size=512, hidden_size=1024,
                             num_layers=2, use_layer_norm=True)
    logits_ln = decoder_ln((h, c), target)
    print(f"logits: {logits_ln.shape}")  # (4, 10, 3000)
