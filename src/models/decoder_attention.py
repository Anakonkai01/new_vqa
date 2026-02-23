"""
LSTMDecoder với Bahdanau (Additive) Attention — dùng cho Model C và D

──────────────────────────────────────────────────────
VẤN ĐỀ CỦA MODEL A/B (No Attention):
  - CNN nén toàn bộ ảnh thành 1 vector (batch, 1024)
  - Vector đó chỉ dùng để khởi tạo h_0 — decoder không nhìn lại ảnh nữa
  - Khi sinh token thứ 5, decoder không biết cần nhìn vào vùng nào của ảnh

GIẢI PHÁP — ATTENTION:
  - CNN giữ spatial: (batch, 49, 1024) — 49 vùng ảnh 7×7
  - Mỗi bước decode, tính "vùng nào quan trọng" dựa trên hidden state hiện tại
  - Tạo context vector = weighted sum của 49 vùng
  - Ghép context vào input của LSTM bước đó

──────────────────────────────────────────────────────
BAHDANAU ATTENTION (1 bước decode):

  1. energy = tanh( W_h(hidden) + W_img(image_regions) )
             shape: (batch, 49, attn_dim)

  2. alpha  = softmax( v(energy) )     ← attention weights, tổng = 1
              shape: (batch, 49)

  3. context = sum(alpha * image_regions, dim=1)
               shape: (batch, hidden_size)   ← weighted sum các vùng ảnh

  4. lstm input = concat(embed, context)
                  shape: (batch, 1, embed_size + hidden_size)

  5. output, hidden = lstm(lstm_input, hidden)

  6. logit = fc(output)

──────────────────────────────────────────────────────
TRONG VQA:
  - Query  = decoder hidden state (cái decoder đang "nghĩ" đến)
  - Key    = image regions (49 vùng ảnh)
  - Value  = image regions (Key = Value trong Bahdanau)
  - Output = context vector (tổng hợp info từ vùng ảnh liên quan)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Tính attention weights và context vector cho 1 bước decode.

    Công thức:
        energy = tanh(W_h(hidden) + W_img(img_features))  # (batch, 49, attn_dim)
        alpha  = softmax(v(energy))                        # (batch, 49)
        context = (alpha.unsqueeze(2) * img_features).sum(1)  # (batch, hidden_size)
    """

    def __init__(self, hidden_size, attn_dim=512):
        super().__init__()

        # W_h: project decoder hidden → attn_dim
        # hidden_size: chiều của decoder hidden state (num_layers, batch, hidden)
        # Chỉ lấy layer cuối → (batch, hidden_size)
        self.W_h = nn.Linear(hidden_size, attn_dim)

        # W_img: project mỗi vùng ảnh → attn_dim
        self.W_img = nn.Linear(hidden_size, attn_dim)

        # v: project attn_dim → scalar score cho mỗi vùng
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden, img_features):
        """
        hidden      : (batch, hidden_size) — hidden state layer cuối của decoder
        img_features: (batch, 49, hidden_size) — spatial features từ CNN

        returns:
          context : (batch, hidden_size)
          alpha   : (batch, 49) — attention weights để visualize sau này
        """

        # Project hidden: (batch, hidden_size) → (batch, attn_dim)
        # unsqueeze(1) để broadcast với img_features (batch, 49, attn_dim)
        h_proj = self.W_h(hidden).unsqueeze(1)        # (batch, 1, attn_dim)

        # Project từng vùng ảnh: (batch, 49, hidden_size) → (batch, 49, attn_dim)
        img_proj = self.W_img(img_features)            # (batch, 49, attn_dim)

        # Cộng 2 phép chiếu → broadcast: (batch, 1, attn_dim) + (batch, 49, attn_dim)
        # = (batch, 49, attn_dim)
        energy = torch.tanh(h_proj + img_proj)         # (batch, 49, attn_dim)

        # v cho ra 1 scalar mỗi vùng → squeeze chiều cuối
        scores = self.v(energy).squeeze(-1)            # (batch, 49)

        # Softmax → attention weights, tổng = 1
        alpha = F.softmax(scores, dim=1)               # (batch, 49)

        # Context = weighted sum: nhân alpha với img_features rồi sum qua 49 vùng
        # alpha.unsqueeze(2): (batch, 49, 1) — để broadcast với (batch, 49, hidden_size)
        context = (alpha.unsqueeze(2) * img_features).sum(dim=1)  # (batch, hidden_size)

        return context, alpha


class LSTMDecoderWithAttention(nn.Module):
    """
    LSTM Decoder với Bahdanau Attention.

    Khác LSTMDecoder (no attention) ở chỗ:
      - LSTM input = concat(embed, context) thay vì chỉ embed
        → input_size = embed_size + hidden_size
      - Mỗi bước decode gọi BahdanauAttention để tạo context vector mới
      - Có thêm method decode_step() cho inference (autoregressive)
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 attn_dim=512, dropout=0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Embedding như cũ
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # Attention module
        self.attention = BahdanauAttention(hidden_size, attn_dim)

        # LSTM: input_size = embed_size + hidden_size
        # (ghép embed token với context vector)
        self.lstm = nn.LSTM(
            input_size=embed_size + hidden_size,  # ← điểm khác so với LSTMDecoder
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc      = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_hidden, img_features, target_seq):
        """
        Training mode — Teacher Forcing: chạy toàn bộ sequence 1 lần
        (vẫn cần loop vì attention cần hidden state của từng bước)

        encoder_hidden: (num_layers, batch, hidden_size) — từ fusion như Model A/B
        img_features  : (batch, 49, hidden_size) — spatial CNN output
        target_seq    : (batch, max_len) — [<start>, w1, w2, ...]

        returns: logits (batch, max_len, vocab_size)
        """
        batch_size = target_seq.size(0)
        max_len    = target_seq.size(1)

        # Embed toàn bộ sequence trước
        embeds = self.dropout(self.embedding(target_seq))  # (batch, max_len, embed_size)

        # Khởi tạo hidden từ encoder (giống Model A/B)
        hidden = encoder_hidden  # (num_layers, batch, hidden_size) tuple (h, c)

        logits_list = []

        # Loop từng bước — cần thiết vì attention phụ thuộc hidden bước trước
        for t in range(max_len):
            # embed của token tại bước t
            embed_t = embeds[:, t, :]             # (batch, embed_size)

            # Lấy hidden state layer cuối để tính attention
            # hidden là tuple (h, c), h shape (num_layers, batch, hidden_size)
            h_top = hidden[0][-1]                 # (batch, hidden_size)

            # Tính context vector từ attention
            context, _ = self.attention(h_top, img_features)  # (batch, hidden_size)

            # Ghép embed với context
            lstm_input = torch.cat([embed_t, context], dim=1)  # (batch, embed+hidden)
            lstm_input = lstm_input.unsqueeze(1)               # (batch, 1, embed+hidden)

            # 1 bước LSTM
            output, hidden = self.lstm(lstm_input, hidden)
            # output: (batch, 1, hidden_size)

            # Project ra vocab
            logit = self.fc(output.squeeze(1))    # (batch, vocab_size)
            logits_list.append(logit)

        # Stack tất cả bước lại
        logits = torch.stack(logits_list, dim=1)  # (batch, max_len, vocab_size)
        return logits

    def decode_step(self, token, hidden, img_features):
        """
        Inference mode — 1 bước autoregressive (dùng trong inference.py)

        token       : (batch, 1) — token hiện tại
        hidden      : tuple (h, c), h shape (num_layers, batch, hidden_size)
        img_features: (batch, 49, hidden_size)

        returns:
          logit  : (batch, vocab_size)
          hidden : tuple mới sau bước này
          alpha  : (batch, 49) — attention weights (để visualize nếu cần)
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

    # Giả lập encoder_hidden (h, c) từ fusion
    h = torch.zeros(NUM_LAYERS, BATCH, HIDDEN_SIZE)
    c = torch.zeros(NUM_LAYERS, BATCH, HIDDEN_SIZE)

    # Giả lập spatial image features từ CNN
    img_features = torch.randn(BATCH, NUM_REGIONS, HIDDEN_SIZE)

    # Giả lập target sequence (teacher forcing)
    target = torch.randint(0, VOCAB_SIZE, (BATCH, MAX_LEN))

    # Forward
    logits = decoder((h, c), img_features, target)
    print(f"logits shape: {logits.shape}")  # expect (4, 10, 3000)

    # Test decode_step
    token = torch.randint(0, VOCAB_SIZE, (BATCH, 1))
    logit, hidden_new, alpha = decoder.decode_step(token, (h, c), img_features)
    print(f"decode_step logit: {logit.shape}")   # expect (4, 3000)
    print(f"attention alpha  : {alpha.shape}")   # expect (4, 49)
