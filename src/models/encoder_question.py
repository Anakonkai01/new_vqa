import torch
import torch.nn as nn
import torch.nn.functional as F


"""
input: (batch, max_len)
embedding: (batch, max_len, embed_size)
LSTM: (batch, max_len, hidden_size)

Attention Pooling (thay vì chỉ lấy hidden[-1]):
  - Score mỗi token: attn(output) → (batch, max_len)
  - Mask <pad> tokens (index 0) → -inf trước softmax
  - alpha = softmax(scores) → (batch, max_len)
  - q_feature = sum(alpha * output) → (batch, hidden_size)

Lợi ích: tổng hợp toàn bộ question tokens thay vì chỉ token cuối,
giúp encoder nắm bắt thông tin từ tất cả từ trong câu hỏi.
"""


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention: score each token position → weighted sum over all tokens
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, questions):
        # questions: (batch, max_len)

        # 1. embedding
        embeds = self.embedding(questions)
        # embeds: (batch, max_len, embed_size)

        # 2. lstm — output contains hidden state at every position
        output, (hidden, cell) = self.lstm(embeds)
        # output: (batch, max_len, hidden_size) — all positions
        # hidden: (num_layers, batch, hidden_size)

        # 3. attention pooling over all token positions
        scores = self.attn(output).squeeze(-1)          # (batch, max_len)

        # mask <pad> tokens (index 0) to prevent them from contributing
        mask = (questions == 0)                          # True where padding
        scores = scores.masked_fill(mask, float('-inf'))

        alpha = F.softmax(scores, dim=1)                 # (batch, max_len)

        # weighted sum: aggregate all token representations
        q_feature = (alpha.unsqueeze(2) * output).sum(dim=1)  # (batch, hidden_size)

        return q_feature


# TESTING
if __name__ == "__main__":
    model = QuestionEncoder(vocab_size=7000, embed_size=512,
                            hidden_size=1024, num_layers=2, dropout=0.5)

    q = torch.randint(0, 7000, (4, 20))
    # Add some padding to test mask
    q[0, 15:] = 0
    q[1, 10:] = 0

    out = model(q)
    print(out.shape)  # expect (4, 1024)
