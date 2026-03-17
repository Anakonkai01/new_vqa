"""
benchmark_5070ti.py — Hardware Profiler for VQA-E Training
===========================================================
Finds the optimal batch size and training configuration for the current GPU.
Tests forward+backward throughput for Model E (ConvNeXt) and Model F (BUTD).

Usage
-----
  python src/scripts/benchmark_5070ti.py
  python src/scripts/benchmark_5070ti.py --model E --quick
  python src/scripts/benchmark_5070ti.py --model F
  python src/scripts/benchmark_5070ti.py --all
"""

import argparse
import time
import gc
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Constants matching real VQA-E inputs ──────────────────────────────────────

IMG_SIZE     = 224           # ConvNeXt input
SEQ_LEN_Q    = 16            # avg question length
SEQ_LEN_A    = 22            # avg "answer because explanation" length
VOCAB_Q      = 10000
VOCAB_A      = 8648
GLOVE_DIM    = 300
HIDDEN_DIM   = 1024
NUM_REGIONS  = 49            # ConvNeXt / ResNet spatial grid
NUM_BUTD     = 36            # BUTD average object proposals
BUTD_DIM     = 2048          # Faster R-CNN feature size

BATCH_SIZES  = [32, 64, 96, 128, 160, 192, 256]
WARMUP_ITERS = 3
BENCH_ITERS  = 10


# ── Synthetic data generators ─────────────────────────────────────────────────

def _rand_long(shape, high, device):
    return torch.randint(0, high, shape, device=device)


def make_batch_e(B, device):
    """Synthetic Model E batch: COCO image + question + target sequence."""
    imgs  = torch.randn(B, 3, IMG_SIZE, IMG_SIZE, device=device)
    q     = _rand_long((B, SEQ_LEN_Q), VOCAB_Q, device)
    a     = _rand_long((B, SEQ_LEN_A), VOCAB_A, device)
    return imgs, q, a


def make_batch_f(B, device):
    """Synthetic Model F batch: pre-extracted BUTD feats (variable-k padded)."""
    # Simulate variable k: each image has 24–36 real regions, padded to 36
    k_max = NUM_BUTD
    feats = torch.randn(B, k_max, BUTD_DIM, device=device)
    # Zero out last few rows to simulate padding
    for i in range(B):
        k_real = torch.randint(24, k_max + 1, (1,)).item()
        feats[i, k_real:] = 0.0
    img_mask = feats.abs().sum(dim=-1) > 0           # (B, k_max) bool
    q        = _rand_long((B, SEQ_LEN_Q), VOCAB_Q, device)
    a        = _rand_long((B, SEQ_LEN_A), VOCAB_A, device)
    return feats, q, a, img_mask


# ── Tiny surrogate model — same shape as real model but no pretrained weights ──

class _SurrogateEncoderE(nn.Module):
    """ConvNeXt-like spatial encoder (same output shape, no pretrained weights)."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 4, 4), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2), nn.GELU(),
            nn.Conv2d(128, HIDDEN_DIM, 3, 1, 1), nn.GELU(),
        )
        self.proj = nn.AdaptiveAvgPool2d((7, 7))   # → (B, 1024, 7, 7) → (B, 49, 1024)

    def forward(self, x):
        x = self.cnn(x)                            # (B, C, H, W)
        x = self.proj(x)                           # (B, C, 7, 7)
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 49, 1024)


class _SurrogateEncoderF(nn.Module):
    """BUTD feature encoder — just a linear projection."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(BUTD_DIM, HIDDEN_DIM)

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=-1)   # (B, k, 1024)


class _SurrogateDecoder(nn.Module):
    """
    Single-layer LSTM decoder with cross-attention and output head.
    Matches the real model's computational profile.
    """
    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(VOCAB_Q, GLOVE_DIM)
        self.q_enc   = nn.LSTM(GLOVE_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.q_proj  = nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)
        self.fusion  = nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)
        self.lstm    = nn.LSTMCell(GLOVE_DIM + 2 * HIDDEN_DIM, HIDDEN_DIM)
        # Cross-attention projections (Q/K/V)
        self.q_proj_mhca  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.k_proj_mhca  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj_mhca  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.kq_proj_mhca = nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)   # BiLSTM q_emb dim
        self.out         = nn.Linear(HIDDEN_DIM, VOCAB_A)

    def forward(self, img_features, questions, targets):
        B = questions.size(0)
        # Question encoding
        q_emb, (q_h, _) = self.q_enc(self.embed(questions))
        q_feat = self.q_proj(torch.cat([q_h[0], q_h[1]], dim=-1))
        img_mean = img_features.mean(1)
        # Init LSTM
        h = torch.tanh(self.fusion(torch.cat([img_mean, q_feat], dim=-1)))
        c = torch.zeros_like(h)
        # Decode (teacher forcing)
        a_emb = self.embed(targets[:, :-1])            # (B, T-1, E)
        logits = []
        for t in range(a_emb.size(1)):
            # MHCA (cross-attention from LSTM state to image)
            Q  = self.q_proj_mhca(h).unsqueeze(1)      # (B, 1, H)
            K  = self.k_proj_mhca(img_features)        # (B, S, H)
            V  = self.v_proj_mhca(img_features)
            sc = (Q @ K.transpose(1, 2)) / (HIDDEN_DIM ** 0.5)
            ctx = (sc.softmax(-1) @ V).squeeze(1)     # (B, H)
            # Q cross-attention (simplified — same as img)
            Qq  = self.q_proj_mhca(h).unsqueeze(1)
            Kq  = self.kq_proj_mhca(q_emb)
            Vq  = self.kq_proj_mhca(q_emb)
            sq  = (Qq @ Kq.transpose(1, 2)) / (HIDDEN_DIM ** 0.5)
            qctx = (sq.softmax(-1) @ Vq).squeeze(1)   # (B, H)
            x_t  = torch.cat([a_emb[:, t], ctx, qctx], dim=-1)
            h, c = self.lstm(x_t, (h, c))
            logits.append(self.out(h))
        return torch.stack(logits, dim=1)              # (B, T-1, V)


class SurrogateModelE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _SurrogateEncoderE()
        self.decoder = _SurrogateDecoder()

    def forward(self, imgs, questions, targets):
        feats = self.encoder(imgs)
        logits = self.decoder(feats, questions, targets)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_A), targets[:, 1:].reshape(-1), ignore_index=0)
        return loss


class SurrogateModelF(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _SurrogateEncoderF()
        self.decoder = _SurrogateDecoder()

    def forward(self, feats, questions, targets, img_mask=None):
        img_features = self.encoder(feats)
        if img_mask is not None:
            valid = img_mask.sum(1, keepdim=True).float().clamp(min=1)
            img_mean = (img_features * img_mask.unsqueeze(-1).float()).sum(1) / valid
        else:
            img_mean = img_features.mean(1)
        logits = self.decoder(img_features, questions, targets)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_A), targets[:, 1:].reshape(-1), ignore_index=0)
        return loss


# ── Benchmark runner ───────────────────────────────────────────────────────────

def _vram_mb():
    return torch.cuda.memory_allocated() / 1024 / 1024


def _peak_vram_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def benchmark_model(model_cls, batch_fn, batch_sizes, amp_dtype, device, label):
    print(f"\n{'═'*62}")
    print(f"  Benchmark: {label}")
    print(f"  AMP dtype : {amp_dtype}")
    print(f"{'═'*62}")
    print(f"  {'Batch':>6}  {'Samp/s':>9}  {'Peak VRAM':>10}  {'Status':<10}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*10}  {'─'*10}")

    results = []
    model = model_cls().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    for B in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        try:
            # warmup
            for _ in range(WARMUP_ITERS):
                batch = batch_fn(B, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    loss = model(*batch)
                loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            # timed run
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(BENCH_ITERS):
                batch = batch_fn(B, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    loss = model(*batch)
                loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            samp_per_sec = B * BENCH_ITERS / elapsed
            peak_mb      = _peak_vram_mb()
            status       = "✓ OK"
            results.append((B, samp_per_sec, peak_mb, True))
            print(f"  {B:>6}  {samp_per_sec:>9.0f}  {peak_mb:>8.0f} MB  {status}")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append((B, 0, 0, False))
            print(f"  {B:>6}  {'':>9}  {'':>10}  OOM ✗")

    del model, opt
    torch.cuda.empty_cache()
    return results


def print_recommendation(e_results, f_results):
    print(f"\n{'═'*62}")
    print("  RECOMMENDATION — RTX 5070 Ti (16 GB VRAM)")
    print(f"{'═'*62}")

    def best(results, target_vram_mb=13_000):
        """Largest batch that fits under target_vram_mb with highest throughput."""
        candidates = [(B, s, v) for B, s, v, ok in results if ok and v < target_vram_mb]
        if not candidates:
            candidates = [(B, s, v) for B, s, v, ok in results if ok]
        if not candidates:
            return None, None, None
        best_row = max(candidates, key=lambda x: x[1])   # max samp/s
        return best_row

    b_e, s_e, v_e = best(e_results)
    b_f, s_f, v_f = best(f_results)

    if b_e:
        accum_e = max(1, 128 // b_e)   # target effective_batch=128
        print(f"\n  Model E (ConvNeXt):")
        print(f"    --batch_size {b_e} --accum_steps {accum_e}")
        print(f"    Peak VRAM : {v_e:.0f} MB / 16384 MB ({v_e/163.84:.0f}% of 16 GB)")
        print(f"    Throughput: {s_e:.0f} samples/s")
        print(f"    Effective batch: {b_e * accum_e}")

    if b_f:
        accum_f = max(1, 192 // b_f)
        print(f"\n  Model F (BUTD — no CNN in training loop):")
        print(f"    --batch_size {b_f} --accum_steps {accum_f}")
        print(f"    Peak VRAM : {v_f:.0f} MB / 16384 MB ({v_f/163.84:.0f}% of 16 GB)")
        print(f"    Throughput: {s_f:.0f} samples/s")
        print(f"    Effective batch: {b_f * accum_f}")

    print(f"""
  Other recommended flags (always use these):
    --num_workers   12        # 28 CPU cores → 12 for I/O-bound loading
    --focal                   # SequenceFocalLoss (better than CE)
    --curriculum              # question-type progressive curriculum
    --coverage                # coverage mechanism (reduces repetition)
    --layer_norm              # LayerNorm LSTM (more stable training)
    --q_highway               # Highway BiLSTM (better gradients)
    --glove --glove_dim 300   # GloVe 840B embeddings
    --augment                 # RandAugment image augmentation
    --label_smoothing 0.1     # label smoothing
    --weight_decay 1e-5       # AdamW regularization
    BF16 AMP + torch.compile  # automatic — no flags needed (SM 12.0)
""")

    print(f"{'═'*62}")
    print("  SHELL SCRIPT COMMANDS")
    print(f"{'═'*62}")
    if b_e:
        print(f"""
  # Model E (ConvNeXt) — 4-phase full training:
  BATCH_SIZE={b_e} bash train_model_e.sh

  # Single phase:
  BATCH_SIZE={b_e} bash train_model_e.sh 1""")
    if b_f:
        print(f"""
  # Model F (BUTD) — 4-phase full training:
  BATCH_SIZE={b_f} bash train_model_f.sh""")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='VQA-E hardware benchmark for RTX 5070 Ti')
    parser.add_argument('--model',  choices=['E', 'F', 'all'], default='all')
    parser.add_argument('--quick',  action='store_true',
                        help='Only test batch sizes 64/128/192 (faster)')
    parser.add_argument('--target_vram_pct', type=float, default=80.0,
                        help='Target max VRAM utilization %% (default 80)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device('cuda')
    p = torch.cuda.get_device_properties(0)
    vram_gb = p.total_memory / 1024 / 1024 / 1024

    print(f"\nHardware Profile")
    print(f"{'─'*40}")
    print(f"  GPU   : {p.name}")
    print(f"  VRAM  : {vram_gb:.1f} GB")
    print(f"  SM    : {p.major}.{p.minor}  ({p.multi_processor_count} SMs)")
    print(f"  BF16  : {torch.cuda.is_bf16_supported()}")
    print(f"  SDPA  : {hasattr(F, 'scaled_dot_product_attention')}")
    import os
    print(f"  CPUs  : {os.cpu_count()}")
    print(f"  PyTorch: {torch.__version__}")

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    batch_sizes = [64, 128, 192] if args.quick else BATCH_SIZES

    e_results = f_results = []

    if args.model in ('E', 'all'):
        e_results = benchmark_model(
            SurrogateModelE, make_batch_e, batch_sizes, amp_dtype, device,
            label="Model E — ConvNeXt Spatial Encoder (training loop)")

    if args.model in ('F', 'all'):
        f_results = benchmark_model(
            SurrogateModelF, make_batch_f, batch_sizes, amp_dtype, device,
            label="Model F — BUTD Pre-extracted Features")

    print_recommendation(e_results, f_results)


if __name__ == '__main__':
    main()
