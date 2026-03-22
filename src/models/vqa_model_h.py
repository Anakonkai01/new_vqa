import os
import sys

_SRC = os.path.dirname(os.path.dirname(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoders.attention import LSTMDecoderG
from models.base import VQAOutput

class ControlUnit(nn.Module):
    def __init__(self, dim, max_seq_len, num_hops=3):
        super().__init__()
        self.dim = dim
        self.control_question_proj = nn.Linear(dim * 2, dim)
        self.step_emb = nn.Embedding(num_hops, dim)
        nn.init.normal_(self.step_emb.weight, std=0.02)
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, step, context, question, control_state, q_mask=None):
        B = control_state.size(0)
        cq = torch.cat([control_state, question], dim=1) # (B, 2D)
        cq_proj = self.control_question_proj(cq) # (B, D)
        step_t = torch.full((B,), step, dtype=torch.long, device=control_state.device)
        cq_proj = cq_proj + self.step_emb(step_t)
        cq_proj = cq_proj.unsqueeze(1) # (B, 1, D)
        interaction = cq_proj * context # (B, L, D)
        attn_logits = self.attn_proj(interaction).squeeze(-1) # (B, L)
        if q_mask is not None:
            attn_logits = attn_logits.masked_fill(~q_mask, -1e4)
        attn_weights = F.softmax(attn_logits, dim=-1) # (B, L)
        new_control = torch.bmm(attn_weights.unsqueeze(1), context).squeeze(1) # (B, D)
        return new_control

class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.memory_proj = nn.Linear(dim, dim)
        self.kb_proj = nn.Linear(dim, dim)
        self.concat_proj = nn.Linear(dim * 2, dim)
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, memory, control, kb, kb_mask=None):
        mem_proj = self.memory_proj(memory).unsqueeze(1) # (B, 1, D)
        kb_proj = self.kb_proj(kb) # (B, K, D)
        interaction = mem_proj * kb_proj # (B, K, D)
        concat = torch.cat([interaction, kb_proj], dim=-1) # (B, K, 2D)
        concat_proj = self.concat_proj(concat) # (B, K, D)
        control_proj = control.unsqueeze(1) # (B, 1, D)
        attn_interaction = concat_proj * control_proj # (B, K, D)
        attn_logits = self.attn_proj(attn_interaction).squeeze(-1) # (B, K)
        if kb_mask is not None:
            attn_logits = attn_logits.masked_fill(~kb_mask, -1e4)
        attn_weights = F.softmax(attn_logits, dim=-1) # (B, K)
        read_vector = torch.bmm(attn_weights.unsqueeze(1), kb).squeeze(1) # (B, D)
        return read_vector, attn_weights # TRẢ VỀ ATTN ĐỂ PHẠT ENTROPY

class WriteUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, memory, read_vector):
        concat = torch.cat([memory, read_vector], dim=1) # (B, 2D)
        return self.proj(concat)

class MACCell(nn.Module):
    def __init__(self, dim, max_seq_len, num_hops=3):
        super().__init__()
        self.control = ControlUnit(dim, max_seq_len, num_hops=num_hops)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim)

    def forward(self, step, context, question, kb, control, memory, kb_mask=None, q_mask=None):
        new_control = self.control(step, context, question, control, q_mask)
        read_vector, attn_weights = self.read(memory, new_control, kb, kb_mask)
        new_memory = self.write(memory, read_vector)
        return new_control, new_memory, attn_weights

class MACNetwork(nn.Module):
    def __init__(self, dim, num_hops=3, max_seq_len=20):
        super().__init__()
        self.num_hops = num_hops
        self.cell = MACCell(dim, max_seq_len, num_hops=num_hops)
        # Small init avoids saturating LayerNorm/tanh in early hops (randn ~ N(0,1) → ||·|| ~ O(√D))
        self.control_init = nn.Parameter(torch.zeros(1, dim))
        self.memory_init = nn.Parameter(torch.zeros(1, dim))
        self.memory_norm = nn.LayerNorm(dim)
        # Gated hop update (Phase B): blend new vs old memory instead of fixed residual add
        self.hop_gate = nn.Linear(dim * 2, dim)

    def forward(self, context, question, kb, kb_mask=None, q_mask=None):
        B = question.size(0)
        control = self.control_init.expand(B, -1)
        memory = self.memory_init.expand(B, -1)
        all_attn_weights = []
        for step in range(self.num_hops):
            control, new_memory, attn_weights = self.cell(step, context, question, kb, control, memory, kb_mask, q_mask)
            g = torch.sigmoid(self.hop_gate(torch.cat([memory, new_memory], dim=-1)))
            memory = self.memory_norm(g * new_memory + (1.0 - g) * memory)
            all_attn_weights.append(attn_weights)
        return memory, torch.stack(all_attn_weights, dim=1)

class ModelH(nn.Module):
    def __init__(self, vocab_q_size, vocab_a_size, args):
        super().__init__()
        self.args = args
        self.dim = 1024
        
        embed_dim = 300
        self.q_emb = nn.Embedding(vocab_q_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, self.dim // 2, batch_first=True, bidirectional=True)
        
        self.v_dim_input = getattr(args, 'v_dim', 2055)
        self.grid_dim_input = getattr(args, 'grid_dim', 2048)
        
        # Nhánh xử lý đặc trưng Vùng (Region)
        self.v_proj = nn.Sequential(
            nn.Linear(self.v_dim_input, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        # Nhánh xử lý Grid (Toàn cục)
        self.grid_proj = nn.Sequential(
            nn.Linear(self.grid_dim_input, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        # Query-conditioned region gating (Phase B): depends on q_feat + global grid
        self.gate_v = nn.Linear(self.dim, self.dim)
        self.gate_q = nn.Linear(self.dim, self.dim, bias=False)
        self.fusion_norm = nn.LayerNorm(self.dim)

        num_mac_hops = getattr(args, 'num_mac_hops', 3)
        self.mac = MACNetwork(dim=self.dim, num_hops=num_mac_hops)
        
        if args.infonce:
            self.v_head = nn.Linear(self.dim, 128)
            self.q_head = nn.Linear(self.dim, 128)
            
        use_mac_dec = not getattr(args, 'no_mac_decoder', False)
        self.decoder = LSTMDecoderG(
            vocab_size=vocab_a_size, embed_size=embed_dim, hidden_size=self.dim,
            num_layers=getattr(args, 'num_layers', 2), dropout=getattr(args, 'dropout', 0.5),
            use_mac_in_decoder=use_mac_dec,
        )
        
        # KHỞI TẠO ĐỘC LẬP CHO LSTM DECODER (Phá vỡ đối xứng)
        self.init_h1 = nn.Linear(self.dim, self.dim)
        self.init_c1 = nn.Linear(self.dim, self.dim)
        self.init_h2 = nn.Linear(self.dim, self.dim)
        self.init_c2 = nn.Linear(self.dim, self.dim)

    def init_decoder_hidden(self, memory: torch.Tensor):
        """Match teacher-forcing init in forward_with_cov (eval / SCST must use this)."""
        h1 = torch.tanh(self.init_h1(memory))
        c1 = torch.tanh(self.init_c1(memory))
        h2 = torch.tanh(self.init_h2(memory))
        c2 = torch.tanh(self.init_c2(memory))
        h_0 = torch.stack([h1, h2], dim=0)
        c_0 = torch.stack([c1, c2], dim=0)
        return h_0, c_0
        
    def encode(self, q_seq, v_feats, grid_feats=None, img_mask=None):
        B = q_seq.size(0)
        q_embs = self.q_emb(q_seq) 
        lstm_out, (hn, cn) = self.lstm(q_embs)
        context = lstm_out
        q_feat = torch.cat([hn[0], hn[1]], dim=-1)
        
        v_proj = self.v_proj(v_feats)        # (B, 36, 1024)
        g_proj = self.grid_proj(grid_feats) if grid_feats is not None else None  # (B, 1024)

        if g_proj is not None:
            gate = torch.sigmoid(self.gate_v(g_proj) + self.gate_q(q_feat))
            kb = v_proj * gate.unsqueeze(1)
            kb = self.fusion_norm(kb)
        else:
            kb = v_proj

        kb_mask = img_mask
        q_mask = (q_seq != 0)

        memory, mac_attn = self.mac(context, q_feat, kb, kb_mask, q_mask)

        return memory, context, q_feat, kb, v_proj, mac_attn
        
    def forward_with_cov(self, q_seq, v_feats, a_seq, 
                         img_mask=None, length_bin=None, label_tokens=None, grid_feats=None):
        memory, context, q_feat, kb, v_proj_raw, mac_attn = self.encode(q_seq, v_feats, grid_feats, img_mask)
        h_0, c_0 = self.init_decoder_hidden(memory)

        mac_in = memory if not getattr(self.args, 'no_mac_decoder', False) else None
        logits, cov_loss = self.decoder(
            (h_0, c_0), kb, context, a_seq,
            q_token_ids=q_seq, img_mask=img_mask, length_bin=length_bin, label_tokens=label_tokens,
            mac_memory=mac_in,
        )
        output = VQAOutput(logits=logits)
        output.mac_attn = mac_attn
        if getattr(self.args, 'infonce', False) and hasattr(self, 'v_head'):
            output.infonce_loss = self._infonce_loss_symmetric(v_proj_raw, q_feat, img_mask)
        else:
            output.infonce_loss = None
        return output, cov_loss

    def _infonce_loss_symmetric(
        self, v_proj: torch.Tensor, q_feat: torch.Tensor, img_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contrastive align visual regions (masked mean) ↔ question — avoids circular MAC-vs-Q alignment."""
        dev = v_proj.device
        with torch.amp.autocast(device_type=dev.type, enabled=False):
            vp = v_proj.float()
            q = q_feat.float()
            if img_mask is not None:
                m = img_mask.float().unsqueeze(-1)
                cnt = m.sum(dim=1, keepdim=True).clamp(min=1.0)
                v_bar = (vp * m).sum(dim=1) / cnt.squeeze(-1)
            else:
                v_bar = vp.mean(dim=1)
            z_v = F.normalize(self.v_head(v_bar), dim=1)
            z_q = F.normalize(self.q_head(q), dim=1)
            B = z_q.size(0)
            if B < 2:
                return v_proj.new_zeros(())
            logits_qv = (torch.mm(z_q, z_v.t()) / 0.07).clamp(-100, 100)
            logits_vq = (torch.mm(z_v, z_q.t()) / 0.07).clamp(-100, 100)
            labels = torch.arange(B, device=dev)
            loss = (
                F.cross_entropy(logits_qv, labels) + F.cross_entropy(logits_vq, labels)
            ) * 0.5
        return loss

    def decode_step(self, token, h, c, V, Q_H=None, coverage=None, img_mask=None, q_token_ids=None, length_bin=None, label_tokens=None, mac_memory=None):
        if getattr(self.args, 'no_mac_decoder', False):
            mm = None
        else:
            if mac_memory is None:
                raise ValueError("mac_memory is required when MAC-in-decoder is enabled (Model H)")
            mm = mac_memory
        logit, (h_new, c_new), img_alpha, coverage_new = self.decoder.decode_step(
            token, (h, c), V, Q_H, coverage=coverage, q_token_ids=q_token_ids,
            img_mask=img_mask, length_bin=length_bin, label_tokens=label_tokens,
            mac_memory=mm,
        )
        return logit, h_new, c_new, img_alpha, coverage_new

    def get_infonce_loss(self, q_seq, v_feats, grid_feats=None, img_mask=None):
        """Legacy path (extra encode). Prefer infonce_loss from forward_with_cov in training."""
        if not hasattr(self, 'v_head'):
            return torch.zeros((), device=q_seq.device, dtype=torch.float32)
        _, _, q_feat, _kb, v_proj_raw, _ = self.encode(q_seq, v_feats, grid_feats, img_mask)
        return self._infonce_loss_symmetric(v_proj_raw, q_feat, img_mask)