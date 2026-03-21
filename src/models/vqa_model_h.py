"""
src/models/vqa_model_h.py — Project Model H
===========================================
The ultimate LSTM-based Generative VQA Model.
Upgrades over Model G:
  1. No MUTAN -> Replaced by MAC Network (Multi-hop Attention).
  2. Integrated handling for Region (BUTD) AND Grid (CNN) features.
  3. Prepares the architecture to ingest FastText / ConceptNet external embeddings.
"""

import os
import sys

_SRC = os.path.dirname(os.path.dirname(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoders.attention import LSTMDecoderG
from models.base import VQAOutput

class ControlUnit(nn.Module):
    """
    Control Unit: Updates the control state based on the question summary 
    and the current context words.
    """
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.position_aware_proj = nn.Linear(dim, dim)
        self.control_question_proj = nn.Linear(dim * 2, dim)
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, step, context, question, control_state, q_mask=None):
        # context: (B, L, D) - sequence of question word embeddings
        # question: (B, D) - global question embedding
        # control_state: (B, D) - previous control state
        
        # Combine previous state and global question
        cq = torch.cat([control_state, question], dim=1) # (B, 2D)
        cq_proj = self.control_question_proj(cq).unsqueeze(1) # (B, 1, D)
        
        # Interact with context words
        interaction = cq_proj * context # (B, L, D)
        attn_logits = self.attn_proj(interaction).squeeze(-1) # (B, L)
        
        # Apply padding mask before softmax!
        if q_mask is not None:
            # Use -1e4 instead of -1e9 to avoid PyTorch Float16 (AMP) overflow crash
            attn_logits = attn_logits.masked_fill(~q_mask, -1e4)
            
        attn_weights = F.softmax(attn_logits, dim=-1) # (B, L)
        
        # New control state is a weighted sum of context words
        new_control = torch.bmm(attn_weights.unsqueeze(1), context).squeeze(1) # (B, D)
        return new_control

class ReadUnit(nn.Module):
    """
    Read Unit: Extracts information from the knowledge base (image features)
    guided by the new control state and the previous memory.
    """
    def __init__(self, dim):
        super().__init__()
        self.memory_proj = nn.Linear(dim, dim)
        self.kb_proj = nn.Linear(dim, dim)
        self.concat_proj = nn.Linear(dim * 2, dim)
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, memory, control, kb, kb_mask=None):
        # memory: (B, D)
        # control: (B, D)
        # kb: (B, K, D) - knowledge base (e.g., image regions)
        
        # Combine memory and kb
        mem_proj = self.memory_proj(memory).unsqueeze(1) # (B, 1, D)
        kb_proj = self.kb_proj(kb) # (B, K, D)
        
        interaction = mem_proj * kb_proj # (B, K, D)
        
        # Concatenate interaction with kb and project
        concat = torch.cat([interaction, kb_proj], dim=-1) # (B, K, 2D)
        concat_proj = self.concat_proj(concat) # (B, K, D)
        
        # Guide attention using the control state
        control_proj = control.unsqueeze(1) # (B, 1, D)
        attn_interaction = concat_proj * control_proj # (B, K, D)
        
        attn_logits = self.attn_proj(attn_interaction).squeeze(-1) # (B, K)
        
        # Apply padding mask before softmax!
        if kb_mask is not None:
            # False values are padding, pull their logits to negative infinity
            # Use -1e4 instead of -1e9 to avoid PyTorch Float16 (AMP) overflow crash
            attn_logits = attn_logits.masked_fill(~kb_mask, -1e4)
            
        attn_weights = F.softmax(attn_logits, dim=-1) # (B, K)
        
        # Read vector is the attended knowledge base
        read_vector = torch.bmm(attn_weights.unsqueeze(1), kb).squeeze(1) # (B, D)
        return read_vector

class WriteUnit(nn.Module):
    """
    Write Unit: Integrates the read vector into the memory state.
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, memory, read_vector):
        concat = torch.cat([memory, read_vector], dim=1) # (B, 2D)
        new_memory = self.proj(concat)
        return new_memory

class MACCell(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.control = ControlUnit(dim, max_seq_len)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim)

    def forward(self, step, context, question, kb, control, memory, kb_mask=None, q_mask=None):
        new_control = self.control(step, context, question, control, q_mask)
        read_vector = self.read(memory, new_control, kb, kb_mask)
        new_memory = self.write(memory, read_vector)
        return new_control, new_memory

class MACNetwork(nn.Module):
    def __init__(self, dim, num_hops=3, max_seq_len=20):
        super().__init__()
        self.num_hops = num_hops
        self.cell = MACCell(dim, max_seq_len)
        
        # Init control and memory vectors
        self.control_init = nn.Parameter(torch.randn(1, dim))
        self.memory_init = nn.Parameter(torch.randn(1, dim))

    def forward(self, context, question, kb, kb_mask=None, q_mask=None):
        # context: (B, L, D) - words
        # question: (B, D) - global query
        # kb: (B, K, D) - image features
        B = question.size(0)
        
        control = self.control_init.expand(B, -1)
        memory = self.memory_init.expand(B, -1)
        
        for step in range(self.num_hops):
            control, memory = self.cell(step, context, question, kb, control, memory, kb_mask, q_mask)
            
        return memory

class ModelH(nn.Module):
    """
    The Ultimate LSTM VQA Architecture: Model H.
    """
    def __init__(self, vocab_q_size, vocab_a_size, args):
        super().__init__()
        self.args = args
        self.dim = 1024
        
        # Note: In Phase H2, these embeddings will be initialized with FastText matrix
        if hasattr(args, 'use_fasttext') and args.use_fasttext:
            embed_dim = 300
        else:
            embed_dim = 300
            
        self.q_emb = nn.Embedding(vocab_q_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, self.dim // 2, batch_first=True, bidirectional=True)
        
        self.v_dim_input = getattr(args, 'v_dim', 2055)
        self.grid_dim_input = getattr(args, 'grid_dim', 2048)
        
        # Feature Projectors
        self.v_proj = nn.Sequential(
            nn.Linear(self.v_dim_input, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        # Optional Grid Features
        self.grid_proj = nn.Sequential(
            nn.Linear(self.grid_dim_input, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        # MAC Network replaces MUTAN
        self.mac = MACNetwork(dim=self.dim, num_hops=3)
        
        # InfoNCE projection heads (G3)
        if args.infonce:
            self.v_head = nn.Linear(self.dim, 128)
            self.q_head = nn.Linear(self.dim, 128)
            
        # Decoder (from Model G, reusing G2 pgn3 and G5 len_cond)
        self.decoder = LSTMDecoderG(
            vocab_size=vocab_a_size,
            embed_size=embed_dim,
            hidden_size=self.dim,
            num_layers=getattr(args, 'num_layers', 2),
            dropout=getattr(args, 'dropout', 0.5)
        )
        
    def encode(self, q_seq, v_feats, grid_feats=None, img_mask=None):
        """
        Extract features using MAC multi-hop reasoning.
        """
        B = q_seq.size(0)
        
        # 1. Question Encoding
        q_embs = self.q_emb(q_seq) # (B, L, 300)
        lstm_out, (hn, cn) = self.lstm(q_embs)
        
        # Context is the sequence of hidden states: (B, L, 1024)
        context = lstm_out
        
        # Global question is the concatenated final hidden states: (B, 1024)
        q_feat = torch.cat([hn[0], hn[1]], dim=-1)
        
        # 2. Image Encoding
        v_proj = self.v_proj(v_feats) # (B, 36, 1024)
        kb = v_proj
        kb_mask = img_mask
        
        if grid_feats is not None: # (B, 1024)
            g_proj = self.grid_proj(grid_feats).unsqueeze(1) # (B, 1, 1024)
            # Concatenate region concepts and global grid context into Knowledge Base
            kb = torch.cat([v_proj, g_proj], dim=1) # (B, 37, 1024)
            if img_mask is not None:
                # Add a 'True' column for the global grid feature since it's always valid
                grid_mask = torch.ones(B, 1, dtype=torch.bool, device=img_mask.device)
                kb_mask = torch.cat([img_mask, grid_mask], dim=1)
            
        # Question padding mask for Control Unit Focus
        q_mask = (q_seq != 0)
            
        # 3. MAC Multi-Hop Reasoning
        memory = self.mac(context, q_feat, kb, kb_mask, q_mask) # (B, 1024)
        
        # The ultimate fused representation is the Memory state.
        # Returning context as well so the Pointer-Generator has words to point to!
        return memory, context, q_feat, v_proj
        
    def forward_with_cov(self, q_seq, v_feats, a_seq, 
                         img_mask=None, length_bin=None, label_tokens=None, grid_feats=None):
        memory, context, q_feat, v_proj = self.encode(q_seq, v_feats, grid_feats, img_mask)
        
        # MAC outputs a fused memory vector (B, H). We use this to initialize the decoder LSTM.
        h_0 = memory.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1) # (num_layers, B, H)
        c_0 = torch.zeros_like(h_0)
        
        # Fix the Q_H Mock: The Sequence of Question Hidden states (Context) for Pointer-Generator
        Q_H = context # (B, L, H) - Real question words
        
        logits, cov_loss = self.decoder(
            (h_0, c_0), v_proj, Q_H, a_seq,
            q_token_ids=q_seq,
            img_mask=img_mask,
            length_bin=length_bin,
            label_tokens=label_tokens
        )
        # Using a mock VQAOutput object
        return VQAOutput(logits=logits), cov_loss

    def decode_step(self, token, h, c, V, Q_H=None, coverage=None, img_mask=None, q_token_ids=None, length_bin=None, label_tokens=None):
        logit, (h_new, c_new), img_alpha, coverage_new = self.decoder.decode_step(
            token, (h, c), V, Q_H,
            coverage=coverage,
            q_token_ids=q_token_ids,
            img_mask=img_mask,
            length_bin=length_bin,
            label_tokens=label_tokens
        )
        return logit, h_new, c_new, img_alpha, coverage_new

    def get_infonce_loss(self, q_seq, v_feats, grid_feats=None, img_mask=None):
        if not hasattr(self, 'v_head'):
            return torch.tensor(0.0)
            
        _, _, q_feat, v_proj = self.encode(q_seq, v_feats, grid_feats, img_mask)
        
        # Aggregate v_proj (mean pooling) for alignment
        v_agg = v_proj.mean(dim=1)
        
        z_v = F.normalize(self.v_head(v_agg), dim=1)
        z_q = F.normalize(self.q_head(q_feat), dim=1)
        
        tau = 0.07
        sim = torch.mm(z_q, z_v.t()) / tau
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)
