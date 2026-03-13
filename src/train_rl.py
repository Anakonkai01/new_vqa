import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Append src to path
import sys
sys.path.append(os.path.dirname(__file__))

from dataset import VQAEDataset, vqa_collate_fn
from vocab import Vocabulary
from models.vqa_models import VQAModelA, VQAModelB, VQAModelC, VQAModelD, VQAModelE

# Setup NLTK
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------------------------------------------------------
# CONSTANTS & CONFIG
# ---------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_IMAGE_DIR = "data/raw/images/train2014"
TRAIN_VQA_E_JSON = "data/raw/vqa_e_json/VQA-E_train_set.json"
VOCAB_Q_PATH = "data/processed/vocab_questions.json"
VOCAB_A_PATH = "data/processed/vocab_answers.json"

smoothie = SmoothingFunction().method1

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def decode_tensor(a_tensor, vocab_a):
    special = {
        vocab_a.word2idx['<pad>'],
        vocab_a.word2idx['<start>'],
        vocab_a.word2idx['<end>']
    }
    words = [vocab_a.idx2word[int(i)] for i in a_tensor if int(i) not in special]
    return words

def compute_rewards(sampled_seqs, greedy_seqs, target_seqs, vocab_a):
    """
    Compute BLEU-4 reward for sampled vs greedy.
    Returns tensors of shape (batch_size,)
    """
    batch_size = target_seqs.size(0)
    sampled_rewards = torch.zeros(batch_size, device=DEVICE)
    greedy_rewards = torch.zeros(batch_size, device=DEVICE)
    
    for i in range(batch_size):
        gt_words = decode_tensor(target_seqs[i], vocab_a)
        if not gt_words: gt_words = ['<unk>']
        
        samp_words = decode_tensor(sampled_seqs[i], vocab_a)
        greed_words = decode_tensor(greedy_seqs[i], vocab_a)
        
        # BLEU-4 formulation
        sr = sentence_bleu([gt_words], samp_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        gr = sentence_bleu([gt_words], greed_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        # small penalization for extremely short generations
        if len(samp_words) < 3: sr *= 0.5
        if len(greed_words) < 3: gr *= 0.5
            
        sampled_rewards[i] = sr
        greedy_rewards[i] = gr
        
    return sampled_rewards, greedy_rewards

def train_rl_epoch(model, loader, optimizer, vocab_a, start_idx, end_idx, max_len=60):
    """
    Self-Critical Sequence Training (SCST) loop for one epoch
    """
    model.train()
    
    # In RL, we typically freeze the feature extractors to stabilize variance
    # We'll assume the encoder is largely frozen and we only train the decoder
    
    total_loss = 0.0
    total_reward = 0.0
    total_greedy_reward = 0.0
    
    pbar = tqdm(loader, desc="RL SCST Training")
    for batch_idx, (imgs, questions, answers) in enumerate(pbar):
        imgs = imgs.to(DEVICE)
        questions = questions.to(DEVICE)
        answers = answers.to(DEVICE)
        
        batch_size = imgs.size(0)
        
        # 1. Forward Encoder (Requires custom handling per model type, assuming Model E API here)
        # To maintain uniformity, we interact directly with the model's encoders
        model.eval() # Freeze BN/Dropout during encoding and greedy decoding for stability
        with torch.no_grad():
            if isinstance(model, VQAModelE) or isinstance(model, VQAModelC) or isinstance(model, VQAModelD):
                # Spatial Models
                img_features = model.i_encoder(imgs)
                img_features = torch.nn.functional.normalize(img_features, p=2, dim=-1)
                
                # Question
                q_feature, q_hidden_states = model.q_encoder(questions)
                if hasattr(model, 'q_norm'): q_feature = model.q_norm(q_feature)
                
                # Fusion
                if isinstance(model, VQAModelE):
                    img_mean = img_features.mean(dim=1)
                    fusion_global = model.fusion(img_mean, q_feature)
                    h_0_base = model.init_h_proj(fusion_global)
                    c_0_base = model.init_c_proj(fusion_global)
                    h_0 = h_0_base.unsqueeze(0).repeat(model.num_layers, 1, 1)
                    c_0 = c_0_base.unsqueeze(0).repeat(model.num_layers, 1, 1)
                    encoder_hidden = (h_0, c_0)
                    modulated_img_features = model.fusion(img_features, q_feature)
                    
                    # Greedy Decoding (Baseline)
                    greedy_seqs, _ = model.decoder.sample(
                        encoder_hidden, modulated_img_features, q_hidden_states, 
                        max_len, start_idx, end_idx, method='greedy'
                    )
                else:
                    img_mean = img_features.mean(dim=1)
                    fusion_global = model.fusion(img_mean, q_feature)
                    h_0 = fusion_global.unsqueeze(0).repeat(model.num_layers, 1, 1)
                    c_0 = torch.zeros_like(h_0)
                    encoder_hidden = (h_0, c_0)
                    modulated_img_features = img_features
                    
                    # Greedy Decoding
                    greedy_seqs, _ = model.decoder.sample(
                        encoder_hidden, modulated_img_features, q_hidden_states, 
                        max_len, start_idx, end_idx, method='greedy'
                    )
            else:
                raise NotImplementedError("RL training script only fully implemented for Spatial Attention Models (C, D, E)")
                
        # 2. Sampled Decoding (Exploration) - ENABLE GRADIENTS
        model.train()
        optimizer.zero_grad()
        
        # Re-run encoder with grads (or freeze and just run decoder)
        # We'll just run decoder with grads on to save memory, utilizing the detached encoder features
        samp_seqs, samp_log_probs = model.decoder.sample(
            encoder_hidden, modulated_img_features, q_hidden_states, 
            max_len, start_idx, end_idx, method='sample'
        )
        
        # 3. Compute Rewards (Non-differentiable)
        with torch.no_grad():
            samp_rewards, greed_rewards = compute_rewards(samp_seqs, greedy_seqs, answers, vocab_a)
            
            # SCST REWARD ADVANTAGE
            reward_diff = samp_rewards - greed_rewards # (batch_size,)
            
        # 4. Compute Loss: L = - (r_sample - r_greedy) * sum(log_prob)
        # Mask out padding in log probs using seq lengths
        mask = (samp_seqs != vocab_a.word2idx['<pad>']).float() # (batch_size, max_len)
        masked_log_probs = samp_log_probs * mask # (batch_size, max_len)
        seq_log_probs = masked_log_probs.sum(dim=1) # (batch_size,)
        
        loss = - (reward_diff * seq_log_probs).mean()
        
        # 5. Backprop
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=5.0)
            optimizer.step()
            
        total_loss += loss.item()
        total_reward += samp_rewards.mean().item()
        total_greedy_reward += greed_rewards.mean().item()
        
        pbar.set_postfix({
            "RL_Loss": f"{loss.item():.4f}", 
            "R_Samp": f"{samp_rewards.mean().item():.3f}",
            "R_Greed": f"{greed_rewards.mean().item():.3f}",
            "Advantage": f"{reward_diff.mean().item():.3f}"
        })
        
    n = len(loader)
    return total_loss / n, total_reward / n, total_greedy_reward / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='E')
    parser.add_argument('--base_checkpoint', type=str, required=True, help="Path to fully CE-trained checkpoint")
    parser.add_argument('--lr', type=float, default=1e-5, help="Extremely small learning rate for RL fine-tuning")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=60)
    args = parser.parse_args()
    
    print(f"Loading vocabulary...")
    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)
    
    start_idx = vocab_a.word2idx['<start>']
    end_idx = vocab_a.word2idx['<end>']
    
    print(f"Loading Train Dataset (RL requires training data)...")
    train_dataset = VQAEDataset(
        image_dir=TRAIN_IMAGE_DIR, vqa_e_json_path=TRAIN_VQA_E_JSON,
        vocab_q=vocab_q, vocab_a=vocab_a, split='train2014', augment=False
    )
    # We use a relatively small dataset for RL because it's slow
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=vqa_collate_fn, num_workers=2)
    
    print(f"Loading Model {args.model_type} from {args.base_checkpoint}")
    # Initialize based on type
    if args.model_type == 'E':
        model = VQAModelE(len(vocab_q), len(vocab_a)).to(DEVICE)
    elif args.model_type == 'C':
        model = VQAModelC(len(vocab_q), len(vocab_a)).to(DEVICE)
    else:
        raise ValueError("Unsupported model for SCST in this script.")
        
    model.load_state_dict(torch.load(args.base_checkpoint, map_location=DEVICE)['model_state_dict'], strict=False)
    
    # Optimizer - ONLY fine-tune the decoder
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.lr)
    
    print("\nStarting Self-Critical Sequence Training (SCST)...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- RL Epoch {epoch}/{args.epochs} ---")
        avg_loss, r_samp, r_greed = train_rl_epoch(model, train_loader, optimizer, vocab_a, start_idx, end_idx, args.max_len)
        
        print(f"Epoch {epoch} Summary: Loss={avg_loss:.4f} | R_samp(BLEU-4)={r_samp:.4f} | R_greed(BLEU-4)={r_greed:.4f}")
        
        # Save SCST Checkpoint
        save_path = f"checkpoints/model_{args.model_type.lower()}_scst_epoch{epoch}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward_samp': r_samp,
            'reward_greed': r_greed
        }, save_path)
        print(f"Saved SCST checkpoint to {save_path}")

if __name__ == "__main__":
    main()
