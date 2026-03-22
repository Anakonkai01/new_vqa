#!/usr/bin/env python3
import argparse
import os
import random
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, ConcatDataset

from fasttext_utils import build_fasttext_matrix
from models.vqa_model_h import ModelH
from data.dataset import VQAGenerativeDataset, make_butd_loader
from data.collate import make_collate_fn
from data.samplers import build_mixed_sampler, build_replay_sampler
from training.losses import build_criterion, attention_entropy_penalty
from vocab import Vocabulary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--vg_feat_dir', type=str, required=True)
    parser.add_argument('--merged_json', type=str, default='data/processed/merged_train_filtered.json')
    parser.add_argument('--vocab_q_path', type=str, default='data/processed/vocab_questions.json')
    parser.add_argument('--vocab_a_path', type=str, default='data/processed/vocab_answers.json')
    parser.add_argument('--use_fasttext', action='store_true')
    parser.add_argument('--infonce', action='store_true')
    parser.add_argument('--scst', action='store_true')
    parser.add_argument('--ohp_lambda', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='vqa-model-h')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--patience', type=int, default=2, help="Early stopping patience")
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--save_legacy_alias', action='store_true')
    return parser.parse_args()

def train_h(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    os.makedirs("checkpoints/h", exist_ok=True)
    phase_tag = f"phase{args.phase}"
    resume_ckpt_path = f"checkpoints/h/model_h_{phase_tag}_resume.pth"
    best_ckpt_path = f"checkpoints/h/model_h_{phase_tag}_best.pth"
    
    _wb = None
    if args.wandb:
        try:
            import wandb
            _wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), resume='allow')
        except ImportError: pass
    
    q_vocab = Vocabulary(); q_vocab.load(args.vocab_q_path)
    a_vocab = Vocabulary(); a_vocab.load(args.vocab_a_path)
    
    h1_loader = make_butd_loader(args.vg_feat_dir)
    ds_expl = VQAGenerativeDataset.from_merged_json(args.merged_json, q_vocab, a_vocab, feature_loader=h1_loader, use_butd=True)
    
    n_val = max(int(0.05 * len(ds_expl)), 1000)
    indices = list(range(len(ds_expl)))
    random.Random(42).shuffle(indices)
    
    ds_val = Subset(ds_expl, indices[:n_val])
    ds_expl_train = Subset(ds_expl, indices[n_val:])
    
    ds_vqa_v2_mock = Subset(ds_expl_train, [i for i, idx in enumerate(indices[n_val:]) if ds_expl.annotations[idx].get("source") == 'vqa_v2'])
    if len(ds_vqa_v2_mock) == 0: ds_vqa_v2_mock = None

    if args.phase == 1:
        train_dataset, train_sampler = build_mixed_sampler([ds_vqa_v2_mock, ds_expl_train], [0.4, 0.6]) if ds_vqa_v2_mock else (ds_expl_train, None)
    elif args.phase in (2, 3):
        train_dataset, train_sampler = build_replay_sampler(ds_expl_train, ds_vqa_v2_mock, replay_fraction=0.2) if ds_vqa_v2_mock else (ds_expl_train, None)
    else:
        train_dataset, train_sampler = ds_expl_train, None
        args.batch_size = min(args.batch_size, 64)

    _collate = make_collate_fn(use_butd=True, a_vocab=a_vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, collate_fn=_collate, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=_collate, num_workers=args.num_workers)

    import glob
    feat_files = glob.glob(os.path.join(args.vg_feat_dir, '*.pt'))
    if feat_files:
        sample = torch.load(feat_files[0], map_location='cpu', weights_only=False)
        args.v_dim = sample['region_feat'].shape[1]
        args.grid_dim = sample['grid_feat'].shape[0] if 'grid_feat' in sample else 2048
    else: args.v_dim, args.grid_dim = 2055, 2048
        
    model = ModelH(len(q_vocab), len(a_vocab), args).to(DEVICE)
    
    if args.use_fasttext:
        q_mat, _ = build_fasttext_matrix(q_vocab.word2idx)
        model.q_emb.weight.data.copy_(q_mat)
        a_mat, _ = build_fasttext_matrix(a_vocab.word2idx)
        model.decoder.embedding.weight.data.copy_(a_mat)
        model.decoder.fc.weight = model.decoder.embedding.weight
        
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try: model = torch.compile(model, mode='default', dynamic=True)
        except Exception: pass

    # Phân tầng Learning Rate
    embed_params = list(model.q_emb.parameters())
    base_params = [p for n, p in model.named_parameters() if 'q_emb' not in n]
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': args.lr, 'weight_decay': 1e-4},
        {'params': embed_params, 'lr': args.lr * 0.1, 'weight_decay': 0.0}
    ])

    if args.warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.lr * 0.01)
    else:
        warmup_sched = None
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)

    criterion = build_criterion(gamma=2.0, label_smoothing=0.1 if args.phase != 4 else 0.0, use_focal=(args.phase != 4), ignore_index=0)
    scaler = GradScaler('cuda')

    start_epoch, best_val_loss, epochs_no_improve = 0, float('inf'), 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if ckpt.get('args', {}).get('phase', args.phase) == args.phase:
            start_epoch = ckpt.get('epoch', 0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            history = ckpt.get('history', history)
            try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception: pass

    glove_embed = None
    if args.scst and args.phase == 4 and args.ohp_lambda > 0:
        glove_embed = {w: model.decoder.embedding.weight[i].detach().cpu().numpy() for w, i in a_vocab.word2idx.items()}

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        ep_loss, ep_acc, ep_gnorm = 0.0, 0.0, 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for step_idx, batch in enumerate(train_pbar):
            feats, questions, targets = batch.feats.to(DEVICE), batch.questions.to(DEVICE), batch.targets.to(DEVICE)
            img_mask = batch.img_mask.to(DEVICE) if batch.img_mask is not None else None
            length_bin = batch.length_bins.to(DEVICE) if getattr(batch, 'length_bins', None) is not None else None
            label_tokens = batch.label_tokens.to(DEVICE) if getattr(batch, 'label_tokens', None) is not None else None
            grid_feats = batch.grid_feats.to(DEVICE) if getattr(batch, 'grid_feats', None) is not None else None
            
            dec_in, dec_tgt = targets[:, :-1], targets[:, 1:]
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=amp_dtype): # FIX LỖI NAN BẰNG BFLOAT16
                out, cov_loss_scalar = model.forward_with_cov(
                    questions, feats, dec_in, img_mask=img_mask, length_bin=length_bin, label_tokens=label_tokens, grid_feats=grid_feats
                )
                ce_loss = criterion(out.logits, dec_tgt)
                mac_attn = getattr(out, 'mac_attn', None)
                entropy_loss = attention_entropy_penalty(mac_attn, eps=1e-7) # CHÈN HÀM PHẠT ENTROPY
                loss = ce_loss + 0.5 * cov_loss_scalar + 0.01 * entropy_loss
                
                with torch.no_grad():
                    preds = out.logits.argmax(dim=-1)
                    mask = dec_tgt != 0
                    acc = (preds[mask] == dec_tgt[mask]).float().mean().item()
                    ep_acc += acc

                if args.scst and args.phase == 4:
                    from training.scst import scst_step
                    target_texts = []
                    for row in dec_tgt:
                        words = []
                        for tid in row.tolist():
                            if tid in (0, 2): break
                            w = a_vocab.idx2word.get(tid, '<unk>') if hasattr(a_vocab, 'idx2word') else a_vocab.idx_to_word.get(tid, '<unk>')
                            words.append(w)
                        target_texts.append(' '.join(words))

                    # SCST ĐÃ ĐƯỢC CHUẨN HÓA VỚI SAMPLE=TRUE, CIDER=1.0
                    rl_loss, _ = scst_step(
                        model, 'H', feats, questions, target_texts, a_vocab, 
                        device=DEVICE, max_len=dec_in.size(1),
                        cider_weight=1.0, exact_match_weight=0.0, bleu_weight=0.0, meteor_weight=0.0, 
                        ohp_weight=args.ohp_lambda, glove_embed=glove_embed, return_stats=True, sample=True
                    )
                    if rl_loss is not None:
                        loss = loss * 0.5 + rl_loss * 0.5

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            g_norm = 0.0
            for p in model.parameters():
                if p.grad is not None: g_norm += p.grad.detach().data.norm(2).item() ** 2
            ep_gnorm += g_norm ** 0.5

            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.3f}", 'acc': f"{acc*100:.1f}%"})
            if args.dry_run and step_idx >= 5: return

        avg_loss, avg_acc = ep_loss / len(train_loader), ep_acc / len(train_loader)
        
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
            for v_batch in val_pbar:
                v_feats, v_q, v_tgt = v_batch.feats.to(DEVICE), v_batch.questions.to(DEVICE), v_batch.targets.to(DEVICE)
                v_mask = v_batch.img_mask.to(DEVICE) if v_batch.img_mask is not None else None
                v_grid = v_batch.grid_feats.to(DEVICE) if getattr(v_batch, 'grid_feats', None) is not None else None
                
                with autocast('cuda', dtype=amp_dtype):
                    v_out, _ = model.forward_with_cov(v_q, v_feats, v_tgt[:, :-1], img_mask=v_mask, grid_feats=v_grid)
                    v_loss = criterion(v_out.logits, v_tgt[:, 1:]).item()
                    val_loss += v_loss
                    v_preds = v_out.logits.argmax(dim=-1)
                    v_m = v_tgt[:, 1:] != 0
                    val_acc += (v_preds[v_m] == v_tgt[:, 1:][v_m]).float().mean().item()
                    
        avg_val, avg_vacc = val_loss / max(1, len(val_loader)), val_acc / max(1, len(val_loader))
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Trn_Loss: {avg_loss:.4f} | Val_Loss: {avg_val:.4f} | Trn_Acc: {avg_acc*100:.1f}% | Val_Acc: {avg_vacc*100:.1f}% | LR: {cur_lr:.2e}")
        
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_val)
        history['lr'].append(cur_lr)
        with open(history_path, 'w') as f: json.dump(history, f, indent=2)
            
        if _wb: _wb.log({'train/loss': avg_loss, 'val/loss': avg_val, 'lr': cur_lr, 'epoch': epoch + 1})
        
        if args.warmup_epochs > 0 and epoch < start_epoch + args.warmup_epochs: warmup_sched.step()
        else: cosine_sched.step()
            
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        ckpt_meta = {'epoch': epoch + 1, 'model_state_dict': sd, 'optimizer_state_dict': optimizer.state_dict(),
                     'scaler_state_dict': scaler.state_dict(), 'best_val_loss': best_val_loss, 'history': history, 'args': vars(args)}
        torch.save(ckpt_meta, resume_ckpt_path)

        if avg_val < best_val_loss:
            best_val_loss, epochs_no_improve = avg_val, 0
            ckpt_meta['best_val_loss'] = best_val_loss
            torch.save(ckpt_meta, best_ckpt_path)
            print("[*] New Best Validation Loss - Checkpoint Saved!")
        else:
            epochs_no_improve += 1
            print(f"[!] Validation Loss did not improve. Patience: {epochs_no_improve}/{args.patience}")
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"\n[CRITICAL] EARLY STOPPING ACTIVATED! Model H đã hội tụ.")
                break

    if _wb: _wb.finish()

if __name__ == '__main__':
    train_h(parse_args())