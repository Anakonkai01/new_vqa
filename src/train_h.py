#!/usr/bin/env python3
import argparse
import math
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
from data.samplers import (
    build_mixed_sampler,
    build_replay_sampler,
    build_subset_weighted_sampler,
    build_phase4_sources,
)
from training.losses import build_criterion, attention_entropy_penalty
from vocab import Vocabulary


def _normalize_ckpt_state_dict(ckpt_sd, model):
    """
    torch.compile wraps the module as _orig_mod; checkpoints then have keys like
    '_orig_mod.q_emb.weight'. Loading into an eager ModelH (no compile) with strict=False
    silently skips them — weights stay random and loss explodes. Map prefixes both ways.
    """
    model_sd = model.state_dict()
    ck_has = any(k.startswith("_orig_mod.") for k in ckpt_sd)
    m_has = any(k.startswith("_orig_mod.") for k in model_sd)
    if ck_has and not m_has:
        return {k[len("_orig_mod.") :]: v for k, v in ckpt_sd.items() if k.startswith("_orig_mod.")}
    if not ck_has and m_has:
        return {"_orig_mod." + k: v for k, v in ckpt_sd.items()}
    return ckpt_sd


def _model_state_dict_for_save(model):
    """Save without _orig_mod. so checkpoints are portable across compile on/off."""
    sd = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items() if k.startswith("_orig_mod.")}
    return sd


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
    parser.add_argument('--infonce_beta', type=float, default=0.1,
                        help='Weight for InfoNCE (G3-style) when --infonce')
    parser.add_argument('--scheduled_sampling', action='store_true',
                        help='Phase 3: token-level scheduled sampling (inverse-sigmoid epsilon)')
    parser.add_argument('--ss_k', type=float, default=5.0, help='SS schedule constant k')
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
    parser.add_argument('--no_mac_decoder', action='store_true',
                        help='Ablation: LSTM without MAC context at each step (smaller decoder)')
    parser.add_argument('--num_mac_hops', type=int, default=3,
                        help='Number of MAC reasoning hops')
    # C: VQA v2.0 real + weighted sampling
    parser.add_argument('--vqa_v2_questions', type=str, default=None,
                        help='e.g. data/annotations/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json')
    parser.add_argument('--vqa_v2_annotations', type=str, default=None,
                        help='e.g. data/annotations/vqa_v2/v2_mscoco_train2014_annotations.json')
    parser.add_argument('--weighted_sampling', action='store_true',
                        help='Upweight vqa_x / a-ok via quality (only when not using VQA2 mix sampler)')
    parser.add_argument('--w_vqa_e', type=float, default=1.0)
    parser.add_argument('--w_vqa_x', type=float, default=1.15)
    parser.add_argument('--w_aokvqa', type=float, default=1.1)
    parser.add_argument('--quality_gamma', type=float, default=0.5)
    # D: entropy + SCST composite
    parser.add_argument('--entropy_mac_coef', type=float, default=0.05,
                        help='Weight for MAC attention entropy penalty')
    parser.add_argument('--scst_bleu', type=float, default=0.3)
    parser.add_argument('--scst_meteor', type=float, default=0.3)
    parser.add_argument('--scst_exact_match', type=float, default=0.0)
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile (debug RL/SCST)')
    return parser.parse_args()


def _ss_forward_h(model, questions, feats, dec_input, epsilon,
                  grid_feats=None, img_mask=None, length_bin=None, label_tokens=None):
    """Scheduled sampling for Model H (token-level). Matches train.py / train_g pattern."""
    B = feats.size(0)
    max_len = dec_input.size(1)
    memory, context, _qfeat, v_proj, _ = model.encode(
        questions, feats, grid_feats=grid_feats, img_mask=img_mask)
    h, c = model.init_decoder_hidden(memory)
    if length_bin is None:
        length_bin = feats.new_full((B,), 2, dtype=torch.long, device=feats.device)
    coverage = None
    current_tok = dec_input[:, 0]
    logits_list = []
    mm = None if getattr(model.args, 'no_mac_decoder', False) else memory
    for t in range(max_len):
        tok = current_tok.unsqueeze(1)
        logit, h, c, _, coverage = model.decode_step(
            tok, h, c, v_proj, context,
            coverage=coverage,
            img_mask=img_mask,
            q_token_ids=questions,
            length_bin=length_bin,
            label_tokens=label_tokens,
            mac_memory=mm,
        )
        logits_list.append(logit)
        if t < max_len - 1:
            current_tok = dec_input[:, t + 1] if random.random() < epsilon else logit.detach().argmax(dim=-1)
    return torch.stack(logits_list, dim=1)


def train_h(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    os.makedirs("checkpoints/h", exist_ok=True)
    phase_tag = f"phase{args.phase}"
    resume_ckpt_path = f"checkpoints/h/model_h_{phase_tag}_resume.pth"
    best_ckpt_path = f"checkpoints/h/model_h_{phase_tag}_best.pth"
    history_path = f"checkpoints/h/history_model_h_{phase_tag}.json"

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
    if args.phase == 4:
        ds_expl = build_phase4_sources(ds_expl)

    n_val = max(int(0.05 * len(ds_expl)), 1000)
    indices = list(range(len(ds_expl)))
    random.Random(42).shuffle(indices)

    ds_val = Subset(ds_expl, indices[:n_val])
    ds_expl_train = Subset(ds_expl, indices[n_val:])

    ds_vqa_v2_real = None
    if args.vqa_v2_questions and args.vqa_v2_annotations:
        if os.path.isfile(args.vqa_v2_questions) and os.path.isfile(args.vqa_v2_annotations):
            ds_vqa_v2_real = VQAGenerativeDataset.from_vqa_v2(
                args.vqa_v2_questions, args.vqa_v2_annotations,
                q_vocab, a_vocab, h1_loader, use_butd=True,
            )

    ds_vqa_v2_mock = Subset(ds_expl_train, [i for i, idx in enumerate(indices[n_val:]) if ds_expl.annotations[idx].get("source") == 'vqa_v2'])
    if len(ds_vqa_v2_mock) == 0:
        ds_vqa_v2_mock = None

    vqa_mix = ds_vqa_v2_real if ds_vqa_v2_real is not None else ds_vqa_v2_mock

    sw = {'vqa_e': args.w_vqa_e, 'vqa_x': args.w_vqa_x, 'aokvqa': args.w_aokvqa, 'vqa_v2': 1.0}

    if args.phase == 1:
        if vqa_mix is not None:
            train_dataset, train_sampler = build_mixed_sampler([vqa_mix, ds_expl_train], [0.4, 0.6])
        elif args.weighted_sampling:
            train_dataset, train_sampler = build_subset_weighted_sampler(
                ds_expl_train, sw, quality_gamma=args.quality_gamma,
            )
        else:
            train_dataset, train_sampler = ds_expl_train, None
    elif args.phase in (2, 3):
        if vqa_mix is not None:
            train_dataset, train_sampler = build_replay_sampler(ds_expl_train, vqa_mix, replay_fraction=0.2)
        elif args.weighted_sampling:
            train_dataset, train_sampler = build_subset_weighted_sampler(
                ds_expl_train, sw, quality_gamma=args.quality_gamma,
            )
        else:
            train_dataset, train_sampler = ds_expl_train, None
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
        
    if torch.cuda.is_available() and hasattr(torch, 'compile') and not getattr(args, 'no_compile', False):
        try:
            model = torch.compile(model, mode='default', dynamic=True)
        except Exception:
            pass

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
    # FP16 benefits from loss scaling; BF16 does not
    use_scaler = torch.cuda.is_available() and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)

    start_epoch, best_val_loss, epochs_no_improve = 0, float('inf'), 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        _sd = _normalize_ckpt_state_dict(ckpt['model_state_dict'], model)
        _miss = model.load_state_dict(_sd, strict=False)
        if getattr(_miss, "missing_keys", None) and len(_miss.missing_keys) > 0:
            print(f"[resume] WARNING: missing_keys ({len(_miss.missing_keys)}): {list(_miss.missing_keys)[:8]} ...")
        if getattr(_miss, "unexpected_keys", None) and len(_miss.unexpected_keys) > 0:
            print(f"[resume] unexpected_keys ({len(_miss.unexpected_keys)}): {list(_miss.unexpected_keys)[:8]} ...")
        if ckpt.get('args', {}).get('phase', args.phase) == args.phase:
            start_epoch = ckpt.get('epoch', 0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            history = ckpt.get('history', history)
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                print(f"[resume] optimizer load skipped: {e}")
            if ckpt.get('cosine_sched_state_dict') is not None:
                try:
                    cosine_sched.load_state_dict(ckpt['cosine_sched_state_dict'])
                except Exception as e:
                    print(f"[resume] cosine scheduler load skipped: {e}")
            if warmup_sched is not None and ckpt.get('warmup_sched_state_dict') is not None:
                try:
                    warmup_sched.load_state_dict(ckpt['warmup_sched_state_dict'])
                except Exception as e:
                    print(f"[resume] warmup scheduler load skipped: {e}")

    glove_embed = None
    if args.scst and args.phase == 4 and args.ohp_lambda > 0:
        glove_embed = {w: model.decoder.embedding.weight[i].detach().cpu().numpy() for w, i in a_vocab.word2idx.items()}

    global_step = start_epoch * max(1, len(train_loader))
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        ep_loss, ep_acc, ep_gnorm = 0.0, 0.0, 0.0
        ep_entropy = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for step_idx, batch in enumerate(train_pbar):
            feats, questions, targets = batch.feats.to(DEVICE), batch.questions.to(DEVICE), batch.targets.to(DEVICE)
            img_mask = batch.img_mask.to(DEVICE) if batch.img_mask is not None else None
            length_bin = batch.length_bins.to(DEVICE) if getattr(batch, 'length_bins', None) is not None else None
            label_tokens = batch.label_tokens.to(DEVICE) if getattr(batch, 'label_tokens', None) is not None else None
            grid_feats = batch.grid_feats.to(DEVICE) if getattr(batch, 'grid_feats', None) is not None else None
            
            dec_in, dec_tgt = targets[:, :-1], targets[:, 1:]
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=amp_dtype):
                ss_on = getattr(args, 'scheduled_sampling', False) and args.phase == 3
                if ss_on:
                    rel_ep = epoch - start_epoch
                    epsilon = args.ss_k / (args.ss_k + math.exp(rel_ep / args.ss_k))
                    logits = _ss_forward_h(
                        model, questions, feats, dec_in, epsilon,
                        grid_feats=grid_feats, img_mask=img_mask,
                        length_bin=length_bin, label_tokens=label_tokens,
                    )
                    ce_loss = criterion(logits, dec_tgt)
                    loss = ce_loss
                    entropy_loss = torch.tensor(0.0, device=DEVICE)
                else:
                    out, cov_loss_scalar = model.forward_with_cov(
                        questions, feats, dec_in, img_mask=img_mask, length_bin=length_bin,
                        label_tokens=label_tokens, grid_feats=grid_feats,
                    )
                    ce_loss = criterion(out.logits, dec_tgt)
                    mac_attn = getattr(out, 'mac_attn', None)
                    entropy_loss = attention_entropy_penalty(mac_attn, eps=1e-7)
                    loss = ce_loss + 0.5 * cov_loss_scalar + args.entropy_mac_coef * entropy_loss

                if args.infonce and not ss_on and getattr(model, 'v_head', None) is not None:
                    ifo = model.get_infonce_loss(questions, feats, grid_feats, img_mask)
                    loss = loss + args.infonce_beta * ifo

                if ss_on:
                    with torch.no_grad():
                        preds = logits.argmax(dim=-1)
                else:
                    with torch.no_grad():
                        preds = out.logits.argmax(dim=-1)
                mask = dec_tgt != 0
                acc = (preds[mask] == dec_tgt[mask]).float().mean().item()
                ep_acc += acc
                if not ss_on:
                    ep_entropy += float(entropy_loss.detach().item())

                if args.scst and args.phase == 4:
                    from training.scst import scst_step
                    target_texts = []
                    for row in dec_tgt:
                        words = []
                        for tid in row.tolist():
                            if tid in (0, 2):
                                break
                            w = a_vocab.idx2word.get(tid, '<unk>') if hasattr(a_vocab, 'idx2word') else a_vocab.idx_to_word.get(tid, '<unk>')
                            words.append(w)
                        target_texts.append(' '.join(words))

                    lbl_names = getattr(batch, 'label_names', None)
                    if lbl_names is not None:
                        lbl_names = [list(x) if x is not None else [] for x in lbl_names]
                    rl_loss, _ = scst_step(
                        model, 'H', feats, questions, target_texts, a_vocab,
                        device=DEVICE, max_len=dec_in.size(1),
                        cider_weight=1.0,
                        bleu_weight=args.scst_bleu,
                        meteor_weight=args.scst_meteor,
                        exact_match_weight=args.scst_exact_match,
                        ohp_weight=args.ohp_lambda, glove_embed=glove_embed, return_stats=True,
                        grid_feats=grid_feats, img_mask=img_mask, label_tokens=label_tokens,
                        visual_labels_batch=lbl_names,
                    )
                    if rl_loss is not None:
                        loss = loss * 0.5 + rl_loss * 0.5

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            g_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g_norm += p.grad.detach().data.norm(2).item() ** 2
            ep_gnorm += g_norm ** 0.5

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            ep_loss += loss.item()
            global_step += 1
            train_pbar.set_postfix({'loss': f"{loss.item():.3f}", 'acc': f"{acc*100:.1f}%"})
            if _wb and not ss_on and step_idx % 50 == 0:
                try:
                    _wb.log({'train/entropy_mac': float(entropy_loss.detach().item()), 'train/step': global_step}, step=global_step)
                except Exception:
                    pass
            if args.dry_run and step_idx >= 5:
                return

        avg_loss, avg_acc = ep_loss / len(train_loader), ep_acc / len(train_loader)
        
        model.eval()
        val_loss, val_correct, val_tokens = 0.0, 0, 0
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
                    val_correct += (v_preds[v_m] == v_tgt[:, 1:][v_m]).sum().item()
                    val_tokens += int(v_m.sum().item())

        avg_val = val_loss / max(1, len(val_loader))
        avg_vacc = val_correct / max(val_tokens, 1)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Trn_Loss: {avg_loss:.4f} | Val_Loss: {avg_val:.4f} | Trn_Acc: {avg_acc*100:.1f}% | Val_Acc: {avg_vacc*100:.1f}% | LR: {cur_lr:.2e}")
        
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_val)
        history['lr'].append(cur_lr)
        with open(history_path, 'w') as f: json.dump(history, f, indent=2)
            
        if _wb:
            log_payload = {'train/loss': avg_loss, 'val/loss': avg_val, 'lr': cur_lr, 'epoch': epoch + 1,
                           'val/token_acc': avg_vacc}
            if ep_entropy > 0:
                log_payload['train/entropy_mac_mean'] = ep_entropy / max(1, len(train_loader))
            _wb.log(log_payload)
        
        # Warmup only for the first `warmup_epochs` of training (epoch index 0..warmup-1), not after resume.
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            warmup_sched.step()
        else:
            cosine_sched.step()

        sd = {k: v.cpu() for k, v in _model_state_dict_for_save(model).items()}
        ckpt_meta = {
            'epoch': epoch + 1,
            'model_state_dict': sd,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
            'args': vars(args),
            'cosine_sched_state_dict': cosine_sched.state_dict(),
        }
        if warmup_sched is not None:
            ckpt_meta['warmup_sched_state_dict'] = warmup_sched.state_dict()
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