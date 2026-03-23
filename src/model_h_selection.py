"""Lightweight official-val selector for Model H checkpoints during training."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import nltk
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader, Dataset

from inference_policy import build_length_bin_tensor, resolve_min_decode_len
from text_contract import normalize_text, split_answer_explanations, strip_empty_explanations

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


DATASET_PATHS = {
    "vqa_e": "data/annotations/vqa_e/vqa_e_val_unified.json",
    "vqa_x": "data/annotations/vqa_x/vqa_x_val_unified.json",
    "aokvqa": "data/annotations/aokvqa/aokvqa_val_unified.json",
}


class OfficialValDatasetH(Dataset):
    """Unified official validation reader for Model H selection-time eval."""

    def __init__(self, json_path, q_vocab, feature_loader, max_q_len=20, max_samples=None):
        with open(json_path, "r") as f:
            self.anns = json.load(f)
        if max_samples is not None:
            self.anns = self.anns[:max_samples]
        self.q_vocab = q_vocab
        self.feature_loader = feature_loader
        self.max_q_len = max_q_len

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_id = ann["img_id"]
        region_feat, grid_feat, labels = self.feature_loader(img_id)
        q_tokens = self.q_vocab.numericalize(ann["question"])[: self.max_q_len]
        q_tensor = torch.tensor(q_tokens, dtype=torch.long)
        gt_answer = normalize_text(ann.get("multiple_choice_answer", "")).lower()
        gt_expls = ann.get("explanation", []) or [gt_answer]
        feat_dict = {
            "region_feat": region_feat,
            "grid_feat": grid_feat,
            "label_names": labels,
        }
        return feat_dict, q_tensor, gt_answer, gt_expls


def collate_official_val(batch):
    feats, q_tensors, gt_answers, gt_expls_list = zip(*batch)

    max_q = max(q.shape[0] for q in q_tensors)
    q_padded = torch.zeros(len(q_tensors), max_q, dtype=torch.long)
    for i, q in enumerate(q_tensors):
        q_padded[i, : q.shape[0]] = q

    region_feats = [f["region_feat"] for f in feats]
    max_r = max(r.shape[0] for r in region_feats)
    v_dim = region_feats[0].shape[1]
    region_padded = torch.zeros(len(region_feats), max_r, v_dim)
    region_mask = torch.zeros(len(region_feats), max_r, dtype=torch.bool)
    for i, r in enumerate(region_feats):
        region_padded[i, : r.shape[0]] = r
        region_mask[i, : r.shape[0]] = True

    grid_feats = [f.get("grid_feat", None) for f in feats]
    grid_padded = None
    grid_valid = None
    if any(g is not None for g in grid_feats):
        ref = next(g for g in grid_feats if g is not None)
        rows, flags = [], []
        for g in grid_feats:
            if g is None:
                rows.append(torch.zeros_like(ref))
                flags.append(False)
            else:
                rows.append(g)
                flags.append(True)
        grid_padded = torch.stack(rows, dim=0)
        grid_valid = torch.tensor(flags, dtype=torch.bool)

    label_names = [f.get("label_names", None) for f in feats]
    return (
        region_padded,
        region_mask,
        q_padded,
        grid_padded,
        grid_valid,
        label_names,
        list(gt_answers),
        list(gt_expls_list),
    )


def _labels_to_token_tensor(label_names_batch, a_vocab, device):
    if not label_names_batch or all(v is None for v in label_names_batch):
        return None

    unk = a_vocab.word2idx.get("<unk>", 3)
    mats = []
    for names in label_names_batch:
        if not names:
            mats.append(torch.zeros(1, 1, dtype=torch.long))
            continue
        tok_lists = []
        for name in names:
            toks = [a_vocab.word2idx.get(w.lower(), unk) for w in str(name).split()]
            tok_lists.append(toks or [unk])
        max_t = max(len(t) for t in tok_lists)
        m = torch.zeros(len(tok_lists), max_t, dtype=torch.long)
        for i, t in enumerate(tok_lists):
            m[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        mats.append(m)

    max_k = max(m.size(0) for m in mats)
    max_t = max(m.size(1) for m in mats)
    out = torch.zeros(len(mats), max_k, max_t, dtype=torch.long)
    for b, m in enumerate(mats):
        out[b, : m.size(0), : m.size(1)] = m
    return out.to(device)


@torch.no_grad()
def greedy_decode_batch_h(
    model,
    region_feat,
    region_mask,
    q_ids,
    grid_feat,
    grid_valid,
    label_names,
    a_vocab,
    device,
    max_len=30,
    min_decode_len=None,
    dataset_name=None,
    length_bin_policy="auto",
):
    B = region_feat.size(0)
    sos = a_vocab.word2idx.get("<start>", 1)
    eos = a_vocab.word2idx.get("<end>", 2)
    pad = a_vocab.word2idx.get("<pad>", 0)
    min_decode_len = resolve_min_decode_len(min_decode_len, dataset_name)

    region_feat = region_feat.to(device)
    region_mask = region_mask.to(device)
    q_ids = q_ids.to(device)
    grid_feat = grid_feat.to(device) if grid_feat is not None else None
    grid_valid_d = grid_valid.to(device) if grid_valid is not None else None
    label_tokens = _labels_to_token_tensor(label_names, a_vocab, device)

    dec_input = torch.full((B, 1), sos, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    outputs = [[] for _ in range(B)]

    memory, q_hidden, _, kb, _v_proj_raw, _ = model.encode(
        q_ids,
        region_feat,
        grid_feats=grid_feat,
        img_mask=region_mask,
        grid_valid=grid_valid_d,
    )
    h, c = model.init_decoder_hidden(memory)
    mm = None if getattr(model.args, "no_mac_decoder", False) else memory
    coverage = None
    length_bin = build_length_bin_tensor(B, device, length_policy=length_bin_policy, dataset_name=dataset_name)

    for _ in range(max_len):
        logit, h, c, _, coverage = model.decode_step(
            dec_input[:, -1:].contiguous(),
            h,
            c,
            kb,
            q_hidden,
            coverage=coverage,
            img_mask=region_mask,
            q_token_ids=q_ids,
            length_bin=length_bin,
            label_tokens=label_tokens,
            mac_memory=mm,
        )
        next_ids = []
        for b in range(B):
            if finished[b]:
                next_ids.append(eos)
                continue
            lp = logit[b]
            if len(outputs[b]) < min_decode_len:
                lp = lp.clone()
                lp[eos] = float("-inf")
            tok = int(lp.argmax())
            if tok == eos:
                finished[b] = True
            else:
                outputs[b].append(tok)
            next_ids.append(tok)
        if finished.all():
            break
        dec_input = torch.cat(
            [dec_input, torch.tensor(next_ids, device=device, dtype=torch.long).unsqueeze(1)],
            dim=1,
        )

    special = {pad, sos, eos, a_vocab.word2idx.get("<unk>", 3)}
    results = []
    for toks in outputs:
        words = [a_vocab.idx2word[t] for t in toks if t not in special]
        results.append(" ".join(words))
    return results


def compute_selection_metrics(preds, gt_expls_list, gt_answers):
    smoothie = SmoothingFunction().method1
    pred_answers, pred_expls = split_answer_explanations(preds)
    pred_answers = [normalize_text(x).lower() for x in pred_answers]
    pred_expls = strip_empty_explanations(pred_expls)

    ans_exact = sum(
        p == normalize_text(a).lower()
        for p, a in zip(pred_answers, gt_answers)
    ) / max(1, len(preds))

    bleu4 = 0.0
    meteor = 0.0
    for pred_expl, refs in zip(pred_expls, gt_expls_list):
        ref_ws = [normalize_text(r).split() or ["<unk>"] for r in refs]
        pred_w = pred_expl.split() or ["<unk>"]
        bleu4 += sentence_bleu(
            ref_ws,
            pred_w,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
        meteor += max(meteor_score([rw], pred_w) for rw in ref_ws)

    n = max(1, len(preds))
    bleu4 /= n
    meteor /= n
    selection_score = (ans_exact + bleu4 + meteor) / 3.0
    return {
        "ans_exact": ans_exact,
        "bleu4": bleu4,
        "meteor": meteor,
        "selection_score": selection_score,
    }


def evaluate_official_val_selection(
    model,
    q_vocab,
    a_vocab,
    feat_loader,
    device,
    datasets: Sequence[str],
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    max_len: int = 30,
    min_decode_len: Optional[int] = None,
    length_bin_policy: str = "auto",
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """Run a lightweight greedy benchmark on official val subsets."""
    per_dataset: List[Dict[str, float]] = []

    for ds_name in datasets:
        json_path = DATASET_PATHS.get(ds_name)
        if json_path is None or not os.path.exists(json_path):
            continue

        ds = OfficialValDatasetH(
            json_path,
            q_vocab,
            feat_loader,
            max_q_len=20,
            max_samples=max_samples,
        )
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "collate_fn": collate_official_val,
            "num_workers": num_workers,
            "pin_memory": (device.type == "cuda"),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        loader = DataLoader(ds, **loader_kwargs)

        preds = []
        gt_answers = []
        gt_expls_list = []
        for region_feat, region_mask, q_ids, grid_feat, grid_valid, label_names, batch_answers, batch_refs in loader:
            batch_preds = greedy_decode_batch_h(
                model,
                region_feat,
                region_mask,
                q_ids,
                grid_feat,
                grid_valid,
                label_names,
                a_vocab,
                device,
                max_len=max_len,
                min_decode_len=min_decode_len,
                dataset_name=ds_name,
                length_bin_policy=length_bin_policy,
            )
            preds.extend(batch_preds)
            gt_answers.extend(batch_answers)
            gt_expls_list.extend(batch_refs)

        metrics = compute_selection_metrics(preds, gt_expls_list, gt_answers)
        metrics["dataset"] = ds_name
        per_dataset.append(metrics)

    macro_score = 0.0
    if per_dataset:
        macro_score = sum(d["selection_score"] for d in per_dataset) / len(per_dataset)

    return {
        "macro_score": macro_score,
        "datasets": per_dataset,
    }
