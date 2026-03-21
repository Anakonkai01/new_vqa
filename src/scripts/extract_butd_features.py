"""
extract_butd_features.py — Tier 3B pre-extraction script
=========================================================
Extracts Faster R-CNN RoI features from COCO images and saves them to disk.
Must be run ONCE before training with --model F.

Output: data/butd_features/{image_id}.pt
  Each file contains: {'feat': Tensor(k, 1029)}
  feat_dim = 1024 (ResNet50 FPN box_head) + 5 (spatial: x1/W, y1/H, x2/W, y2/H, area)
  k = number of proposals kept (top_k by objectness score, default 36)

Usage:
  python src/scripts/extract_butd_features.py \\
      --splits train2014 val2014 \\
      --image_dir data/images \\
      --output_dir data/butd_features \\
      --top_k 36 \\
      --batch_size 1

Time estimate: ~3-4 hours for full COCO on a single GPU.
              Use --max_images N for a quick test run.
"""

import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from PIL import Image
import os
import argparse
import json
from tqdm import tqdm


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    """Load Faster R-CNN ResNet50 FPN v2 (pretrained on COCO)."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()
    model.to(DEVICE)
    # Increase proposals so top_k can always be satisfied
    model.rpn.post_nms_top_n_test   = 200
    model.rpn._post_nms_top_n_test  = 200
    model.roi_heads.detections_per_img = 200
    return model


def extract_one(model, img_tensor, top_k=36):
    """
    Extract top_k RoI features for a single image.

    Strategy:
      1. Run backbone + FPN to get multi-scale feature maps.
      2. Run RPN to get object proposals (ranked by objectness score).
      3. Apply RoI pooling + box_head MLP on top proposals.
      4. Return top_k feature vectors + normalized spatial coordinates.

    Returns: Tensor (k, 1029) on CPU
      Columns: [box_features (1024), x1/W, y1/H, x2/W, y2/H, area/(W*H)]
    """
    img_tensor = img_tensor.to(DEVICE)
    W = img_tensor.shape[2]
    H = img_tensor.shape[1]

    with torch.no_grad():
        # Step 1: backbone features
        images_norm, _ = model.transform([img_tensor], None)
        features = model.backbone(images_norm.tensors)   # FPN dict

        # Step 2: RPN proposals (sorted by objectness descending)
        proposals, _ = model.rpn(images_norm, features, None)  # list of (N, 4)
        props = proposals[0]  # (N, 4) — xyxy format, image-space coords

        # Clamp to valid range
        props[:, 0::2].clamp_(0, images_norm.image_sizes[0][1])  # x
        props[:, 1::2].clamp_(0, images_norm.image_sizes[0][0])  # y

        k = min(top_k, len(props))
        selected = [props[:k]]   # keep top-k by objectness (RPN already sorted)

        # Step 3: RoI pool + box_head
        box_feats_pooled = model.roi_heads.box_roi_pool(
            features, selected, images_norm.image_sizes)     # (k, 256, 7, 7)
        box_feats = model.roi_heads.box_head(box_feats_pooled)  # (k, 1024)

        # Step 4: normalized spatial features
        boxes = selected[0].float()
        iW = images_norm.image_sizes[0][1]
        iH = images_norm.image_sizes[0][0]
        x1 = (boxes[:, 0] / iW).unsqueeze(1)
        y1 = (boxes[:, 1] / iH).unsqueeze(1)
        x2 = (boxes[:, 2] / iW).unsqueeze(1)
        y2 = (boxes[:, 3] / iH).unsqueeze(1)
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)
        area = (w * h).clamp(min=0)

        spatial = torch.cat([x1, y1, x2, y2, w, h, area], dim=1)   # (k, 7)
        feat = torch.cat([box_feats, spatial], dim=1).cpu()   # (k, 1031)

    return feat


def process_split(model, split, image_dir, output_dir, top_k, max_images):
    """Process all images in one COCO split."""
    split_dir = os.path.join(image_dir, split)
    if not os.path.isdir(split_dir):
        print(f"  [SKIP] {split_dir} not found.")
        return

    images = sorted(f for f in os.listdir(split_dir) if f.endswith('.jpg'))
    if max_images:
        images = images[:max_images]

    to_tensor = torchvision.transforms.ToTensor()
    skipped   = 0

    for fname in tqdm(images, desc=split):
        # image_id is the numeric part of the filename
        img_id_str = fname.replace('.jpg', '').split('_')[-1]
        out_path   = os.path.join(output_dir, f"{int(img_id_str)}.pt")

        if os.path.exists(out_path):
            continue   # already extracted

        try:
            img  = Image.open(os.path.join(split_dir, fname)).convert('RGB')
            feat = extract_one(model, to_tensor(img), top_k=top_k)
            torch.save({'feat': feat}, out_path)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  [WARN] {fname}: {e}")

    if skipped:
        print(f"  {skipped} images skipped due to errors.")


def main():
    parser = argparse.ArgumentParser(description="Extract BUTD RoI features from COCO images.")
    parser.add_argument('--splits',     nargs='+', default=['train2014', 'val2014'])
    parser.add_argument('--image_dir',  type=str,  default='data/images',
                        help='Root dir containing train2014/ and val2014/ subdirs')
    parser.add_argument('--output_dir', type=str,  default='data/butd_features',
                        help='Directory to save .pt feature files')
    parser.add_argument('--top_k',      type=int,  default=36,
                        help='Number of region proposals to keep per image')
    parser.add_argument('--max_images', type=int,  default=None,
                        help='Cap number of images per split (None = all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device    : {DEVICE}")
    print(f"Output dir: {args.output_dir}")
    print(f"Top-K     : {args.top_k}")
    print("Loading Faster R-CNN ResNet50 FPN v2 ...")
    model = build_model()
    print("Model loaded.")

    for split in args.splits:
        print(f"\nProcessing split: {split}")
        process_split(model, split, args.image_dir, args.output_dir,
                      args.top_k, args.max_images)

    print("\nExtraction complete.")
    print(f"Feature files saved to: {args.output_dir}")
    print("Next: train with --model F --butd_feat_dir data/butd_features")


if __name__ == '__main__':
    main()
