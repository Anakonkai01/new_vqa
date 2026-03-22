"""
extract_features_model_h.py — Pre-extract visual features for Model H (VQA-H)
==============================================================================
Detectron2 Faster R-CNN (e.g. ResNeXt-101) RoI features + FPN grid + per-box labels.

Output: e.g. data/vg_features/{image_id}.pt
  - 'region_feat': (k, 1029) = 1024-d box + 5 spatial (normalized xyxy + w*h)
  - 'grid_feat': 1D (e.g. 256-d from FPN p5 pool)
  - 'label_names': list[str] for PGN visual copy

Do NOT mix this output folder with Model F extraction (extract_features_model_f.py):
  different tensor width (1031) and keys ({'feat'} only).

Former name: extract_vg_features.py

Usage:
  python src/scripts/extract_features_model_h.py \
      --image_dir data/images \
      --output_dir data/vg_features \
      --top_k 36
"""

import torch
import torchvision
import os
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

# For maximum pre-Transformer capacity, we use a ResNet-152 based Faster R-CNN if available, 
# or a ResNet-50 FPN V2 with the updated Grid Extraction pipeline.
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model():
    """
    Load Faster R-CNN model.
    To reach SOTA, we MUST use a deeper backbone like ResNeXt-101-32x8d or ResNet-152.
    HOWEVER, PyTorch's native `torchvision` only provides pre-trained detection weights 
    for ResNet-50. 
    
    To use ResNet-152 / ResNeXt-101, you must install Facebook's Detectron2:
    `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`
    """
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo

        print("[INFO] Detectron2 found! Loading ResNeXt-101-32x8d Faster R-CNN...")
        cfg = get_cfg()
        # Load the massive ResNeXt-101 FPN config
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        # Ensure we don't return hundreds of proposals during training
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 200 
        predictor = DefaultPredictor(cfg)
        
        from detectron2.data import MetadataCatalog
        dataset_name = cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "coco_2017_train"
        meta = MetadataCatalog.get(dataset_name)
        return predictor, 'detectron2', meta.thing_classes
    except ImportError:
        raise ImportError("\n[CRITICAL ERROR] Model H has been strictly configured to ONLY use ResNeXt-101-32x8d!\n"
                          "Detectron2 was not found. Please install it using:\n"
                          "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'")

def extract_one(model, img_tensor, top_k=36, categories=None):
    """
    Extract top_k RoI features AND Grid features for a single image.
    """
    img_tensor = img_tensor.to(DEVICE)
    
    with torch.no_grad():
        images_norm, _ = model.transform([img_tensor], None)
        features = model.backbone(images_norm.tensors)
        
        # Grid feature: Average-pooled representation of the deepest FPN layer (e.g., '3' or 'pool')
        # This provides the scene-level context (Grid Feature)
        grid_map = features['3'] # Typically the deepest high-res feature map
        # Model H uses a flat 1D grid representation for the visual MAC memory
        grid_feat = torch.nn.functional.adaptive_avg_pool2d(grid_map, (1, 1)).view(-1).cpu()

        # RPN proposals
        proposals, _ = model.rpn(images_norm, features, None)
        props = proposals[0]
        
        props[:, 0::2].clamp_(0, images_norm.image_sizes[0][1])
        props[:, 1::2].clamp_(0, images_norm.image_sizes[0][0])
        
        k = min(top_k, len(props))
        selected = [props[:k]]
        
        # Region features
        box_feats_pooled = model.roi_heads.box_roi_pool(features, selected, images_norm.image_sizes)
        box_feats = model.roi_heads.box_head(box_feats_pooled)
        class_logits, _ = model.roi_heads.box_predictor(box_feats)
        pred_classes = class_logits.argmax(dim=1).tolist()
        
        label_names = []
        for c in pred_classes:
            if categories and c < len(categories):
                label_names.append(categories[c])
            else:
                label_names.append("background")
                
        boxes = selected[0].float()
        iW = images_norm.image_sizes[0][1]
        iH = images_norm.image_sizes[0][0]
        x1 = (boxes[:, 0] / iW).unsqueeze(1)
        y1 = (boxes[:, 1] / iH).unsqueeze(1)
        x2 = (boxes[:, 2] / iW).unsqueeze(1)
        y2 = (boxes[:, 3] / iH).unsqueeze(1)
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)
        
        spatial = torch.cat([x1, y1, x2, y2, w*h], dim=1)
        
        # Final output feature map: concatenation of visual and 5-dim spatial
        region_feat = torch.cat([box_feats, spatial], dim=1).cpu()

    return region_feat, grid_feat, label_names

def extract_batch_d2(predictor, images_cv2, top_k=36, thing_classes=None):
    """
    Extract top_k RoI features AND Grid features natively from Detectron2 for a batch of images.
    Returns: list of region_feats, list of grid_feats
    """
    with torch.no_grad():
        inputs = []
        for img_cv2 in images_cv2:
            height, width = img_cv2.shape[:2]
            # Use original size/transform as expected by predictor
            image = predictor.aug.get_transform(img_cv2).apply_image(img_cv2)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": height, "width": width})

        # Pad and batch images automatically via preprocess
        images = predictor.model.preprocess_image(inputs)
        
        # 1. Run Backbone
        features = predictor.model.backbone(images.tensor)
        
        # Grid feature: FPN deepest layer is usually 'p5'
        grid_map = features['p5'] if 'p5' in features else list(features.values())[-1]
        grid_feats_tensor = torch.nn.functional.adaptive_avg_pool2d(grid_map, (1, 1)).view(len(images_cv2), -1).cpu() # (B, 256)
        
        # 2. RPN Proposals
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposals_k = [p[:top_k] for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals_k]
        
        # 3. ROI Pooling -> Box Head
        features_list = [features[f] for f in predictor.model.roi_heads.box_in_features]
        box_features = predictor.model.roi_heads.box_pooler(features_list, proposal_boxes)
        box_features = predictor.model.roi_heads.box_head(box_features) # (sum(top_k), 1024)
        
        scores, _ = predictor.model.roi_heads.box_predictor(box_features)
        pred_classes = scores.argmax(dim=1)
        
        sizes = [len(p) for p in proposals_k]
        box_features_split = torch.split(box_features, sizes)
        pred_classes_split = torch.split(pred_classes, sizes)
        
        region_feats_list = []
        grid_feats_list = []
        label_names_list = []
        
        for i in range(len(images_cv2)):
            boxes = proposal_boxes[i].tensor # (k, 4)
            # FIX: Detectron2 proposal boxes are relative to the padded/resized image_sizes, not original inputs
            iH, iW = float(images.image_sizes[i][0]), float(images.image_sizes[i][1])
            
            x1 = (boxes[:, 0] / iW).unsqueeze(1)
            y1 = (boxes[:, 1] / iH).unsqueeze(1)
            x2 = (boxes[:, 2] / iW).unsqueeze(1)
            y2 = (boxes[:, 3] / iH).unsqueeze(1)
            w = (x2 - x1).clamp(min=0)
            h = (y2 - y1).clamp(min=0)
            
            spatial = torch.cat([x1, y1, x2, y2, w*h], dim=1).to(box_features_split[i].device) # (k, 5)
            region_feat = torch.cat([box_features_split[i], spatial], dim=1).cpu()
            
            img_classes = pred_classes_split[i].tolist()
            label_names = []
            for c in img_classes:
                if thing_classes and c < len(thing_classes):
                    label_names.append(thing_classes[c])
                else:
                    label_names.append("background")
            
            region_feats_list.append(region_feat)
            grid_feats_list.append(grid_feats_tensor[i])
            label_names_list.append(label_names)

        return region_feats_list, grid_feats_list, label_names_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing images")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for .pt files")
    parser.add_argument('--top_k', type=int, default=36, help="Number of boxes per image")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for extraction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, model_type, categories = build_model()
    
    valid_exts = ('.jpg', '.jpeg', '.png')
    all_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(valid_exts)]
    
    # Filter files that are already processed
    to_process = []
    for f in all_files:
        # COCO filenames look like: COCO_train2014_00000012345.jpg
        # We must extract the integer part natively to align exactly with dataset JSON 'img_id'.
        img_id_str = os.path.splitext(f)[0].split('_')[-1]
        try:
            img_id = int(img_id_str)
        except ValueError:
            img_id = img_id_str # Fallback for non-COCO formats
            
        out_path = os.path.join(args.output_dir, f"{img_id}.pt")
        if not os.path.exists(out_path):
            to_process.append((f, out_path))
            
    print(f"Skipped {len(all_files) - len(to_process)} existing files. Processing {len(to_process)} images in batches of {args.batch_size}...")
    
    for i in tqdm(range(0, len(to_process), args.batch_size)):
        batch_items = to_process[i:i+args.batch_size]
        valid_batch = []
        
        # Load batch
        for img_name, out_path in batch_items:
            img_path = os.path.join(args.image_dir, img_name)
            try:
                if model_type == 'detectron2':
                    import cv2
                    img_cv2 = cv2.imread(img_path)
                    if img_cv2 is None:
                        print(f"[SKIP] OpenCV cannot decode: {img_name}")
                        continue
                    valid_batch.append((img_cv2, out_path, img_name))
                else:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = TF.to_tensor(img)
                    valid_batch.append((img_tensor, out_path, img_name))
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                
        if not valid_batch:
            continue
            
        # Extract batch
        try:
            if model_type == 'detectron2':
                images_cv2 = [item[0] for item in valid_batch]
                region_feats, grid_feats, label_names_batch = extract_batch_d2(model, images_cv2, top_k=args.top_k, thing_classes=categories)
                
                for idx, (_, out_path, _) in enumerate(valid_batch):
                    torch.save({
                        'region_feat': region_feats[idx],
                        'grid_feat': grid_feats[idx],
                        'label_names': label_names_batch[idx]
                    }, out_path)
            else:
                # Fallback PyTorch processes sequentially as it's just a fallback
                for img_tensor, out_path, img_name in valid_batch:
                    region_feat, grid_feat, label_names_batch = extract_one(model, img_tensor, top_k=args.top_k, categories=categories)
                    torch.save({'region_feat': region_feat, 'grid_feat': grid_feat, 'label_names': label_names_batch}, out_path)
        except Exception as e:
            print(f"Error processing batch {i}-{i+args.batch_size}: {e}")

if __name__ == '__main__':
    main()
