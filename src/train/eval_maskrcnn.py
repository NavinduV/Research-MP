"""
Evaluate Mask R-CNN Model on Validation Dataset.

Computes metrics: IoU, F1 Score, AP50, and Accuracy for a trained Mask R-CNN model.
"""

import argparse
import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm

NUM_CLASSES = 2  # Binary: background + 1 type
CROP_SIZE = 128

class SingleTypeCropDataset(Dataset):
    """Dataset for ONE microplastic type with SAM masks."""
    def __init__(self, crops_dir: str, transforms=None, max_samples: int = None):
        self.crops_dir = Path(crops_dir)
        self.transforms = transforms

        ann_file = self.crops_dir / 'annotations.json'
        if not ann_file.exists():
            raise FileNotFoundError(f"annotations.json not found in {crops_dir}")

        with open(ann_file) as f:
            self.annotations = json.load(f)

        self.images_dir = self.crops_dir / 'images'
        self.masks_dir = self.crops_dir / 'masks'

        self.samples = [
            name for name in self.annotations
            if (self.images_dir / name).exists()
        ]

        if max_samples is not None and max_samples < len(self.samples):
            import random
            random.seed(42)
            self.samples = sorted(random.sample(self.samples, max_samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        ann = self.annotations[sample_name]

        img_path = self.images_dir / sample_name
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        mask = None
        mask_file = ann.get('mask_file')
        if mask_file and (self.masks_dir / mask_file).exists():
            raw = cv2.imread(str(self.masks_dir / mask_file), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                mask = (raw > 127).astype(np.uint8)

        if mask is None:
            default_mask = self.masks_dir / sample_name.replace('.png', '_mask.png')
            if default_mask.exists():
                raw = cv2.imread(str(default_mask), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    mask = (raw > 127).astype(np.uint8)

        if mask is None:
            mask = self._create_ellipse_mask(h, w, ann.get('rel_box'))

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            box = [xs.min(), ys.min(), xs.max(), ys.max()]
        else:
            margin = min(h, w) // 10
            box = [margin, margin, w - margin, h - margin]

        class_id = 1
        boxes = np.array([box], dtype=np.float32)
        labels = np.array([class_id], dtype=np.int64)
        masks_arr = np.array([mask], dtype=np.uint8)

        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist(),
                masks=list(masks_arr),
                class_labels=labels.tolist(),
            )
            image = transformed['image']
            if len(transformed['bboxes']) > 0:
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['class_labels'], dtype=np.int64)
                masks_arr = np.array(transformed['masks'], dtype=np.uint8)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(masks_arr, dtype=torch.uint8),
            'image_id': torch.tensor([idx]),
            'area': torch.as_tensor(
                [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32
            ),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return image, target, {
            'sample_name': sample_name,
            'original_size': (h, w),
            'gt_mask': mask,
            'gt_box': boxes[0] if len(boxes) > 0 else None,
        }

    @staticmethod
    def _create_ellipse_mask(h: int, w: int, rel_box=None):
        mask = np.zeros((h, w), dtype=np.uint8)
        if rel_box:
            x1, y1, x2, y2 = rel_box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        else:
            center = (w // 2, h // 2)
            axes = (int(w * 0.4), int(h * 0.4))
        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        return mask

def get_transforms(train: bool = False, img_size: int = CROP_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))

def collate_fn(batch):
    images = []
    targets = []
    metadata = []
    for image, target, meta in batch:
        images.append(image)
        targets.append(target)
        metadata.append(meta)
    return images, targets, metadata

def get_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    if pretrained:
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def compute_iou(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_box_iou(pred_box, gt_box):
    if pred_box is None or gt_box is None:
        return 0.0
    x1_inter = max(pred_box[0], gt_box[0])
    y1_inter = max(pred_box[1], gt_box[1])
    x2_inter = min(pred_box[2], gt_box[2])
    y2_inter = min(pred_box[3], gt_box[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = area_pred + area_gt - intersection
    if union == 0:
        return 0.0
    return intersection / union

def compute_ap50(ious, confidences):
    if len(ious) == 0:
        return 0.0
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_ious = np.array(ious)[sorted_indices]
    tp = (sorted_ious >= 0.5).astype(np.float32)
    fp = 1.0 - tp
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / max(len(ious), 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)
    ap = 0.0
    for i in range(len(precisions) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
    return ap

def compute_metrics(predictions, ground_truths, confidences, iou_threshold=0.5):
    matched_ious = []
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    # Track which GT boxes have been matched to prevent duplicate assignments
    matched_gts = set()
    
    # For each prediction, find the best matching GT
    for i, (pred_box, pred_mask) in enumerate(predictions):
        if pred_box is None or len(pred_box) == 0:
            fp_count += 1
            continue
            
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, (gt_box, gt_mask) in enumerate(ground_truths):
            if gt_box is None or len(gt_box) == 0 or j in matched_gts:
                continue
            box_iou = compute_box_iou(pred_box, gt_box)
            if box_iou > best_iou:
                best_iou = box_iou
                best_gt_idx = j
                
        matched_ious.append(best_iou)
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp_count += 1
            matched_gts.add(best_gt_idx)
            if pred_mask is not None and len(ground_truths[best_gt_idx][1]) > 0:
                mask_iou = compute_iou(pred_mask[0] if pred_mask.ndim == 3 else pred_mask, ground_truths[best_gt_idx][1])
                matched_ious[-1] = mask_iou
        else:
            fp_count += 1
            
    fn_count = len(ground_truths) - len(matched_gts)
    
    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    ap50 = compute_ap50(matched_ious, confidences)
    accuracy = tp_count / max(len(predictions), 1)
    return {
        'mean_iou': float(mean_iou),
        'f1_score': float(f1_score),
        'ap50': float(ap50),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp_count),
        'fp': int(fp_count),
        'fn': int(fn_count),
    }

@torch.no_grad()
def evaluate(model_path: str, crops_dir: str, device_str: str = None, batch_size: int = 8, conf_threshold: float = 0.5):
    device = torch.device(device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print(f"\n{'='*60}")
    print("EVALUATING MASK R-CNN")
    print(f"{'='*60}")
    print(f"  Device       : {device}")
    print(f"  Model        : {model_path}")
    print(f"  Crops dir    : {crops_dir}")
    print(f"  Conf thresh  : {conf_threshold}")
    print(f"{'='*60}\n")

    dataset = SingleTypeCropDataset(crops_dir=crops_dir, transforms=get_transforms(train=False))
    if len(dataset) == 0:
        print("ERROR: No validation samples found!")
        return None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model = get_model(NUM_CLASSES, pretrained=False)
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return None

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    all_ious = []
    all_predictions = []
    all_ground_truths = []
    all_confidences = []

    for images, targets, metadata in tqdm(dataloader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            scores = output['scores'].cpu().numpy()
            boxes = output['boxes'].cpu().numpy()
            masks = output['masks'].cpu().numpy()

            valid_idx = scores >= conf_threshold
            boxes_filtered = boxes[valid_idx]
            masks_filtered = masks[valid_idx]
            scores_filtered = scores[valid_idx]

            gt_target = targets[i]
            gt_boxes = gt_target['boxes'].numpy()
            gt_masks = gt_target['masks'].numpy()

            pred_boxes_list = list(boxes_filtered) if len(boxes_filtered) > 0 else [None]
            pred_masks_list = list(masks_filtered) if len(masks_filtered) > 0 else [None]
            gt_boxes_list = list(gt_boxes) if len(gt_boxes) > 0 else [None]
            gt_masks_list = list(gt_masks) if len(gt_masks) > 0 else [None]

            all_predictions.append(list(zip(pred_boxes_list, pred_masks_list)))
            all_ground_truths.append(list(zip(gt_boxes_list, gt_masks_list)))
            all_confidences.append(scores_filtered.tolist())

            if len(boxes_filtered) > 0 and len(gt_boxes) > 0:
                best_box_iou = compute_box_iou(boxes_filtered[0], gt_boxes[0])
                best_mask_iou = compute_iou(masks_filtered[0, 0], gt_masks[0]) if masks_filtered.size > 0 else 0.0
                all_ious.append(best_mask_iou)

    flat_predictions = []
    flat_ground_truths = []
    flat_confidences = []

    for preds, confs in zip(all_predictions, all_confidences):
        for pred in preds: flat_predictions.append(pred)
        for conf in confs: flat_confidences.append(conf)

    for gt in all_ground_truths:
        for g in gt: flat_ground_truths.append(g)

    metrics = compute_metrics(flat_predictions, flat_ground_truths, flat_confidences)
    if len(all_ious) > 0:
        metrics['sample_mean_iou'] = float(np.mean(all_ious))

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Mean IoU    : {metrics['mean_iou']:.4f}")
    print(f"  F1 Score    : {metrics['f1_score']:.4f}")
    print(f"  AP50        : {metrics['ap50']:.4f}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  TP/FP/FN    : {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    if 'sample_mean_iou' in metrics:
        print(f"  Sample Mean IoU : {metrics['sample_mean_iou']:.4f}")
    print(f"{'='*60}\n")
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--crops_dir', type=str, required=True, help='Path to crops')
    parser.add_argument('--device', type=str, default=None, help='Device')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Conf threshold')
    args = parser.parse_args()
    evaluate(args.model, args.crops_dir, args.device, args.batch_size, args.conf_threshold)