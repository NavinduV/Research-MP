"""
Train Mask R-CNN on Cropped Detections from YOLO.

This script trains Mask R-CNN on cropped regions extracted using YOLO detections.
Instead of learning to detect AND segment on full images, Mask R-CNN only needs
to learn segmentation on simple, single-object crops.

Workflow:
1. Run YOLO on training images to get detections
2. Crop each detection with padding
3. Generate pseudo-masks for each crop (ellipse filling most of the crop)
4. Train Mask R-CNN on these crops

Usage:
    # Generate training crops from YOLO detections
    python src/train/train_maskrcnn_crops.py --mode prepare --images data/stitched --yolo experiments/microplastic_yolo/weights/best.pt
    
    # Train Mask R-CNN on crops
    python src/train/train_maskrcnn_crops.py --mode train --epochs 50
    
    # Test on a single crop
    python src/train/train_maskrcnn_crops.py --mode predict --image path/to/crop.png
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
from ultralytics import YOLO


# ============================================================================
# Configuration
# ============================================================================

NUM_CLASSES = 4  # background + fiber + film + fragment
CLASS_NAMES = ['background', 'fiber', 'film', 'fragment']
YOLO_TO_MASKRCNN = {0: 1, 1: 2, 2: 3}  # fiber=0->1, film=1->2, fragment=2->3

CROP_SIZE = 128  # Target crop size for training
PADDING = 30     # Padding around detection for cropping


# ============================================================================
# Dataset for Crop-Based Training
# ============================================================================

class CropDataset(Dataset):
    """Dataset of cropped detections for Mask R-CNN training.
    
    Supports two directory layouts:
      1. Flat: crops_dir/images/*.png + crops_dir/annotations.json
      2. Class-organized: crops_dir/{fiber,film,fragment}/*.png + crops_dir/annotations.json
    
    If annotations.json is missing but class subdirs exist, annotations are
    auto-generated from the directory structure (class inferred from subdir name).
    """
    
    CLASS_NAME_TO_YOLO_ID = {'fiber': 0, 'film': 1, 'fragment': 2}
    
    def __init__(self, crops_dir: str, transforms=None):
        """
        Args:
            crops_dir: Directory containing crops and annotations
            transforms: Albumentations transforms
        """
        self.crops_dir = Path(crops_dir)
        self.transforms = transforms
        
        # Determine layout and load / build annotations
        ann_file = self.crops_dir / 'annotations.json'
        has_flat_images = (self.crops_dir / 'images').is_dir()
        has_class_dirs = any((self.crops_dir / c).is_dir() for c in self.CLASS_NAME_TO_YOLO_ID)
        
        if ann_file.exists():
            # Preferred: use existing annotations.json
            with open(ann_file) as f:
                self.annotations = json.load(f)
            # Determine where images live
            if has_flat_images:
                self.images_dir = self.crops_dir / 'images'
            elif has_class_dirs:
                self.images_dir = None  # resolved per-sample from class_name
            else:
                self.images_dir = self.crops_dir / 'images'
        elif has_class_dirs:
            # Auto-generate annotations from class-organized subdirectories
            self.annotations = {}
            self.images_dir = None  # resolved per-sample
            for cls_name, cls_id in self.CLASS_NAME_TO_YOLO_ID.items():
                cls_dir = self.crops_dir / cls_name
                if not cls_dir.is_dir():
                    continue
                for img_file in sorted(cls_dir.glob('*.png')):
                    crop = cv2.imread(str(img_file))
                    if crop is None:
                        continue
                    h, w = crop.shape[:2]
                    self.annotations[img_file.name] = {
                        'source_image': '',
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'yolo_confidence': 1.0,
                        'rel_box': [0, 0, w, h],
                        'crop_size': [w, h]
                    }
            print(f"Auto-generated annotations for {len(self.annotations)} crops from class subdirs")
        else:
            raise FileNotFoundError(
                f"No annotations.json or class subdirs found in {crops_dir}. "
                "Run predict with the YOLO model first to generate crops."
            )
        
        self.samples = list(self.annotations.keys())
        print(f"Loaded {len(self.samples)} crop samples from {crops_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        ann = self.annotations[sample_name]
        
        # Load crop image - supports flat images/ dir or class subdirs
        if self.images_dir is not None:
            img_path = self.images_dir / sample_name
        else:
            # Resolve from class subdirectory
            cls_name = ann.get('class_name', '')
            img_path = self.crops_dir / cls_name / sample_name
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        
        # Load mask — prefer SAM mask, fall back to ellipse
        mask = None
        mask_file = ann.get('mask_file')
        masks_dir = self.crops_dir / 'masks'
        
        if mask_file and (masks_dir / mask_file).exists():
            # SAM mask from annotations (mask_file key)
            raw = cv2.imread(str(masks_dir / mask_file), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                mask = (raw > 127).astype(np.uint8)
        
        if mask is None:
            # Try default naming convention: <stem>_mask.png
            default_mask = masks_dir / sample_name.replace('.png', '_mask.png')
            if default_mask.exists():
                raw = cv2.imread(str(default_mask), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    mask = (raw > 127).astype(np.uint8)
        
        if mask is None:
            # Fallback: ellipse from rel_box
            mask = self._create_ellipse_mask(h, w, ann.get('rel_box'))
        
        # Resize mask to match image if needed
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Derive tight bounding box from mask
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            box = [xs.min(), ys.min(), xs.max(), ys.max()]
        else:
            # Mask is empty — fall back to padded crop bounds
            margin = min(h, w) // 10
            box = [margin, margin, w - margin, h - margin]
        
        # Class label
        class_id = YOLO_TO_MASKRCNN[ann['class_id']]
        
        # Prepare data
        boxes = np.array([box], dtype=np.float32)
        labels = np.array([class_id], dtype=np.int64)
        masks = np.array([mask], dtype=np.uint8)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist(),
                masks=list(masks),
                class_labels=labels.tolist()
            )
            image = transformed['image']
            if len(transformed['bboxes']) > 0:
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['class_labels'], dtype=np.int64)
                masks = np.array(transformed['masks'], dtype=np.uint8)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(masks, dtype=torch.uint8),
            'image_id': torch.tensor([idx]),
            'area': torch.as_tensor([(box[2]-box[0])*(box[3]-box[1]) for box in boxes], dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target
    
    def _create_ellipse_mask(self, h: int, w: int, rel_box=None):
        """Create an ellipse mask that fills most of the crop."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if rel_box:
            # Use relative box from detection
            x1, y1, x2, y2 = rel_box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        else:
            # Fill most of the crop
            center = (w // 2, h // 2)
            axes = (int(w * 0.4), int(h * 0.4))
        
        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        
        return mask


def get_transforms(train=True, img_size=128):
    """Get augmentation transforms for crops."""
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.03, 0.15), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))


def collate_fn(batch):
    """Custom collate for Mask R-CNN."""
    return tuple(zip(*batch))


# ============================================================================
# Model
# ============================================================================

def get_model(num_classes: int, pretrained: bool = True):
    """Create Mask R-CNN model."""
    if pretrained:
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = maskrcnn_resnet50_fpn(weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model


# ============================================================================
# Prepare Crops from YOLO Detections
# ============================================================================

def prepare_crops(images_dir: str, yolo_model_path: str, output_dir: str = 'data/crops',
                  yolo_conf: float = 0.25):
    """
    Generate training crops from YOLO detections.
    
    Args:
        images_dir: Directory containing training images
        yolo_model_path: Path to trained YOLO model
        output_dir: Directory to save crops
    """
    print(f"\n{'='*60}")
    print("PREPARING TRAINING CROPS FROM YOLO DETECTIONS")
    print(f"{'='*60}")
    print(f"Images directory: {images_dir}")
    print(f"YOLO model: {yolo_model_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load YOLO model
    yolo_model = YOLO(yolo_model_path)
    
    # Create output directories
    output_path = Path(output_dir)
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    
    # Find images
    images_path = Path(images_dir)
    image_files = list(images_path.glob('*.png')) + list(images_path.glob('*.jpg'))
    print(f"Found {len(image_files)} images")
    
    annotations = {}
    crop_count = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for img_path in tqdm(image_files, desc="Processing images"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Run YOLO detection
        results = yolo_model(str(img_path), conf=yolo_conf, verbose=False)[0]
        
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            # Add padding
            x1_pad = max(0, x1 - PADDING)
            y1_pad = max(0, y1 - PADDING)
            x2_pad = min(w, x2 + PADDING)
            y2_pad = min(h, y2 + PADDING)
            
            # Crop
            crop = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            
            # Skip tiny crops
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            
            # Save crop
            crop_name = f"{img_path.stem}_crop{crop_count:04d}.png"
            cv2.imwrite(str(output_path / 'images' / crop_name), crop)
            
            # Relative box within crop
            rel_box = [x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad]
            
            annotations[crop_name] = {
                'source_image': img_path.name,
                'class_id': class_id,
                'class_name': CLASS_NAMES[YOLO_TO_MASKRCNN[class_id]],
                'yolo_confidence': conf,
                'rel_box': rel_box,
                'crop_size': [crop.shape[1], crop.shape[0]]
            }
            
            class_counts[class_id] += 1
            crop_count += 1
    
    # Save annotations
    with open(output_path / 'annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n{'='*60}")
    print("CROP PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total crops generated: {crop_count}")
    print(f"  - Fiber: {class_counts[0]}")
    print(f"  - Film: {class_counts[1]}")
    print(f"  - Fragment: {class_counts[2]}")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")


# ============================================================================
# Prepare Crops from Original YOLO Labels (Ground Truth)
# ============================================================================

def prepare_from_yolo_labels(yolo_dir: str = 'data/yolo', output_dir: str = 'data/crops_gt',
                              padding: int = 20, splits=('train', 'val')):
    """
    Generate Mask R-CNN training crops directly from YOLO ground-truth labels.
    
    This uses the ORIGINAL human-annotated bounding boxes, not YOLO predictions,
    so the crops are guaranteed to be correct — no prediction errors propagate.
    
    YOLO label format per line:  class_id  cx  cy  w  h   (all normalised 0-1)
    
    Args:
        yolo_dir:   Root of the YOLO dataset (contains images/ and labels/ with train/val splits)
        output_dir: Where to save crops, images/, annotations.json and class subdirs
        padding:    Pixels of context around each bbox
        splits:     Which splits to process
    """
    yolo_path = Path(yolo_dir)
    out_path = Path(output_dir)
    (out_path / 'images').mkdir(parents=True, exist_ok=True)
    for cls_name in CLASS_NAMES[1:]:   # fiber, film, fragment
        (out_path / cls_name).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PREPARING MASK R-CNN CROPS FROM YOLO GROUND-TRUTH LABELS")
    print(f"{'='*60}")
    print(f"YOLO directory : {yolo_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Padding        : {padding}px")
    print(f"Splits         : {splits}")
    print(f"{'='*60}\n")

    annotations = {}
    crop_count = 0
    class_counts = {0: 0, 1: 0, 2: 0}

    for split in splits:
        img_dir = yolo_path / 'images' / split
        lbl_dir = yolo_path / 'labels' / split

        if not img_dir.exists():
            print(f"Skipping split '{split}' — {img_dir} not found")
            continue

        image_files = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
        print(f"\n[{split}] Found {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  Warning: could not read {img_path}")
                continue

            h, w = image.shape[:2]

            # Corresponding label file
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if not lbl_path.exists():
                continue

            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue

                cls_id = int(float(parts[0]))
                cx_n, cy_n, bw_n, bh_n = map(float, parts[1:5])

                # Convert normalised → pixel coords
                cx = cx_n * w
                cy = cy_n * h
                bw = bw_n * w
                bh = bh_n * h
                det_x1 = int(cx - bw / 2)
                det_y1 = int(cy - bh / 2)
                det_x2 = int(cx + bw / 2)
                det_y2 = int(cy + bh / 2)

                # Crop with padding, clipped to image bounds
                x1 = max(0, det_x1 - padding)
                y1 = max(0, det_y1 - padding)
                x2 = min(w, det_x2 + padding)
                y2 = min(h, det_y2 + padding)

                crop_img = image[y1:y2, x1:x2]
                if crop_img.size == 0 or crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
                    continue

                cls_name = CLASS_NAMES[YOLO_TO_MASKRCNN[cls_id]]  # e.g. 'fiber'
                crop_filename = f"{img_path.stem}_gt{crop_count:04d}_{cls_name}.png"

                # Save in flat images/ and class subdirectory
                cv2.imwrite(str(out_path / 'images' / crop_filename), crop_img)
                cv2.imwrite(str(out_path / cls_name / crop_filename), crop_img)

                # Relative box within the crop
                rel_x1 = det_x1 - x1
                rel_y1 = det_y1 - y1
                rel_x2 = det_x2 - x1
                rel_y2 = det_y2 - y1

                annotations[crop_filename] = {
                    'source_image': img_path.name,
                    'split': split,
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'yolo_confidence': 1.0,   # ground-truth → perfect confidence
                    'rel_box': [rel_x1, rel_y1, rel_x2, rel_y2],
                    'crop_size': [crop_img.shape[1], crop_img.shape[0]]
                }

                class_counts[cls_id] += 1
                crop_count += 1

    # Save annotations
    with open(out_path / 'annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"\n{'='*60}")
    print("CROP PREPARATION COMPLETE (from YOLO ground-truth)")
    print(f"{'='*60}")
    print(f"Total crops: {crop_count}")
    print(f"  Fiber    : {class_counts[0]}")
    print(f"  Film     : {class_counts[1]}")
    print(f"  Fragment : {class_counts[2]}")
    print(f"Saved to   : {out_path}")
    print(f"{'='*60}\n")


# ============================================================================
# Training
# ============================================================================

def train(crops_dir: str = 'data/crops', epochs: int = 50, batch_size: int = 8,
          lr: float = 0.001, save_dir: str = 'experiments', resume: str = None):
    """Train Mask R-CNN on cropped detections."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print("TRAINING MASK R-CNN ON CROPS")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Crops directory: {crops_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}\n")
    
    # Create dataset
    dataset = CropDataset(
        crops_dir=crops_dir,
        transforms=get_transforms(train=True, img_size=CROP_SIZE)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create model
    model = get_model(NUM_CLASSES, pretrained=True)
    model.to(device)
    
    # Resume if specified
    start_epoch = 0
    if resume and Path(resume).exists():
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Training loop
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Skip if no valid boxes
            if not all(len(t['boxes']) > 0 for t in targets):
                continue
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += losses.item()
            pbar.set_postfix({'loss': losses.item()})
        
        lr_scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, f"{save_dir}/maskrcnn_crops_latest.pth")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, f"{save_dir}/maskrcnn_crops_best.pth")
            print(f"  -> New best model saved!")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best model: {save_dir}/maskrcnn_crops_best.pth")
    print(f"{'='*60}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on YOLO crops')
    parser.add_argument('--mode', type=str, choices=['prepare', 'prepare-gt', 'train'], required=True,
                        help='Mode: prepare (crops from YOLO model), prepare-gt (crops from YOLO labels), or train')
    parser.add_argument('--images', type=str, default='data/stitched',
                        help='Image directory for crop preparation')
    parser.add_argument('--yolo', type=str, default='experiments/microplastic_yolo/weights/best.pt',
                        help='YOLO model for detection')
    parser.add_argument('--yolo-dir', type=str, default='data/yolo',
                        help='YOLO dataset directory (for prepare-gt mode)')
    parser.add_argument('--crops-dir', type=str, default='data/crops',
                        help='Directory for crop data')
    parser.add_argument('--padding', type=int, default=20,
                        help='Padding around bbox when cropping (pixels)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--yolo-conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'prepare':
        prepare_crops(
            images_dir=args.images,
            yolo_model_path=args.yolo,
            output_dir=args.crops_dir,
            yolo_conf=args.yolo_conf
        )
    
    elif args.mode == 'prepare-gt':
        prepare_from_yolo_labels(
            yolo_dir=args.yolo_dir,
            output_dir=args.crops_dir,
            padding=args.padding,
        )
    
    elif args.mode == 'train':
        train(
            crops_dir=args.crops_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            resume=args.resume
        )


if __name__ == "__main__":
    main()

