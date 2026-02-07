"""
Mask R-CNN Training for Microplastic Segmentation.

This script trains a Mask R-CNN model for instance segmentation of microplastics.
It supports COCO-format annotations and can generate pseudo-masks from bounding boxes
if segmentation data is not available.

Classes:
    0: background (not used in annotations, but present in model output)
    1: fiber
    2: film
    3: fragment

Usage:
    # Train with existing data
    python src/train_maskrcnn.py --mode train --epochs 50
    
    # Validate model
    python src/train_maskrcnn.py --mode val --model experiments/maskrcnn_best.pth
    
    # Predict on an image
    python src/train_maskrcnn.py --mode predict --model experiments/maskrcnn_best.pth --image path/to/image.png
"""

import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

NUM_CLASSES = 4  # background + fiber + film + fragment
CLASS_NAMES = ['background', 'fiber', 'film', 'fragment']

# Map from annotation category_id to model class (COCO uses 0-indexed, we use 1-indexed for model)
# In annotations: Fiber=0, Film=1, Fragment=2
# In model: background=0, fiber=1, film=2, fragment=3
CATEGORY_MAP = {0: 1, 1: 2, 2: 3}  # annotation_id -> model_class


# ============================================================================
# Dataset
# ============================================================================

class MicroplasticMaskRCNNDataset(Dataset):
    """
    Dataset for Mask R-CNN training with COCO-style annotations.
    
    Supports:
    - Polygon segmentation masks
    - Generating pseudo-masks from bounding boxes (if no segmentation)
    - Image augmentation
    """
    
    def __init__(self, annotation_file: str, image_dir: str = None, 
                 transforms=None, generate_pseudo_masks: bool = True):
        """
        Args:
            annotation_file: Path to COCO-format annotations JSON
            image_dir: Override directory for images (if None, uses paths from annotations)
            transforms: Albumentations transforms
            generate_pseudo_masks: If True, generate ellipse masks from bboxes when segmentation is empty
        """
        with open(annotation_file) as f:
            self.coco = json.load(f)
        
        self.image_dir = image_dir
        self.transforms = transforms
        self.generate_pseudo_masks = generate_pseudo_masks
        
        # Index images by id for fast lookup
        self.images = {img['id']: img for img in self.coco['images']}
        
        # Group annotations by image_id
        self.image_annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # List of valid image ids (those with annotations)
        self.valid_image_ids = list(self.image_annotations.keys())
        
        print(f"Loaded {len(self.valid_image_ids)} images with {len(self.coco['annotations'])} annotations")
    
    def __len__(self):
        return len(self.valid_image_ids)
    
    def _get_image_path(self, img_info):
        """Resolve actual image path from annotation info."""
        file_name = img_info['file_name']
        
        # If image_dir is specified, use it
        if self.image_dir:
            # Extract just the filename
            base_name = Path(file_name).name
            return Path(self.image_dir) / base_name
        
        # Try to find the image in common locations
        possible_paths = [
            Path(file_name),
            Path('data/stitched') / Path(file_name).name,
            Path('data/yolo/images/train') / Path(file_name).name,
            Path('data/annotations/images') / Path(file_name).name,
        ]
        
        for p in possible_paths:
            if p.exists():
                return p
        
        # Return as-is and let it fail with a clear error
        return Path(file_name)
    
    def _generate_mask_from_bbox(self, bbox, height, width, shape='ellipse'):
        """Generate a pseudo-mask from bounding box."""
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = [int(v) for v in bbox]
        
        if shape == 'ellipse':
            # Create an ellipse that fills the bounding box
            center = (x + w // 2, y + h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        else:
            # Rectangle mask
            cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)
        
        return mask
    
    def __getitem__(self, idx):
        img_id = self.valid_image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self._get_image_path(img_info)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        
        # Get annotations for this image
        anns = self.image_annotations[img_id]
        
        boxes = []
        labels = []
        masks = []
        areas = []
        
        for ann in anns:
            # Get bounding box [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Clip to image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2, y2])
            
            # Map category to model class (add 1 for background class)
            cat_id = ann['category_id']
            labels.append(CATEGORY_MAP.get(cat_id, cat_id + 1))
            
            # Get or generate mask
            if ann['segmentation'] and len(ann['segmentation']) > 0:
                # Use provided polygon segmentation
                mask = np.zeros((height, width), dtype=np.uint8)
                for poly in ann['segmentation']:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
            elif self.generate_pseudo_masks:
                # Generate pseudo-mask from bbox
                mask = self._generate_mask_from_bbox(ann['bbox'], height, width, shape='ellipse')
            else:
                # Empty mask (not recommended)
                mask = np.zeros((height, width), dtype=np.uint8)
            
            masks.append(mask)
            areas.append((x2 - x1) * (y2 - y1))
        
        # Handle edge case: no valid annotations
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            masks = np.zeros((0, height, width), dtype=np.uint8)
            areas = np.zeros((0,), dtype=np.float32)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            masks = np.array(masks, dtype=np.uint8)
            areas = np.array(areas, dtype=np.float32)
        
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
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
                masks = np.zeros((0, height, width), dtype=np.uint8)
        else:
            # Normalize image manually
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Create target dict
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(masks, dtype=torch.uint8),
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target


def get_transforms(train=True, img_size=800):
    """Get augmentation transforms."""
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1),
            ], p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))


def collate_fn(batch):
    """Custom collate function for Mask R-CNN."""
    return tuple(zip(*batch))


# ============================================================================
# Model
# ============================================================================

def get_model(num_classes: int, pretrained: bool = True):
    """
    Create Mask R-CNN model with custom number of classes.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Use pretrained backbone
    """
    # Load pretrained model
    if pretrained:
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = maskrcnn_resnet50_fpn(weights=None)
    
    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model


# ============================================================================
# Training
# ============================================================================

def train(annotation_file: str, image_dir: str = None, epochs: int = 50, 
          batch_size: int = 2, lr: float = 0.001, img_size: int = 800,
          save_dir: str = 'experiments', resume: str = None):
    """
    Train Mask R-CNN model.
    
    Args:
        annotation_file: Path to COCO annotations
        image_dir: Override image directory
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        img_size: Image size for training
        save_dir: Directory to save models
        resume: Path to checkpoint to resume from
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("MASK R-CNN TRAINING - MICROPLASTIC SEGMENTATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Annotations: {annotation_file}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Image size: {img_size}")
    print(f"{'='*60}\n")
    
    # Create dataset
    dataset = MicroplasticMaskRCNNDataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        transforms=get_transforms(train=True, img_size=img_size),
        generate_pseudo_masks=True
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=collate_fn
    )
    
    # Create model
    model = get_model(NUM_CLASSES, pretrained=True)
    model.to(device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume and Path(resume).exists():
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Training loop
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        loss_components = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_mask': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, targets in pbar:
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Skip batch if no valid targets
            valid_batch = all(len(t['boxes']) > 0 for t in targets)
            if not valid_batch:
                continue
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            # Track losses
            epoch_loss += losses.item()
            for k, v in loss_dict.items():
                if k in loss_components:
                    loss_components[k] += v.item()
            
            pbar.set_postfix({'loss': losses.item()})
        
        # Update learning rate
        lr_scheduler.step()
        
        # Calculate average losses
        n_batches = len(dataloader)
        avg_loss = epoch_loss / n_batches
        
        print(f"\nEpoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        print(f"  Classifier: {loss_components['loss_classifier']/n_batches:.4f}, "
              f"Box Reg: {loss_components['loss_box_reg']/n_batches:.4f}, "
              f"Mask: {loss_components['loss_mask']/n_batches:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, f"{save_dir}/maskrcnn_latest.pth")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, f"{save_dir}/maskrcnn_best.pth")
            print(f"  -> New best model saved!")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best model saved to: {save_dir}/maskrcnn_best.pth")
    print(f"{'='*60}")


# ============================================================================
# Inference
# ============================================================================

def predict(model_path: str, image_path: str, output_dir: str = 'experiments/maskrcnn_predictions',
            conf_threshold: float = 0.5, mask_threshold: float = 0.5):
    """
    Run inference on an image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        output_dir: Directory to save predictions
        conf_threshold: Confidence threshold for detections
        mask_threshold: Threshold for mask binarization
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print("MASK R-CNN INFERENCE")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"{'='*60}\n")
    
    # Load model
    model = get_model(NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for inference
    transform = A.Compose([
        A.LongestMaxSize(max_size=800),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(input_tensor)[0]
    
    # Process predictions
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    masks = predictions['masks'].cpu().numpy()
    
    # Filter by confidence
    keep = scores >= conf_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks = masks[keep]
    
    print(f"Detected {len(boxes)} objects:")
    
    # Colors for visualization
    colors = {
        1: (255, 0, 0),    # fiber - red
        2: (0, 255, 255),  # film - yellow
        3: (0, 255, 0),    # fragment - green
    }
    
    # Draw on original image (scaled)
    scale = 800 / max(image.shape[:2])
    vis_image = cv2.resize(image, None, fx=scale, fy=scale)
    overlay = vis_image.copy()
    
    for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        class_name = CLASS_NAMES[label]
        color = colors.get(label, (255, 255, 255))
        
        print(f"  - {class_name}: {score:.2f}")
        
        # Draw bounding box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        text = f"{class_name}: {score:.2f}"
        cv2.putText(vis_image, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw mask
        mask_binary = (mask[0] > mask_threshold).astype(np.uint8)
        mask_resized = cv2.resize(mask_binary, (vis_image.shape[1], vis_image.shape[0]))
        overlay[mask_resized == 1] = color
    
    # Blend mask overlay
    vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
    
    # Save result
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{Path(image_path).stem}_maskrcnn.png"
    cv2.imwrite(str(output_path), vis_image)
    print(f"\nResult saved to: {output_path}")
    
    return predictions


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for microplastic segmentation')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'prepare'], 
                        default='train', help='Mode: train, predict, or prepare')
    parser.add_argument('--annotations', type=str, default='data/annotations/annotations.json',
                        help='Path to COCO annotations')
    parser.add_argument('--image-dir', type=str, default='data/stitched',
                        help='Image directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=800, help='Image size')
    parser.add_argument('--model', type=str, help='Path to model for inference')
    parser.add_argument('--image', type=str, help='Image path for prediction')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(
            annotation_file=args.annotations,
            image_dir=args.image_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            img_size=args.img_size,
            resume=args.resume
        )
    
    elif args.mode == 'predict':
        model_path = args.model or 'experiments/maskrcnn_best.pth'
        if not args.image:
            print("Error: --image required for predict mode")
            return
        predict(model_path, args.image, conf_threshold=args.conf)
    
    elif args.mode == 'prepare':
        print("Data preparation mode - fixing annotation paths...")
        fix_annotation_paths(args.annotations, args.image_dir)


def fix_annotation_paths(annotation_file: str, image_dir: str):
    """Fix image paths in annotation file to point to actual images."""
    with open(annotation_file) as f:
        data = json.load(f)
    
    image_dir = Path(image_dir)
    fixed_count = 0
    
    for img in data['images']:
        old_path = img['file_name']
        base_name = Path(old_path).name
        new_path = str(image_dir / base_name)
        
        if Path(new_path).exists():
            img['file_name'] = base_name  # Store just the filename
            fixed_count += 1
        else:
            print(f"Warning: Image not found: {new_path}")
    
    # Save fixed annotations
    output_path = annotation_file.replace('.json', '_fixed.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed {fixed_count} image paths")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
