"""
Train EfficientNet-B0 for Microplastic Classification.

EfficientNet classifies cropped microplastic detections into:
    0: fiber
    1: film
    2: fragment

This is used as a dedicated classifier in the pipeline:
    YOLO (detect) → Crop → EfficientNet (classify) + Mask R-CNN (segment)

Usage:
    # Train on crops from YOLO ground-truth labels (with SAM masks)
    python src/train_effnet.py --mode train --crops-dir data/crops_gt_sam --epochs 50

    # Train on crops from YOLO ground-truth (class folders only)
    python src/train_effnet.py --mode train --crops-dir data/crops_gt --epochs 50

    # Evaluate on validation set
    python src/train_effnet.py --mode val --crops-dir data/crops_gt_sam

    # Classify a single crop image
    python src/train_effnet.py --mode predict --image path/to/crop.png
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import timm


# ============================================================================
# Configuration
# ============================================================================

NUM_CLASSES = 3  # fiber, film, fragment
CLASS_NAMES = ['fiber', 'film', 'fragment']
CLASS_NAME_TO_ID = {'fiber': 0, 'film': 1, 'fragment': 2}
IMG_SIZE = 224


# ============================================================================
# Dataset
# ============================================================================

class CropClassificationDataset(Dataset):
    """
    Dataset for microplastic crop classification.
    
    Supports two layouts:
      1. annotations.json + images/ directory
      2. Class subdirectories: crops_dir/{fiber,film,fragment}/*.png
    """
    
    def __init__(self, crops_dir: str, transform=None, split: str = None):
        self.crops_dir = Path(crops_dir)
        self.transform = transform
        self.samples = []  # list of (image_path, class_id)
        
        ann_file = self.crops_dir / 'annotations.json'
        
        if ann_file.exists():
            with open(ann_file) as f:
                annotations = json.load(f)
            
            for filename, ann in annotations.items():
                if split and ann.get('split', '') != split:
                    continue
                
                cls_id = ann['class_id']
                
                img_path = self.crops_dir / 'images' / filename
                if not img_path.exists():
                    cls_name = ann.get('class_name', CLASS_NAMES[cls_id])
                    img_path = self.crops_dir / cls_name / filename
                
                if img_path.exists():
                    self.samples.append((str(img_path), cls_id))
        else:
            for cls_name, cls_id in CLASS_NAME_TO_ID.items():
                cls_dir = self.crops_dir / cls_name
                if not cls_dir.is_dir():
                    continue
                for img_file in sorted(cls_dir.glob('*.png')):
                    self.samples.append((str(img_file), cls_id))
        
        # Count per class
        class_counts = [0] * NUM_CLASSES
        for _, cls_id in self.samples:
            class_counts[cls_id] += 1
        
        split_label = f" ({split})" if split else ""
        print(f"Loaded {len(self.samples)} samples{split_label} from {crops_dir}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {class_counts[i]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_class_weights(self):
        """Compute inverse-frequency class weights for imbalanced data."""
        counts = np.zeros(NUM_CLASSES)
        for _, cls_id in self.samples:
            counts[cls_id] += 1
        weights = 1.0 / np.maximum(counts, 1)
        weights = weights / weights.sum() * NUM_CLASSES
        return torch.FloatTensor(weights)


# ============================================================================
# Transforms
# ============================================================================

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ============================================================================
# Model
# ============================================================================

def get_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Create EfficientNet-B0 classifier."""
    model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
    return model


# ============================================================================
# Training
# ============================================================================

def train(crops_dir: str, epochs: int = 50, batch_size: int = 16, lr: float = 0.0001,
          save_dir: str = 'experiments', resume: str = None, val_split: float = 0.2):
    """Train EfficientNet-B0 on microplastic crops."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print("EFFICIENTNET-B0 TRAINING — MICROPLASTIC CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Device       : {device}")
    print(f"Crops dir    : {crops_dir}")
    print(f"Epochs       : {epochs}")
    print(f"Batch size   : {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}\n")
    
    # Check if annotations have split field
    ann_file = Path(crops_dir) / 'annotations.json'
    has_splits = False
    if ann_file.exists():
        with open(ann_file) as f:
            anns = json.load(f)
        has_splits = any('split' in v for v in anns.values())
    
    if has_splits:
        train_dataset = CropClassificationDataset(
            crops_dir, transform=get_transforms(train=True), split='train')
        val_dataset = CropClassificationDataset(
            crops_dir, transform=get_transforms(train=False), split='val')
    else:
        full_dataset = CropClassificationDataset(
            crops_dir, transform=get_transforms(train=True))
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        print(f"\nRandom split: {train_size} train, {val_size} val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = get_model(NUM_CLASSES, pretrained=True)
    model.to(device)
    
    # Class weights for imbalanced data
    if has_splits:
        class_weights = train_dataset.get_class_weights().to(device)
    else:
        counts = np.zeros(NUM_CLASSES)
        for _, cls_id in CropClassificationDataset(crops_dir).samples:
            counts[cls_id] += 1
        weights = 1.0 / np.maximum(counts, 1)
        weights = weights / weights.sum() * NUM_CLASSES
        class_weights = torch.FloatTensor(weights).to(device)
    
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Resume
    start_epoch = 0
    if resume and Path(resume).exists():
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Training loop
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images)
            loss = loss_fn(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (preds.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.2%}"
            })
        
        scheduler.step()
        
        train_acc = train_correct / max(train_total, 1)
        avg_train_loss = train_loss / max(len(train_loader), 1)
        
        # --- Validate ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        class_correct = np.zeros(NUM_CLASSES)
        class_total = np.zeros(NUM_CLASSES)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                loss = loss_fn(preds, labels)
                
                val_loss += loss.item()
                predicted = preds.argmax(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                for i in range(labels.size(0)):
                    lbl = labels[i].item()
                    class_total[lbl] += 1
                    if predicted[i] == lbl:
                        class_correct[lbl] += 1
        
        val_acc = val_correct / max(val_total, 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        
        print(f"Epoch {epoch+1}/{epochs} — "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")
        
        for i, name in enumerate(CLASS_NAMES):
            if class_total[i] > 0:
                print(f"  {name}: {class_correct[i]:.0f}/{class_total[i]:.0f} = {class_correct[i]/class_total[i]:.2%}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }
        torch.save(checkpoint, f"{save_dir}/efficientnet_latest.pth")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, f"{save_dir}/efficientnet_best.pth")
            print(f"  → New best model! Val Acc: {val_acc:.2%}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best Val Accuracy: {best_val_acc:.2%}")
    print(f"Best model: {save_dir}/efficientnet_best.pth")
    print(f"{'='*60}\n")


# ============================================================================
# Validation
# ============================================================================

def validate(crops_dir: str, model_path: str = 'experiments/efficientnet_best.pth'):
    """Evaluate EfficientNet on validation data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    dataset = CropClassificationDataset(
        crops_dir, transform=get_transforms(train=False), split='val')
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    correct = 0
    total = 0
    class_correct = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            predicted = preds.argmax(1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            for i in range(labels.size(0)):
                lbl = labels[i].item()
                pred = predicted[i].item()
                class_total[lbl] += 1
                confusion[lbl][pred] += 1
                if pred == lbl:
                    class_correct[lbl] += 1
    
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {correct}/{total} = {correct/max(total,1):.2%}")
    print(f"\nPer-class:")
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            print(f"  {name}: {class_correct[i]:.0f}/{class_total[i]:.0f} = {class_correct[i]/class_total[i]:.2%}")
    
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>12} {'fiber':>8} {'film':>8} {'fragment':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>12} {confusion[i][0]:>8} {confusion[i][1]:>8} {confusion[i][2]:>8}")
    print(f"{'='*60}\n")


# ============================================================================
# Prediction
# ============================================================================

def predict(image_path: str, model_path: str = 'experiments/efficientnet_best.pth'):
    """Classify a single crop image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
    
    print(f"\nPrediction for: {image_path}")
    print(f"  Class: {CLASS_NAMES[pred_class]}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[i]:.4f}")
    
    return pred_class, probs.cpu().numpy()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet for microplastic classification')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'predict'], required=True,
                        help='Mode: train, val, or predict')
    parser.add_argument('--crops-dir', type=str, default='data/crops_gt_sam',
                        help='Directory with crop images')
    parser.add_argument('--model', type=str, default='experiments/efficientnet_best.pth',
                        help='Model path (for val/predict)')
    parser.add_argument('--image', type=str, help='Image path (for predict)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(
            crops_dir=args.crops_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            resume=args.resume,
        )
    elif args.mode == 'val':
        validate(crops_dir=args.crops_dir, model_path=args.model)
    elif args.mode == 'predict':
        if not args.image:
            print("Error: --image required for predict mode")
            return
        predict(image_path=args.image, model_path=args.model)


if __name__ == "__main__":
    main()
