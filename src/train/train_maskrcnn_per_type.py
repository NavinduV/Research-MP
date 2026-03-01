"""
Train Specialised Mask R-CNN Models Per Microplastic Type.

Trains 3 independent binary Mask R-CNN models — one for each microplastic type
(fiber, film, fragment).  Each model learns to segment only ONE type, using
SAM-generated masks created by sam_annotate_per_type.py.

Expected input directories (created by sam_annotate_per_type.py):
    data/crops_fiber_sam/      images/ masks/ annotations.json
    data/crops_film_sam/       images/ masks/ annotations.json
    data/crops_fragment_sam/   images/ masks/ annotations.json

Trained weights are saved to:
    experiments/maskrcnn_fiber/    maskrcnn_best.pth  maskrcnn_latest.pth
    experiments/maskrcnn_film/     ...
    experiments/maskrcnn_fragment/ ...

Usage:
    # Train all three models (default 50 epochs each)
    python src/train/train_maskrcnn_per_type.py

    # Train only the fiber model for 100 epochs
    python src/train/train_maskrcnn_per_type.py --types fiber --epochs 100

    # Resume training from checkpoint
    python src/train/train_maskrcnn_per_type.py --types fragment --resume experiments/maskrcnn_fragment/maskrcnn_latest.pth

    # Predict / test a single crop
    python src/train/train_maskrcnn_per_type.py --mode predict --types fiber --image path/to/crop.png
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
from typing import Optional, List

# ============================================================================
# Configuration
# ============================================================================

TYPES = ['fiber', 'film', 'fragment']

# Each per-type model is BINARY: background (0) + the target class (1)
NUM_CLASSES = 2

CROP_SIZE = 128
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 4  # 4 is safer on CPU; increase to 8+ with a GPU
DEFAULT_LR = 0.001


# ============================================================================
# Dataset — single-type crops with SAM masks
# ============================================================================

class SingleTypeCropDataset(Dataset):
    """Dataset for ONE microplastic type with SAM masks.

    Directory layout:
        crops_dir/
            images/       *.png
            masks/        *_mask.png
            annotations.json
    """

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

        # Filter to samples that actually have images
        self.samples = [
            name for name in self.annotations
            if (self.images_dir / name).exists()
        ]

        # Balanced mode: randomly subsample to max_samples
        if max_samples is not None and max_samples < len(self.samples):
            import random
            random.seed(42)
            self.samples = sorted(random.sample(self.samples, max_samples))
            print(f"[{Path(crops_dir).name}] Balanced to {len(self.samples)} samples "
                  f"(from {len(self.annotations)} annotations)")
        else:
            print(f"[{Path(crops_dir).name}] Loaded {len(self.samples)} samples "
                  f"({len(self.annotations)} annotations)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        ann = self.annotations[sample_name]

        # Load image
        img_path = self.images_dir / sample_name
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Load mask (SAM mask preferred, else fallback to ellipse)
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
            # Ellipse fallback
            mask = self._create_ellipse_mask(h, w, ann.get('rel_box'))

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Bounding box from mask
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            box = [xs.min(), ys.min(), xs.max(), ys.max()]
        else:
            margin = min(h, w) // 10
            box = [margin, margin, w - margin, h - margin]

        # Binary label: 1 = the target type (only class in this model)
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

        return image, target

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


# ============================================================================
# Transforms
# ============================================================================

def get_transforms(train: bool = True, img_size: int = CROP_SIZE):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.03, 0.15), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================================
# Model
# ============================================================================

def get_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Create a Mask R-CNN model with the given number of output classes."""
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
# Training
# ============================================================================

def train_single_type(
    mp_type: str,
    crops_dir: str = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    save_dir: str = None,
    resume: str = None,
    device_str: str = None,
    max_samples: int = None,
):
    """Train a binary Mask R-CNN for one microplastic type."""

    if crops_dir is None:
        crops_dir = f'data/crops_{mp_type}_sam'
    if save_dir is None:
        save_dir = f'experiments/maskrcnn_{mp_type}'

    device = torch.device(device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Use all CPU cores when no GPU is available
    if device.type == 'cpu':
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        print(f"WARNING: No GPU found — training on CPU with {num_threads} threads.")
        print("  This will be slow. Consider using a machine with a CUDA GPU.")

    print(f"\n{'='*60}")
    print(f"TRAINING MASK R-CNN — {mp_type.upper()}")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Crops dir   : {crops_dir}")
    print(f"  Save dir    : {save_dir}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  LR          : {lr}")
    print(f"  Classes     : 2 (background + {mp_type})")
    print(f"{'='*60}\n")

    if not Path(crops_dir).exists():
        print(f"ERROR: Data directory not found: {crops_dir}")
        print(f"  Run SAM annotation first:")
        print(f"  python src/data_preparation/sam_annotate_per_type.py --types {mp_type}")
        return

    # Dataset & dataloader
    dataset = SingleTypeCropDataset(
        crops_dir=crops_dir,
        transforms=get_transforms(train=True),
        max_samples=max_samples,
    )

    if len(dataset) == 0:
        print(f"ERROR: No samples found in {crops_dir}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Model
    model = get_model(NUM_CLASSES, pretrained=True)
    model.to(device)

    # Resume
    start_epoch = 0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    # Optimiser
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"[{mp_type}] Epoch {epoch + 1}/{epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if not all(len(t['boxes']) > 0 for t in targets):
                continue

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})

        lr_scheduler.step()
        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"  [{mp_type}] Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}")

        ckpt = {
            'epoch': epoch + 1,
            'mp_type': mp_type,
            'num_classes': NUM_CLASSES,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(ckpt, f'{save_dir}/maskrcnn_latest.pth')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, f'{save_dir}/maskrcnn_best.pth')
            print(f"  -> New best model saved! (loss={avg_loss:.4f})")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {mp_type.upper()}")
    print(f"  Best model : {save_dir}/maskrcnn_best.pth")
    print(f"{'='*60}\n")

    return f'{save_dir}/maskrcnn_best.pth'


def _count_available_samples(mp_type: str) -> int:
    """Count how many valid images exist in a type's SAM directory."""
    d = Path(f'data/crops_{mp_type}_sam')
    if not d.exists():
        return 0
    imgs = d / 'images'
    if not imgs.exists():
        return 0
    return len(list(imgs.glob('*.png')) + list(imgs.glob('*.jpg')))


def train_all(
    types: List[str],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    resume: str = None,
    device_str: str = None,
    balanced: bool = False,
):
    """Train a specialised Mask R-CNN for each requested type sequentially.

    By default each model trains on ALL available samples for its type.
    Set balanced=True to downsample all types to the smallest type's count.
    """
    # Determine balanced sample count
    max_samples = None
    if balanced and len(types) > 1:
        counts = {t: _count_available_samples(t) for t in types}
        print(f"\nPer-type sample counts: {counts}")
        min_count = min(c for c in counts.values() if c > 0) if any(counts.values()) else 0
        if min_count > 0:
            max_samples = min_count
            print(f"Balanced training: limiting each type to {max_samples} samples\n")

    results = {}
    for mp_type in types:
        best_path = train_single_type(
            mp_type=mp_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            resume=resume,
            device_str=device_str,
            max_samples=max_samples,
        )
        results[mp_type] = best_path

    print(f"\n{'='*60}")
    print("ALL PER-TYPE MODELS TRAINED")
    print(f"{'='*60}")
    for t, p in results.items():
        print(f"  {t:>10}: {p}")
    print(f"{'='*60}\n")

    return results


# ============================================================================
# Prediction / Visualisation
# ============================================================================

def predict_single(
    mp_type: str,
    image_path: str,
    model_path: str = None,
    device_str: str = None,
    save_path: str = None,
):
    """Run a per-type Mask R-CNN on a single crop and visualise the result."""
    import torchvision.transforms.functional as F

    if model_path is None:
        model_path = f'experiments/maskrcnn_{mp_type}/maskrcnn_best.pth'

    device = torch.device(device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu'))

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    # Load model
    ckpt = torch.load(model_path, map_location=device)
    num_classes = ckpt.get('num_classes', NUM_CLASSES)
    model = get_model(num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize + normalise
    resized = cv2.resize(image_rgb, (CROP_SIZE, CROP_SIZE))
    tensor = F.to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)[0]

    # Draw results
    vis = resized.copy()
    masks = outputs.get('masks', torch.zeros(0))
    scores = outputs.get('scores', torch.zeros(0))

    if len(masks) > 0:
        # Best prediction
        best_idx = scores.argmax()
        best_mask = (masks[best_idx, 0].cpu().numpy() > 0.5).astype(np.uint8)
        best_score = scores[best_idx].item()

        overlay = vis.copy()
        overlay[best_mask == 1] = (overlay[best_mask == 1] * 0.5 +
                                    np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        h, w = vis.shape[:2]
        panel = np.zeros((h, w * 3, 3), dtype=np.uint8)
        panel[:, :w] = vis
        panel[:, w:w*2] = cv2.applyColorMap((best_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        panel[:, w*2:] = overlay

        cv2.putText(panel, f'{mp_type} ({best_score:.2f})', (10, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            print(f"Saved visualisation: {save_path}")
        else:
            cv2.imshow(f'Prediction - {mp_type}', cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"No detections for {image_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Specialised Mask R-CNN Per Microplastic Type')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                        help='train: train models; predict: test on single image')
    parser.add_argument('--types', nargs='+', default=TYPES, choices=TYPES,
                        help='Which types to train (default: all)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Training epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH, help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda or cpu)')
    parser.add_argument('--image', type=str, default=None, help='Image for predict mode')
    parser.add_argument('--model', type=str, default=None, help='Model path for predict mode')
    parser.add_argument('--save-vis', type=str, default=None, help='Save prediction visualisation')
    parser.add_argument('--balanced', action='store_true', default=False,
                        help='Downsample all types to the smallest type count')

    args = parser.parse_args()

    if args.mode == 'train':
        train_all(
            types=args.types,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            resume=args.resume,
            device_str=args.device,
            balanced=args.balanced,
        )

    elif args.mode == 'predict':
        if not args.image:
            print("ERROR: --image is required for predict mode")
            return
        if len(args.types) != 1:
            print("ERROR: specify exactly one --types for predict mode")
            return

        predict_single(
            mp_type=args.types[0],
            image_path=args.image,
            model_path=args.model,
            device_str=args.device,
            save_path=args.save_vis,
        )


if __name__ == '__main__':
    main()
