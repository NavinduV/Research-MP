"""
Evaluate EfficientNet model on microplastic crop classification dataset.

Computes detailed classification metrics including:
- TP, FP, FN (overall and per class)
- Precision, Recall, F1 (per class + macro/micro/weighted)
- Accuracy, Top-2, Top-3 accuracy
- Confusion matrix
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from src.train.train_effnet import (
        CLASS_NAMES,
        CropClassificationDataset,
        NUM_CLASSES,
        get_transforms,
    )
except ImportError:
    from train_effnet import (  # type: ignore
        CLASS_NAMES,
        CropClassificationDataset,
        NUM_CLASSES,
        get_transforms,
    )


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        # Some checkpoints are raw state_dicts saved directly.
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("Unsupported checkpoint format. Expected model_state_dict or raw state_dict.")


def _infer_efficientnet_name(
    state_dict: Dict[str, torch.Tensor],
    checkpoint,
    model_name_override: Optional[str] = None,
) -> str:
    if model_name_override:
        return model_name_override

    if isinstance(checkpoint, dict):
        for key in ("model_name", "arch", "backbone"):
            value = checkpoint.get(key)
            if isinstance(value, str) and value.startswith("efficientnet_b"):
                return value

    stem = state_dict.get("conv_stem.weight")
    head = state_dict.get("conv_head.weight")
    cls = state_dict.get("classifier.weight")

    stem_out = int(stem.shape[0]) if stem is not None else None
    head_out = int(head.shape[0]) if head is not None else None
    cls_in = int(cls.shape[1]) if cls is not None and cls.ndim == 2 else None

    # Match by known channel signatures.
    signature_to_name = {
        (32, 1280): "efficientnet_b0",  # b0/b1 share this; default to b0
        (32, 1408): "efficientnet_b2",
        (40, 1536): "efficientnet_b3",
        (48, 1792): "efficientnet_b4",
        (48, 2048): "efficientnet_b5",
        (56, 2304): "efficientnet_b6",
        (64, 2560): "efficientnet_b7",
    }

    if stem_out is not None and head_out is not None:
        name = signature_to_name.get((stem_out, head_out))
        if name:
            return name

    # Secondary match on classifier input width.
    cls_to_name = {
        1280: "efficientnet_b0",
        1408: "efficientnet_b2",
        1536: "efficientnet_b3",
        1792: "efficientnet_b4",
        2048: "efficientnet_b5",
        2304: "efficientnet_b6",
        2560: "efficientnet_b7",
    }
    if cls_in is not None and cls_in in cls_to_name:
        return cls_to_name[cls_in]

    return "efficientnet_b0"


def compute_metrics_from_confusion(confusion: np.ndarray, class_names: List[str]) -> Dict:
    total = int(confusion.sum())
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    tn = total - (tp + fp + fn)

    precision = np.array([safe_div(tp[i], tp[i] + fp[i]) for i in range(len(class_names))], dtype=np.float64)
    recall = np.array([safe_div(tp[i], tp[i] + fn[i]) for i in range(len(class_names))], dtype=np.float64)
    f1 = np.array(
        [safe_div(2.0 * precision[i] * recall[i], precision[i] + recall[i]) for i in range(len(class_names))],
        dtype=np.float64,
    )
    support = confusion.sum(axis=1).astype(np.float64)

    accuracy = safe_div(float(tp.sum()), float(total))

    valid = support > 0
    macro_precision = float(np.mean(precision[valid])) if np.any(valid) else 0.0
    macro_recall = float(np.mean(recall[valid])) if np.any(valid) else 0.0
    macro_f1 = float(np.mean(f1[valid])) if np.any(valid) else 0.0

    weighted_precision = float(np.sum(precision * support) / support.sum()) if support.sum() > 0 else 0.0
    weighted_recall = float(np.sum(recall * support) / support.sum()) if support.sum() > 0 else 0.0
    weighted_f1 = float(np.sum(f1 * support) / support.sum()) if support.sum() > 0 else 0.0

    micro_tp = float(tp.sum())
    micro_fp = float(fp.sum())
    micro_fn = float(fn.sum())
    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(2.0 * micro_precision * micro_recall, micro_precision + micro_recall)

    per_class = []
    for i, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_id": int(i),
                "class_name": class_name,
                "support": int(support[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "tn": int(tn[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
            }
        )

    return {
        "accuracy": float(accuracy),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(
    model_path: str,
    crops_dir: str,
    split: str = "val",
    batch_size: int = 16,
    device: Optional[str] = None,
    num_workers: int = 0,
    model_name: Optional[str] = None,
):
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    requested_split = None if split == "all" else split
    dataset = CropClassificationDataset(
        crops_dir=crops_dir,
        transform=get_transforms(train=False),
        split=requested_split,
    )

    if len(dataset) == 0 and split != "all":
        print(f"Warning: split '{split}' is empty. Falling back to all samples.")
        dataset = CropClassificationDataset(
            crops_dir=crops_dir,
            transform=get_transforms(train=False),
            split=None,
        )

    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in crops directory: {crops_dir}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    checkpoint = torch.load(model_path_obj, map_location=device_obj)

    state_dict = _extract_state_dict(checkpoint)
    inferred_model_name = _infer_efficientnet_name(
        state_dict=state_dict,
        checkpoint=checkpoint,
        model_name_override=model_name,
    )

    # Infer class count from checkpoint classifier layer when available.
    ckpt_num_classes = NUM_CLASSES
    cls_weight = state_dict.get("classifier.weight")
    if cls_weight is not None and cls_weight.ndim == 2:
        ckpt_num_classes = int(cls_weight.shape[0])

    model = timm.create_model(inferred_model_name, pretrained=False, num_classes=ckpt_num_classes)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load EfficientNet checkpoint. "
            "Try passing --model-name explicitly (for example efficientnet_b3).\n"
            f"Original error:\n{exc}"
        ) from exc

    model.to(device_obj)
    model.eval()

    class_names = list(CLASS_NAMES)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0

    print(f"\n{'='*68}")
    print("EFFICIENTNET EVALUATION")
    print(f"{'='*68}")
    print(f"  Device       : {device_obj}")
    print(f"  Model        : {model_path}")
    print(f"  Backbone     : {inferred_model_name}")
    print(f"  Crops dir    : {crops_dir}")
    print(f"  Split        : {split}")
    print(f"  Batch size   : {batch_size}")
    print(f"{'='*68}\n")

    for images, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device_obj)
        labels = labels.to(device_obj)

        logits = model(images)
        loss = loss_fn(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        k2 = min(2, probs.shape[1])
        k3 = min(3, probs.shape[1])
        top2 = torch.topk(probs, k=k2, dim=1).indices
        top3 = torch.topk(probs, k=k3, dim=1).indices

        batch_size_now = labels.size(0)
        total_samples += batch_size_now
        total_loss += float(loss.item()) * batch_size_now
        top1_correct += int((preds == labels).sum().item())

        labels_expanded = labels.unsqueeze(1)
        top2_correct += int((top2 == labels_expanded).any(dim=1).sum().item())
        top3_correct += int((top3 == labels_expanded).any(dim=1).sum().item())

        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        for true_label, pred_label in zip(labels_np, preds_np):
            confusion[int(true_label), int(pred_label)] += 1

    metrics = compute_metrics_from_confusion(confusion, class_names)
    metrics.update(
        {
            "num_samples": int(total_samples),
            "avg_loss": float(total_loss / max(total_samples, 1)),
            "top1_accuracy": float(safe_div(top1_correct, total_samples)),
            "top2_accuracy": float(safe_div(top2_correct, total_samples)),
            "top3_accuracy": float(safe_div(top3_correct, total_samples)),
            "tp": int(np.trace(confusion)),
            "fp": int(np.sum(confusion, axis=0).sum() - np.trace(confusion)),
            "fn": int(np.sum(confusion, axis=1).sum() - np.trace(confusion)),
            "precision": float(metrics["micro_precision"]),
            "recall": float(metrics["micro_recall"]),
            "f1_score": float(metrics["micro_f1"]),
            "confusion_matrix": confusion.tolist(),
            "class_names": class_names,
            "mean_iou": None,
            "note": "IoU is not applicable for pure image classification models.",
        }
    )

    print(f"\n{'='*68}")
    print("EFFICIENTNET EVALUATION RESULTS")
    print(f"{'='*68}")
    print(f"  Samples       : {metrics['num_samples']}")
    print(f"  Avg Loss      : {metrics['avg_loss']:.4f}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Top-2 Acc     : {metrics['top2_accuracy']:.4f}")
    print(f"  Top-3 Acc     : {metrics['top3_accuracy']:.4f}")
    print(f"  TP/FP/FN      : {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1 Score      : {metrics['f1_score']:.4f}")
    print(f"  Macro P/R/F1  : {metrics['macro_precision']:.4f} / {metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    print(f"{'='*68}")
    print("Per-class:")
    for row in metrics["per_class"]:
        print(
            f"  {row['class_name']:<10} "
            f"Support={row['support']:<4d} TP={row['tp']:<4d} FP={row['fp']:<4d} FN={row['fn']:<4d} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1_score']:.4f}"
        )
    print("Confusion Matrix (rows=true, cols=pred):")
    header = " " * 12 + " ".join([f"{name:>10}" for name in class_names])
    print(header)
    for row_idx, name in enumerate(class_names):
        row_values = " ".join([f"{int(v):>10}" for v in confusion[row_idx]])
        print(f"{name:>12} {row_values}")
    print(f"{'='*68}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet with detailed classification metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to EfficientNet checkpoint")
    parser.add_argument("--crops-dir", type=str, required=True, help="Path to crop dataset directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Computation device (cpu, cuda, cuda:0)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional backbone override (e.g., efficientnet_b3). Default: auto-detect from checkpoint.",
    )
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    metrics = evaluate(
        model_path=args.model,
        crops_dir=args.crops_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        model_name=args.model_name,
    )

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Saved metrics JSON: {save_path}")


if __name__ == "__main__":
    main()
