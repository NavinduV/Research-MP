"""
Evaluate YOLO model on a split from dataset.yaml.

Computes detailed detection metrics including:
- TP, FP, FN
- Precision, Recall, F1
- Detection Accuracy
- Mean IoU (matched detections)
- AP50, AP75, mAP50, mAP75
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def normalize_class_names(names_field) -> List[str]:
    if isinstance(names_field, list):
        return [str(x) for x in names_field]
    if isinstance(names_field, dict):
        parsed = {}
        for key, value in names_field.items():
            try:
                parsed[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        if parsed:
            max_id = max(parsed)
            return [parsed.get(i, f"class_{i}") for i in range(max_id + 1)]
    return []


def resolve_root_from_yaml(data_yaml: Path, dataset_cfg: Dict) -> Path:
    default_root = data_yaml.parent.resolve()
    root = dataset_cfg.get("path")
    if root is None:
        return default_root

    root_path = Path(str(root).strip())
    if not root_path.is_absolute():
        root_path = (default_root / root_path).resolve()

    # YAML files are often moved across machines. If the configured absolute
    # root no longer exists, fall back to the local YAML directory.
    if root_path.exists():
        return root_path
    return default_root


def resolve_image_entries(root: Path, split_entry, yaml_path: Path) -> List[Path]:
    entries = split_entry if isinstance(split_entry, list) else [split_entry]
    images: List[Path] = []

    for entry in entries:
        entry_path = Path(entry)
        if not entry_path.is_absolute():
            entry_path = (root / entry_path).resolve()

        if entry_path.is_file() and entry_path.suffix.lower() == ".txt":
            with open(entry_path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            for line in lines:
                line_path = Path(line)
                candidates = []
                if line_path.is_absolute():
                    candidates.append(line_path)
                else:
                    candidates.append((root / line_path).resolve())
                    candidates.append((entry_path.parent / line_path).resolve())
                    candidates.append((yaml_path.parent / line_path).resolve())

                for candidate in candidates:
                    if candidate.exists():
                        images.append(candidate)
                        break
        elif entry_path.is_dir():
            for image_path in sorted(entry_path.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(image_path)
        elif entry_path.is_file() and entry_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry_path)

    unique_sorted = sorted({p.resolve() for p in images})
    return unique_sorted


def resolve_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [i for i, part in enumerate(parts) if part.lower() == "images"]
    for idx in reversed(image_indices):
        candidate_parts = parts.copy()
        candidate_parts[idx] = "labels"
        candidate = Path(*candidate_parts).with_suffix(".txt")
        if candidate.exists():
            return candidate

    candidate = image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
    if candidate.exists():
        return candidate

    return candidate


def load_yolo_labels(label_path: Path, image_width: int, image_height: int) -> List[Dict]:
    if not label_path.exists():
        return []

    labels = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cls_id = int(float(parts[0]))
                x_center, y_center, width, height = [float(v) for v in parts[1:5]]
            except ValueError:
                continue

            normalized = max(abs(x_center), abs(y_center), abs(width), abs(height)) <= 1.5
            if normalized:
                x1 = (x_center - width / 2.0) * image_width
                y1 = (y_center - height / 2.0) * image_height
                x2 = (x_center + width / 2.0) * image_width
                y2 = (y_center + height / 2.0) * image_height
            else:
                x1 = x_center - width / 2.0
                y1 = y_center - height / 2.0
                x2 = x_center + width / 2.0
                y2 = y_center + height / 2.0

            x1 = max(0.0, min(float(image_width), x1))
            y1 = max(0.0, min(float(image_height), y1))
            x2 = max(0.0, min(float(image_width), x2))
            y2 = max(0.0, min(float(image_height), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            labels.append(
                {
                    "cls": cls_id,
                    "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                }
            )
    return labels


def match_detections_class_aware(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if not predictions:
        return [], [], list(range(len(ground_truths)))
    if not ground_truths:
        return [], list(range(len(predictions))), []

    sorted_pred_indices = sorted(range(len(predictions)), key=lambda idx: predictions[idx]["conf"], reverse=True)
    matched_gt_indices = set()
    matches: List[Tuple[int, int, float]] = []
    unmatched_pred_indices: List[int] = []

    for pred_idx in sorted_pred_indices:
        pred = predictions[pred_idx]
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt_indices:
                continue
            if int(pred["cls"]) != int(gt["cls"]):
                continue

            iou = box_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched_gt_indices.add(best_gt_idx)
            matches.append((pred_idx, best_gt_idx, best_iou))
        else:
            unmatched_pred_indices.append(pred_idx)

    unmatched_gt_indices = [idx for idx in range(len(ground_truths)) if idx not in matched_gt_indices]
    return matches, unmatched_pred_indices, unmatched_gt_indices


def gather_ap_records(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_id: int,
    iou_threshold: float,
) -> List[Tuple[float, int]]:
    preds_c = [pred for pred in predictions if int(pred["cls"]) == class_id]
    gts_c = [gt for gt in ground_truths if int(gt["cls"]) == class_id]
    preds_c = sorted(preds_c, key=lambda row: row["conf"], reverse=True)

    used_gt = set()
    records: List[Tuple[float, int]] = []

    for pred in preds_c:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gts_c):
            if gt_idx in used_gt:
                continue
            iou = box_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            records.append((float(pred["conf"]), 1))
        else:
            records.append((float(pred["conf"]), 0))

    return records


def compute_ap(records: List[Tuple[float, int]], num_gt: int) -> float:
    if num_gt <= 0:
        return 0.0
    if not records:
        return 0.0

    records_sorted = sorted(records, key=lambda row: row[0], reverse=True)
    tp = np.array([row[1] for row in records_sorted], dtype=np.float64)
    fp = 1.0 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / float(num_gt)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
    return float(ap)


def evaluate(
    model_path: str,
    data_yaml: str,
    split: str = "val",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    nms_iou: float = 0.7,
    imgsz: int = 1280,
    batch_size: int = 8,
    device: Optional[str] = None,
    max_images: Optional[int] = None,
):
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as handle:
        dataset_cfg = yaml.safe_load(handle)

    class_names = normalize_class_names(dataset_cfg.get("names", []))
    root = resolve_root_from_yaml(data_yaml_path, dataset_cfg)
    if split not in dataset_cfg:
        raise ValueError(f"Split '{split}' not found in dataset.yaml")

    image_paths = resolve_image_entries(root, dataset_cfg[split], data_yaml_path)

    # Extra guard: if split paths still resolve to nothing, retry with the
    # dataset.yaml directory as root and continue if that works.
    yaml_local_root = data_yaml_path.parent.resolve()
    if len(image_paths) == 0 and root != yaml_local_root:
        fallback_paths = resolve_image_entries(yaml_local_root, dataset_cfg[split], data_yaml_path)
        if len(fallback_paths) > 0:
            print(
                f"Warning: dataset root '{root}' did not resolve images; "
                f"using local root '{yaml_local_root}' instead."
            )
            root = yaml_local_root
            image_paths = fallback_paths

    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found for split '{split}'")

    print(f"\n{'='*68}")
    print("YOLO EVALUATION")
    print(f"{'='*68}")
    print(f"  Model        : {model_path}")
    print(f"  Dataset yaml : {data_yaml}")
    print(f"  Split        : {split}")
    print(f"  Images       : {len(image_paths)}")
    print(f"  Conf thresh  : {conf_threshold}")
    print(f"  IoU thresh   : {iou_threshold}")
    print(f"  Device       : {device if device else 'auto'}")
    print(f"{'='*68}\n")

    model = YOLO(str(model_path_obj))

    tp_total = 0
    fp_total = 0
    fn_total = 0
    matched_ious: List[float] = []
    confidence_values: List[float] = []

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    per_class_gt_count = defaultdict(int)
    per_class_pred_count = defaultdict(int)

    ap_thresholds = [0.5, 0.75]
    ap_records = {thr: defaultdict(list) for thr in ap_thresholds}

    for start_idx in tqdm(range(0, len(image_paths), batch_size), desc="Evaluating", unit="batch"):
        batch_paths = image_paths[start_idx:start_idx + batch_size]
        batch_results = model.predict(
            source=[str(path) for path in batch_paths],
            conf=conf_threshold,
            iou=nms_iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        for image_path, result in zip(batch_paths, batch_results):
            image_height, image_width = result.orig_shape

            predictions: List[Dict] = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                for box, cls_id, conf in zip(boxes_xyxy, classes, confidences):
                    predictions.append(
                        {
                            "cls": int(cls_id),
                            "conf": float(conf),
                            "box": box.astype(np.float32),
                        }
                    )
                    confidence_values.append(float(conf))
                    per_class_pred_count[int(cls_id)] += 1

            label_path = resolve_label_path(image_path)
            ground_truths = load_yolo_labels(label_path, image_width=image_width, image_height=image_height)
            for gt in ground_truths:
                per_class_gt_count[int(gt["cls"])] += 1

            matches, unmatched_pred_indices, unmatched_gt_indices = match_detections_class_aware(
                predictions=predictions,
                ground_truths=ground_truths,
                iou_threshold=iou_threshold,
            )

            tp_total += len(matches)
            fp_total += len(unmatched_pred_indices)
            fn_total += len(unmatched_gt_indices)

            for pred_idx, gt_idx, matched_iou in matches:
                cls_id = int(ground_truths[gt_idx]["cls"])
                per_class_tp[cls_id] += 1
                matched_ious.append(float(matched_iou))

            for pred_idx in unmatched_pred_indices:
                cls_id = int(predictions[pred_idx]["cls"])
                per_class_fp[cls_id] += 1

            for gt_idx in unmatched_gt_indices:
                cls_id = int(ground_truths[gt_idx]["cls"])
                per_class_fn[cls_id] += 1

            class_ids_for_ap = set([int(gt["cls"]) for gt in ground_truths] + [int(pred["cls"]) for pred in predictions])
            for class_id in class_ids_for_ap:
                for thr in ap_thresholds:
                    records = gather_ap_records(
                        predictions=predictions,
                        ground_truths=ground_truths,
                        class_id=class_id,
                        iou_threshold=thr,
                    )
                    ap_records[thr][class_id].extend(records)

    precision = safe_div(tp_total, tp_total + fp_total)
    recall = safe_div(tp_total, tp_total + fn_total)
    f1_score = safe_div(2.0 * precision * recall, precision + recall)
    accuracy = safe_div(tp_total, tp_total + fp_total + fn_total)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    median_iou = float(np.median(matched_ious)) if matched_ious else 0.0

    class_ids = sorted(
        set(per_class_gt_count.keys())
        | set(per_class_pred_count.keys())
        | set(per_class_tp.keys())
        | set(per_class_fp.keys())
        | set(per_class_fn.keys())
    )

    def class_name_for(class_id: int) -> str:
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return f"class_{class_id}"

    ap_summary = {}
    map_summary = {}
    for thr in ap_thresholds:
        per_class_ap = {}
        valid_ap_values = []
        for class_id in class_ids:
            num_gt = per_class_gt_count[class_id]
            class_ap = compute_ap(ap_records[thr][class_id], num_gt=num_gt)
            per_class_ap[class_name_for(class_id)] = class_ap
            if num_gt > 0:
                valid_ap_values.append(class_ap)
        ap_summary[f"ap{int(thr * 100)}_per_class"] = per_class_ap
        map_summary[f"mAP{int(thr * 100)}"] = float(np.mean(valid_ap_values)) if valid_ap_values else 0.0

    per_class_metrics = []
    for class_id in class_ids:
        tp_c = per_class_tp[class_id]
        fp_c = per_class_fp[class_id]
        fn_c = per_class_fn[class_id]
        prec_c = safe_div(tp_c, tp_c + fp_c)
        rec_c = safe_div(tp_c, tp_c + fn_c)
        f1_c = safe_div(2.0 * prec_c * rec_c, prec_c + rec_c)

        per_class_metrics.append(
            {
                "class_id": class_id,
                "class_name": class_name_for(class_id),
                "gt_count": int(per_class_gt_count[class_id]),
                "pred_count": int(per_class_pred_count[class_id]),
                "tp": int(tp_c),
                "fp": int(fp_c),
                "fn": int(fn_c),
                "precision": float(prec_c),
                "recall": float(rec_c),
                "f1_score": float(f1_c),
                "ap50": float(ap_summary["ap50_per_class"].get(class_name_for(class_id), 0.0)),
                "ap75": float(ap_summary["ap75_per_class"].get(class_name_for(class_id), 0.0)),
            }
        )

    metrics = {
        "num_images": int(len(image_paths)),
        "num_classes": int(len(class_ids)),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "accuracy": float(accuracy),
        "mean_iou": float(mean_iou),
        "median_iou": float(median_iou),
        "avg_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
        "ap50": float(map_summary["mAP50"]),
        "ap75": float(map_summary["mAP75"]),
        "mAP50": float(map_summary["mAP50"]),
        "mAP75": float(map_summary["mAP75"]),
        "per_class": per_class_metrics,
        "per_class_ap50": ap_summary["ap50_per_class"],
        "per_class_ap75": ap_summary["ap75_per_class"],
    }

    print(f"\n{'='*68}")
    print("YOLO EVALUATION RESULTS")
    print(f"{'='*68}")
    print(f"  TP/FP/FN      : {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1 Score      : {metrics['f1_score']:.4f}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Mean IoU      : {metrics['mean_iou']:.4f}")
    print(f"  Median IoU    : {metrics['median_iou']:.4f}")
    print(f"  AP50 / mAP50  : {metrics['ap50']:.4f}")
    print(f"  AP75 / mAP75  : {metrics['ap75']:.4f}")
    print(f"{'='*68}")
    print("Per-class:")
    for row in metrics["per_class"]:
        print(
            f"  {row['class_name']:<10} "
            f"GT={row['gt_count']:<4d} Pred={row['pred_count']:<4d} "
            f"TP={row['tp']:<4d} FP={row['fp']:<4d} FN={row['fn']:<4d} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1_score']:.4f} "
            f"AP50={row['ap50']:.4f}"
        )
    print(f"{'='*68}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model with detailed detection metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--split", type=str, default="val", help="Dataset split key in dataset.yaml (default: val)")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for TP/FP matching")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU used during model prediction")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--device", type=str, default=None, help="Computation device (cpu, cuda, cuda:0)")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick evaluation")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    metrics = evaluate(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        nms_iou=args.nms_iou,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        max_images=args.max_images,
    )

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Saved metrics JSON: {save_path}")


if __name__ == "__main__":
    main()
