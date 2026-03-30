"""
YOLO + EfficientNet only demo.

Given an input image, this script:
1. Runs YOLO detection
2. Applies class-agnostic NMS for one box per physical MP
3. Crops each detection and classifies with EfficientNet
4. Saves annotated image and JSON outputs

It intentionally does not run Mask R-CNN.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


MP_NAMES = ["fiber", "film", "fragment"]


def class_agnostic_nms(boxes: List[List[float]], iou_threshold: float = 0.3) -> List[List[float]]:
    """Class-agnostic NMS over boxes in [x1, y1, x2, y2, cls_id, conf] format."""
    if not boxes:
        return []

    arr = np.array(boxes, dtype=np.float64)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]
    scores = arr[:, 5]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        min_area = np.minimum(areas[i], areas[order[1:]])
        containment = inter / np.maximum(min_area, 1e-6)

        inds = np.where((iou <= iou_threshold) & (containment <= 0.6))[0]
        order = order[inds + 1]

    return arr[keep].tolist()


def get_model_class_name(names, class_id: int) -> str:
    """Resolve class name from YOLO result.names (dict or list)."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def load_effnet_checkpoint(model_path: str, device: torch.device):
    """
    Load EfficientNet checkpoint with B3->B0 fallback.

    This supports checkpoints saved with key 'model_state_dict' and plain state dicts.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    for variant in ("efficientnet_b3", "efficientnet_b0"):
        model = timm.create_model(variant, pretrained=False, num_classes=3)
        try:
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"Loaded EfficientNet ({variant}): {model_path}")
            return model
        except Exception:
            continue

    raise RuntimeError("Could not load EfficientNet checkpoint into b3 or b0 architecture.")


def get_effnet_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def classify_crop(effnet_model, crop_bgr: np.ndarray, device: torch.device, transform):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = effnet_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    return pred, MP_NAMES[pred], float(probs[pred].item()), probs.cpu().numpy().tolist()


def run_yolo_effnet_only(
    image_path: str,
    yolo_model_path: str,
    effnet_model_path: str,
    output_dir: str,
    conf_threshold: float,
    crop_padding: int,
) -> Dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    h, w = image.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(yolo_model_path)
    effnet = load_effnet_checkpoint(effnet_model_path, device)
    effnet_tf = get_effnet_transform()

    result = yolo(image_path, conf=conf_threshold, verbose=False)[0]
    raw_boxes: List[List[float]] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        raw_boxes.append([x1, y1, x2, y2, cls_id, conf])

    final_boxes = class_agnostic_nms(raw_boxes, iou_threshold=0.3)

    vis = image.copy()
    class_counts = {name: 0 for name in MP_NAMES}
    detections = []

    for idx, bx in enumerate(final_boxes, start=1):
        x1, y1, x2, y2 = map(int, bx[:4])
        yolo_cls = int(bx[4])
        yolo_conf = float(bx[5])

        x1c = max(0, x1 - crop_padding)
        y1c = max(0, y1 - crop_padding)
        x2c = min(w, x2 + crop_padding)
        y2c = min(h, y2 + crop_padding)
        crop = image[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue

        yolo_name = get_model_class_name(result.names, yolo_cls)
        eff_cls_id, eff_cls_name, eff_conf, eff_probs = classify_crop(effnet, crop, device, effnet_tf)
        class_counts[eff_cls_name] += 1

        color = [(255, 80, 80), (80, 220, 220), (80, 220, 80)][eff_cls_id]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{eff_cls_name} {eff_conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"YOLO: {yolo_name} {yolo_conf:.2f}",
            (x1, min(h - 8, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "id": idx,
                "box": [x1, y1, x2, y2],
                "yolo": {
                    "class_id": yolo_cls,
                    "class_name": yolo_name,
                    "confidence": round(yolo_conf, 4),
                },
                "efficientnet": {
                    "class_id": eff_cls_id,
                    "class_name": eff_cls_name,
                    "confidence": round(eff_conf, 4),
                    "probabilities": {
                        MP_NAMES[i]: round(float(eff_probs[i]), 4) for i in range(len(MP_NAMES))
                    },
                },
                # Only compare when YOLO class is one of the same type labels.
                "classification_changed": (yolo_name in MP_NAMES and yolo_name != eff_cls_name),
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    out_image = out_dir / f"{stem}_yolo_effnet_only.png"
    out_json = out_dir / f"{stem}_yolo_effnet_only.json"

    cv2.imwrite(str(out_image), vis)

    changed_count = sum(1 for d in detections if d["classification_changed"])
    payload = {
        "image": str(image_path),
        "yolo_model": str(yolo_model_path),
        "effnet_model": str(effnet_model_path),
        "confidence_threshold": conf_threshold,
        "raw_detection_count": len(raw_boxes),
        "final_detection_count": len(detections),
        "efficientnet_changed_count": changed_count,
        "class_counts": class_counts,
        "detections": detections,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved image: {out_image}")
    print(f"Saved labels: {out_json}")
    print(f"Class counts: {class_counts}")
    print(f"EfficientNet changed: {changed_count}/{len(detections)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + EfficientNet only MP pipeline demo")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--yolo-model", type=str, required=True, help="YOLO model path")
    parser.add_argument("--effnet-model", type=str, required=True, help="EfficientNet checkpoint path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prediction/demo_yolo_effnet_only",
        help="Directory for output image and JSON",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--crop-padding", type=int, default=15, help="Padding around YOLO boxes for ENet crops")
    args = parser.parse_args()

    run_yolo_effnet_only(
        image_path=args.image,
        yolo_model_path=args.yolo_model,
        effnet_model_path=args.effnet_model,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        crop_padding=args.crop_padding,
    )


if __name__ == "__main__":
    main()
