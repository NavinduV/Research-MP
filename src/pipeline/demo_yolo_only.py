"""
YOLO-only microplastic labeling demo.

Given an input image, this script:
1. Runs YOLO detection
2. Applies class-agnostic NMS for one box per physical MP
3. Draws MP labels on the image
4. Saves an annotated image and a JSON report

It intentionally does not run EfficientNet or Mask R-CNN.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


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


def run_yolo_only(
    image_path: str,
    yolo_model_path: str,
    output_dir: str,
    conf_threshold: float,
    use_agnostic_nms: bool,
) -> Dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    model = YOLO(yolo_model_path)
    result = model(image_path, conf=conf_threshold, verbose=False)[0]

    raw_boxes: List[List[float]] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        raw_boxes.append([x1, y1, x2, y2, cls_id, conf])

    final_boxes = class_agnostic_nms(raw_boxes) if use_agnostic_nms else raw_boxes

    vis = image.copy()
    class_counts: Dict[str, int] = {}
    detections = []
    palette = [
        (255, 80, 80),
        (80, 220, 220),
        (80, 220, 80),
        (220, 180, 80),
        (180, 80, 220),
        (80, 180, 220),
    ]

    for idx, bx in enumerate(final_boxes, start=1):
        x1, y1, x2, y2 = map(int, bx[:4])
        yolo_cls = int(bx[4])
        yolo_conf = float(bx[5])

        yolo_name = get_model_class_name(result.names, yolo_cls)
        class_counts[yolo_name] = class_counts.get(yolo_name, 0) + 1

        color = palette[yolo_cls % len(palette)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{yolo_name} {yolo_conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "id": idx,
                "box": [x1, y1, x2, y2],
                "class_id": yolo_cls,
                "class_name": yolo_name,
                "confidence": round(yolo_conf, 4),
                "source": "yolo",
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    out_image = out_dir / f"{stem}_yolo_only.png"
    out_json = out_dir / f"{stem}_yolo_only.json"

    cv2.imwrite(str(out_image), vis)

    payload = {
        "image": str(image_path),
        "yolo_model": str(yolo_model_path),
        "confidence_threshold": conf_threshold,
        "raw_detection_count": len(raw_boxes),
        "final_detection_count": len(final_boxes),
        "class_counts": class_counts,
        "detections": detections,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved image: {out_image}")
    print(f"Saved labels: {out_json}")
    print(f"Class counts: {class_counts}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-only MP labeling demo")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--yolo-model", type=str, required=True, help="YOLO model path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prediction/demo_yolo_only",
        help="Directory for output image and JSON",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument(
        "--no-agnostic-nms",
        action="store_true",
        help="Disable class-agnostic NMS post-processing",
    )
    args = parser.parse_args()

    run_yolo_only(
        image_path=args.image,
        yolo_model_path=args.yolo_model,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        use_agnostic_nms=not args.no_agnostic_nms,
    )


if __name__ == "__main__":
    main()
