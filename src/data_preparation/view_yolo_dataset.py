"""
View YOLO dataset images with label overlays.

Supports:
- Standard YOLO detection labels: class cx cy w h
- YOLO segmentation labels: class x1 y1 x2 y2 ...

Examples:
    python src/data_preparation/view_yolo_dataset.py \
        --yolo-dir data/macro/yolo

    python src/data_preparation/view_yolo_dataset.py \
        --yolo-dir "D:/Research_Dev/mp-detect/data/macro/yolo" \
        --image IMG_0001.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_class_names(yolo_dir: Path) -> dict[int, str]:
    """Load class names from dataset metadata when available."""
    candidates = [
        yolo_dir / "dataset.yaml",
        yolo_dir / "data.yaml",
        Path("dataset.yaml"),
    ]

    for cfg_path in candidates:
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"[WARN] Failed to read class names from {cfg_path}: {exc}")
            continue

        names = cfg.get("names", {})
        if isinstance(names, list):
            mapped = {i: str(name) for i, name in enumerate(names)}
            print(f"Loaded class names from {cfg_path}")
            return mapped

        if isinstance(names, dict):
            mapped: dict[int, str] = {}
            for k, v in names.items():
                try:
                    mapped[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
            if mapped:
                print(f"Loaded class names from {cfg_path}")
                return mapped

    print("No dataset.yaml/data.yaml class names found. Falling back to class_<id> labels.")
    return {}


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Generate a stable, distinct BGR color per class id."""
    b = (37 * class_id + 79) % 255
    g = (17 * class_id + 131) % 255
    r = (53 * class_id + 191) % 255
    return int(b), int(g), int(r)


def parse_label_file(label_path: Path, width: int, height: int) -> list[dict]:
    """Parse YOLO label file into drawable annotations."""
    if not label_path.exists():
        return []

    annotations: list[dict] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                continue

            if len(coords) == 4:
                cx, cy, bw, bh = coords
                x1 = int((cx - bw / 2.0) * width)
                y1 = int((cy - bh / 2.0) * height)
                x2 = int((cx + bw / 2.0) * width)
                y2 = int((cy + bh / 2.0) * height)

                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))

                annotations.append(
                    {
                        "type": "bbox",
                        "class_id": class_id,
                        "bbox": (x1, y1, x2, y2),
                    }
                )
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
                pts[:, 0] *= width
                pts[:, 1] *= height

                pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

                annotations.append(
                    {
                        "type": "polygon",
                        "class_id": class_id,
                        "points": pts.astype(np.int32),
                    }
                )

    return annotations


def find_image_files(yolo_dir: Path) -> list[Path]:
    """Collect image files from YOLO dataset structure."""
    image_files: list[Path] = []

    images_root = yolo_dir / "images"
    if images_root.exists():
        for split in ["train", "val", "test"]:
            split_dir = images_root / split
            if split_dir.exists():
                image_files.extend(
                    sorted([p for p in split_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
                )

        image_files.extend(
            sorted(
                [
                    p
                    for p in images_root.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS
                ]
            )
        )
    else:
        image_files.extend(
            sorted([p for p in yolo_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        )

    return image_files


def get_label_path(yolo_dir: Path, image_path: Path) -> Path:
    """Map image path to expected YOLO label path."""
    images_root = yolo_dir / "images"
    labels_root = yolo_dir / "labels"

    if images_root.exists() and labels_root.exists():
        try:
            relative = image_path.relative_to(images_root)
            return labels_root / relative.with_suffix(".txt")
        except ValueError:
            pass

    return image_path.with_suffix(".txt")


def draw_annotations(
    image: np.ndarray,
    annotations: list[dict],
    class_names: dict[int, str],
) -> np.ndarray:
    """Draw labels on an image copy."""
    vis = image.copy()

    for ann in annotations:
        class_id = ann["class_id"]
        color = color_for_class(class_id)
        class_name = class_names.get(class_id, f"class_{class_id}")

        if ann["type"] == "bbox":
            x1, y1, x2, y2 = ann["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            tx, ty = x1, max(20, y1 - 8)
            cv2.putText(
                vis,
                class_name,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        elif ann["type"] == "polygon":
            pts = ann["points"]
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)

            x, y, w, h = cv2.boundingRect(pts)
            tx, ty = x, max(20, y - 8)
            cv2.putText(
                vis,
                class_name,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return vis


def resize_for_screen(image: np.ndarray, max_w: int = 1600, max_h: int = 900) -> np.ndarray:
    """Resize large images to fit typical screens while preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return image

    nw = int(w * scale)
    nh = int(h * scale)
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def resolve_single_image(image_arg: str, image_files: list[Path], yolo_dir: Path) -> Path | None:
    """Resolve user image input as absolute path, relative path, or filename."""
    candidate = Path(image_arg)

    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    relative_candidate = (yolo_dir / image_arg).resolve()
    if relative_candidate.exists() and relative_candidate.is_file():
        return relative_candidate

    needle = image_arg.lower()
    for img_path in image_files:
        if img_path.name.lower() == needle:
            return img_path.resolve()

    for img_path in image_files:
        if img_path.stem.lower() == Path(needle).stem:
            return img_path.resolve()

    return None


def run_viewer(yolo_dir: Path, image_arg: str | None) -> None:
    image_files = find_image_files(yolo_dir)
    if not image_files:
        print(f"No image files found in {yolo_dir}")
        return

    class_names = load_class_names(yolo_dir)

    if image_arg:
        target = resolve_single_image(image_arg, image_files, yolo_dir)
        if target is None:
            print(f"Could not find image: {image_arg}")
            return
        image_files = [target]

    index = 0
    total = len(image_files)

    win_name = "YOLO Dataset Viewer  [A/Left] Prev  [D/Right] Next  [Q/Esc] Quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        image_path = image_files[index]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Unable to read image: {image_path}")
            if total == 1:
                break
            index = (index + 1) % total
            continue

        height, width = image.shape[:2]
        label_path = get_label_path(yolo_dir, image_path)
        annotations = parse_label_file(label_path, width, height)
        vis = draw_annotations(image, annotations, class_names)

        rel = image_path
        try:
            rel = image_path.relative_to(yolo_dir)
        except ValueError:
            pass

        title_1 = f"{index + 1}/{total}  |  {rel}"
        title_2 = f"objects: {len(annotations)}  |  label: {label_path.name if label_path.exists() else 'missing'}"
        cv2.putText(vis, title_1, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, title_2, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        vis = resize_for_screen(vis)
        cv2.imshow(win_name, vis)

        key = cv2.waitKeyEx(0)
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("d"), ord("D"), ord("n"), ord("N"), 2555904, 83):
            index = (index + 1) % total
            continue
        if key in (ord("a"), ord("A"), ord("p"), ord("P"), 2424832, 81):
            index = (index - 1) % total
            continue

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="View YOLO images with label overlays")
    parser.add_argument(
        "--yolo-dir",
        type=str,
        required=True,
        help="Path to YOLO dataset root (expects images/ and labels/ folders)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path or filename to view only that image",
    )

    args = parser.parse_args()
    yolo_dir = Path(args.yolo_dir).resolve()

    if not yolo_dir.exists():
        print(f"YOLO directory does not exist: {yolo_dir}")
        return

    run_viewer(yolo_dir, args.image)


if __name__ == "__main__":
    main()