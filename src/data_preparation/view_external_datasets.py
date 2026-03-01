"""
Viewer for external datasets in datasets/external/.
Displays each image side-by-side with:
  - Left: image with OBB label polygons drawn on top
  - Right: binary mask generated from the OBB polygons

Navigation:
  Click  [< Prev]  button  → previous image
  Click  [Next >]  button  → next image
  Right arrow / D          → next image
  Left arrow  / A          → previous image
  Q / Escape               → quit

Usage:
  python src/data_preparation/view_external_datasets.py                     # view all splits
  python src/data_preparation/view_external_datasets.py --split train       # only train split
  python src/data_preparation/view_external_datasets.py --dataset maguayan  # only Maguayan dataset
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT / "datasets" / "external"

DATASETS = {
    "maguayan": EXTERNAL_DIR / "Maguayan Project.v9i.yolov8-obb",
    "seamap": EXTERNAL_DIR / "SEAMaP Binary Full.v6i.yolov8-obb",
}

# Colours for drawing (BGR)
LABEL_COLOR = (0, 255, 0)   # green polygons
MASK_COLOR = (255, 255, 255) # white fill on black


def parse_obb_label(label_path: Path, img_w: int, img_h: int):
    """Parse a YOLOv8-OBB label file and return a list of polygons.

    Each line can be:
      - OBB:  class x1 y1 x2 y2 x3 y3 x4 y4   (9 tokens, 4 corner-points)
      - AABB: class cx cy w h                    (5 tokens, axis-aligned box)

    Returns list of np.ndarray polygons, each shape (N, 2) in pixel coords.
    """
    polygons = []
    if not label_path.exists():
        return polygons

    with open(label_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue

            values = list(map(float, tokens[1:]))  # skip class id

            if len(values) == 8:
                # OBB: 4 corner-points (x1,y1,...,x4,y4) normalised
                pts = np.array(values, dtype=np.float64).reshape(4, 2)
                pts[:, 0] *= img_w
                pts[:, 1] *= img_h
                polygons.append(pts.astype(np.int32))

            elif len(values) == 4:
                # AABB: cx, cy, w, h  (normalised)
                cx, cy, w, h = values
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                pts = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ], dtype=np.int32)
                polygons.append(pts)

    return polygons


def draw_labels_on_image(img: np.ndarray, polygons: list) -> np.ndarray:
    """Draw OBB polygons on a copy of the image."""
    vis = img.copy()
    for pts in polygons:
        cv2.polylines(vis, [pts], isClosed=True, color=LABEL_COLOR, thickness=2)
        # semi-transparent fill
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
    return vis


def make_mask(img_h: int, img_w: int, polygons: list) -> np.ndarray:
    """Create a binary mask from the polygons."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for pts in polygons:
        cv2.fillPoly(mask, [pts], color=255)
    return mask


def collect_image_entries(dataset_names: list[str], splits: list[str]):
    """Collect (image_path, label_path, dataset_name, split) tuples."""
    entries = []
    for name in dataset_names:
        ds_root = DATASETS[name]
        if not ds_root.exists():
            print(f"[WARN] Dataset folder not found: {ds_root}")
            continue
        for split in splits:
            img_dir = ds_root / split / "images"
            lbl_dir = ds_root / split / "labels"
            if not img_dir.exists():
                continue
            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    continue
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                entries.append((img_file, lbl_file, name, split))
    return entries


TOOLBAR_H = 56          # height of the bottom toolbar in pixels
BTN_W = 160             # button width
BTN_H = 38              # button height
BTN_COLOR = (60, 60, 60)
BTN_HOVER_COLOR = (100, 100, 200)
BTN_TEXT_COLOR = (255, 255, 255)


def _draw_toolbar(canvas_w: int, idx: int, total: int, hovered: str | None) -> np.ndarray:
    """Return a (TOOLBAR_H x canvas_w) BGR toolbar with Prev / Next buttons."""
    bar = np.full((TOOLBAR_H, canvas_w, 3), 30, dtype=np.uint8)

    margin = 20
    btn_y0 = (TOOLBAR_H - BTN_H) // 2
    btn_y1 = btn_y0 + BTN_H

    # ── Prev button (left side) ──────────────────────────────────────────────
    prev_x0 = margin
    prev_x1 = margin + BTN_W
    prev_col = BTN_HOVER_COLOR if hovered == "prev" else BTN_COLOR
    cv2.rectangle(bar, (prev_x0, btn_y0), (prev_x1, btn_y1), prev_col, cv2.FILLED)
    cv2.rectangle(bar, (prev_x0, btn_y0), (prev_x1, btn_y1), (120, 120, 120), 1)
    cv2.putText(bar, "< Prev", (prev_x0 + 28, btn_y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BTN_TEXT_COLOR, 2, cv2.LINE_AA)

    # ── Next button (right side) ─────────────────────────────────────────────
    next_x1 = canvas_w - margin
    next_x0 = next_x1 - BTN_W
    next_col = BTN_HOVER_COLOR if hovered == "next" else BTN_COLOR
    cv2.rectangle(bar, (next_x0, btn_y0), (next_x1, btn_y1), next_col, cv2.FILLED)
    cv2.rectangle(bar, (next_x0, btn_y0), (next_x1, btn_y1), (120, 120, 120), 1)
    cv2.putText(bar, "Next >", (next_x0 + 28, btn_y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BTN_TEXT_COLOR, 2, cv2.LINE_AA)

    # ── Counter in centre ────────────────────────────────────────────────────
    counter = f"{idx + 1} / {total}"
    (tw, th), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(bar, counter, (canvas_w // 2 - tw // 2, btn_y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    return bar


def view(entries: list, start_idx: int = 0):
    """Interactive OpenCV viewer with clickable Prev / Next buttons."""
    if not entries:
        print("No images found.")
        return

    # Shared mutable state accessed by the mouse callback
    state = {"idx": start_idx, "action": None, "hover": None}
    total = len(entries)

    WIN = "External Dataset Viewer  [A/Left] Prev  [D/Right] Next  [Q/Esc] Quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1400, 660)

    # Cached dimensions for the current composite frame so the mouse callback
    # can map click coords to button regions without needing a closure over a
    # mutable variable that changes each frame.
    dims = {"canvas_w": 1400, "img_h": 600}

    def _btn_regions(canvas_w: int):
        margin = 20
        btn_y0 = (TOOLBAR_H - BTN_H) // 2
        btn_y1 = btn_y0 + BTN_H
        img_h = dims["img_h"]
        # y coords are offset by img_h (toolbar is below the image)
        prev = (margin, img_h + btn_y0, margin + BTN_W, img_h + btn_y1)
        next_x0 = canvas_w - margin - BTN_W
        nxt = (next_x0, img_h + btn_y0, canvas_w - margin, img_h + btn_y1)
        return prev, nxt

    def on_mouse(event, x, y, flags, param):
        canvas_w = dims["canvas_w"]
        prev_btn, next_btn = _btn_regions(canvas_w)

        def inside(rect):
            return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

        new_hover = None
        if inside(prev_btn):
            new_hover = "prev"
        elif inside(next_btn):
            new_hover = "next"

        if state["hover"] != new_hover:
            state["hover"] = new_hover
            state["action"] = "redraw"  # repaint for hover effect

        if event == cv2.EVENT_LBUTTONDOWN:
            if inside(prev_btn):
                state["action"] = "prev"
            elif inside(next_btn):
                state["action"] = "next"

    cv2.setMouseCallback(WIN, on_mouse)

    needs_render = True
    idx = state["idx"]

    while True:
        # ── Render frame if needed ────────────────────────────────────────────
        if needs_render:
            img_path, lbl_path, ds_name, split = entries[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Cannot read {img_path}, skipping.")
                idx = (idx + 1) % total
                continue

            img_h, img_w = img.shape[:2]
            polygons = parse_obb_label(lbl_path, img_w, img_h)
            n_labels = len(polygons)

            left = draw_labels_on_image(img, polygons)
            mask = make_mask(img_h, img_w, polygons)
            right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            info = (f"[{idx + 1}/{total}]  {ds_name}/{split}  |  "
                    f"{img_path.name}  |  {n_labels} labels")
            for panel in (left, right):
                cv2.putText(panel, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

            cv2.putText(left, "Image + Labels", (10, img_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(right, "Mask", (10, img_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

            sep = np.full((img_h, 3, 3), 100, dtype=np.uint8)
            panels = np.hstack([left, sep, right])
            canvas_w = panels.shape[1]

            dims["canvas_w"] = canvas_w
            dims["img_h"] = img_h

            toolbar = _draw_toolbar(canvas_w, idx, total, state["hover"])
            combined = np.vstack([panels, toolbar])

            cv2.imshow(WIN, combined)
            state["idx"] = idx
            needs_render = False

        # ── Redraw toolbar only (hover changed) ───────────────────────────────
        elif state["action"] == "redraw":
            toolbar = _draw_toolbar(dims["canvas_w"], idx, total, state["hover"])
            combined[-TOOLBAR_H:, :] = toolbar
            cv2.imshow(WIN, combined)
            state["action"] = None

        # ── Window closed via X button ────────────────────────────────────────
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(30) & 0xFF

        # ── Keyboard navigation ───────────────────────────────────────────────
        if key in (ord("q"), 27):
            break
        elif key in (ord("d"), 83, 0x27) or state["action"] == "next":
            idx = (idx + 1) % total
            state["action"] = None
            needs_render = True
        elif key in (ord("a"), 81, 0x25) or state["action"] == "prev":
            idx = (idx - 1) % total
            state["action"] = None
            needs_render = True

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="View external dataset images with labels & masks")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "maguayan", "seamap"],
                        help="Which dataset to view (default: all)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "valid", "test"],
                        help="Which split to view (default: all)")
    args = parser.parse_args()

    dataset_names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    print(f"Scanning datasets: {dataset_names}  |  splits: {splits}")
    entries = collect_image_entries(dataset_names, splits)
    print(f"Found {len(entries)} images. Launching viewer …")
    view(entries)


if __name__ == "__main__":
    main()
