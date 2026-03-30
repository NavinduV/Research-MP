"""
Filter detection only demo.

Given an input image, this script:
1. Detects the filter circle (edge-Hough and fallbacks)
2. Draws only the detected filter overlay
3. Saves the annotated image

It intentionally does not run YOLO, EfficientNet, or Mask R-CNN.
"""

import argparse
from pathlib import Path

import cv2

try:
    from src.pipeline.filter_mask import detect_filter_circle_from_array
except ImportError:
    from filter_mask import detect_filter_circle_from_array


def run_filter_only(image_path: str, output_path: str = None) -> Path:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    center, radius, mask, method, full_coverage = detect_filter_circle_from_array(image)

    vis = image.copy()

    if full_coverage:
        # Show full-coverage mode in the output image.
        cv2.putText(
            vis,
            "Filter detection: full-coverage",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.circle(vis, center, radius, (0, 255, 0), 3)
        cv2.circle(vis, center, 5, (0, 0, 255), -1)

        # Light mask overlay so the kept region is obvious.
        overlay = vis.copy()
        overlay[mask == 0] = (20, 20, 20)
        vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0.0)

        cv2.putText(
            vis,
            f"method={method} center={center} r={radius}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if output_path:
        out_path = Path(output_path)
    else:
        out_dir = Path("prediction") / "demo_filter_only"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(image_path).stem}_filter_only.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter-only detection demo")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    run_filter_only(args.image, args.output)


if __name__ == "__main__":
    main()
