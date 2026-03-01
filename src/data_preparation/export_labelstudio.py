"""
export_labelstudio.py
---------------------
Exports all 3 Label Studio projects and converts them directly into the
images/ + masks/ + annotations.json structure needed for Mask R-CNN training.

What it does per project:
  1. Downloads the JSON export  (task → filename + RLE brush masks)
  2. Decodes RLE brush masks → numpy binary masks
  3. Downloads task images from LS API
  4. Writes output to data/crops_{type}_sam/

Output structure:
    data/crops_fiber_sam/
        images/          <- original crop images
        masks/           <- binary PNG masks (255=object, 0=bg)
        annotations.json
    data/crops_film_sam/   (same)
    data/crops_fragment_sam/ (same)

Usage:
    python src/data_preparation/export_labelstudio.py --api-key <TOKEN>
    python src/data_preparation/export_labelstudio.py --api-key <TOKEN> --url http://localhost:8080
    python src/data_preparation/export_labelstudio.py --api-key <TOKEN> --types fiber film
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from label_studio_converter.brush import decode_rle

LS_URL = "http://localhost:8080"
CROPS_ROOT = Path("data/crops")
OUTPUT_ROOT = Path("data")

TYPE_TO_TITLE = {
    "fiber":    "MP Labelling — Fiber",
    "film":     "MP Labelling — Film",
    "fragment": "MP Labelling — Fragment",
}
TYPE_TO_CLASS_ID = {"fiber": 0, "film": 1, "fragment": 2}


# ── API helpers ───────────────────────────────────────────────────────────────

def create_session(ls_url: str, email: str, password: str) -> requests.Session:
    """
    Login to Label Studio with email/password and return an authenticated Session.
    Works with Label Studio 1.21.0+ where legacy token auth is disabled.
    """
    session = requests.Session()

    # Try token auth first (works if LABEL_STUDIO_LEGACY_TOKEN_AUTH=true)
    # Then fall back to session auth with email/password
    return session


def get_auth_session(ls_url: str, api_key: str = None,
                     email: str = None, password: str = None) -> requests.Session:
    """
    Returns an authenticated requests.Session.
    Tries token auth first; falls back to email/password session login.
    """
    session = requests.Session()

    if api_key:
        # Try token auth
        session.headers["Authorization"] = f"Token {api_key}"
        r = session.get(f"{ls_url}/api/projects/?page_size=1", timeout=10)
        if r.status_code == 200:
            print("Authenticated with token ✅")
            return session
        print(f"Token auth failed ({r.status_code}), trying session login...")
        del session.headers["Authorization"]

    if not email or not password:
        # Prompt interactively
        email = email or input("Label Studio email: ").strip()
        password = password or input("Label Studio password: ").strip()

    # Session login
    # First get CSRF token
    login_page = session.get(f"{ls_url}/user/login/", timeout=10)
    csrf = session.cookies.get("csrftoken", "")

    r = session.post(
        f"{ls_url}/user/login/",
        data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{ls_url}/user/login/"},
        timeout=10,
        allow_redirects=False,
    )
    # Successful login returns 302 redirect
    if r.status_code in (200, 302):
        # Verify we're actually logged in
        test = session.get(f"{ls_url}/api/projects/?page_size=1", timeout=10)
        if test.status_code == 200:
            print("Authenticated with session login ✅")
            return session

    print(f"ERROR: Login failed (HTTP {r.status_code})")
    sys.exit(1)


def api_get(session: requests.Session, url: str, stream: bool = False) -> requests.Response:
    r = session.get(url, stream=stream, timeout=120)
    if r.status_code != 200:
        print(f"  ERROR {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    return r


def get_projects(session: requests.Session, ls_url: str) -> dict[str, int]:
    """Returns {title: project_id} for all projects."""
    r = api_get(session, f"{ls_url}/api/projects/?page_size=100")
    return {p["title"]: p["id"] for p in r.json().get("results", [])}


def export_download(session: requests.Session, ls_url: str, project_id: int,
                    export_type: str) -> bytes:
    """Download an export as raw bytes using the simple GET export endpoint."""
    url = f"{ls_url}/api/projects/{project_id}/export?exportType={export_type}"
    print(f"  Downloading {export_type} export ...", end=" ", flush=True)
    r = api_get(session, url, stream=True)
    data = r.content
    print(f"{len(data)//1024} KB")
    return data


# ── Core conversion ───────────────────────────────────────────────────────────

def _strip_upload_prefix(filename: str) -> str:
    """Strip the 8-hex UUID prefix that Label Studio adds on upload."""
    parts = filename.split("-", 1)
    if len(parts) == 2 and len(parts[0]) == 8:
        try:
            int(parts[0], 16)
            return parts[1]
        except ValueError:
            pass
    return filename


def parse_json_export(json_bytes: bytes) -> tuple[dict[int, str], dict[int, list[np.ndarray]]]:
    """
    Parse Label Studio JSON export.
    Returns:
        task_id_to_name:  {task_id: original_filename}
        task_id_to_masks: {task_id: [mask_array, ...]}  — decoded from RLE
    """
    tasks = json.loads(json_bytes)
    id_to_name: dict[int, str] = {}
    id_to_masks: dict[int, list[np.ndarray]] = {}

    for task in tasks:
        task_id = task["id"]
        img_url = task.get("data", {}).get("image", "")

        # Extract clean filename
        if "?d=" in img_url:
            name = Path(img_url.split("?d=")[-1]).name
        elif "/upload/" in img_url:
            name = _strip_upload_prefix(Path(img_url).name)
        else:
            name = Path(img_url).name
        id_to_name[task_id] = name

        # Decode brush RLE masks from annotations
        masks: list[np.ndarray] = []
        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                if res.get("type") != "brushlabels":
                    continue
                val = res.get("value", {})
                rle = val.get("rle")
                if not rle:
                    continue
                w = res.get("original_width", 0)
                h = res.get("original_height", 0)
                if w == 0 or h == 0:
                    continue
                # decode_rle returns flat RGBA byte list
                flat = decode_rle(rle)
                arr = np.array(flat, dtype=np.uint8)
                if len(arr) == w * h * 4:
                    rgba = arr.reshape((h, w, 4))
                    mask = rgba[:, :, 3]  # alpha channel = brush mask
                    masks.append(mask)
        if masks:
            id_to_masks[task_id] = masks

    return id_to_name, id_to_masks


def build_training_dir(mp_type: str, task_id_to_name: dict[int, str],
                        task_id_to_masks: dict[int, list[np.ndarray]],
                        task_images: dict[int, bytes],
                        crops_source: Path, output_dir: Path) -> None:
    """
    Build the images/ + masks/ + annotations.json training directory.
    task_id_to_masks: {task_id: [mask_np_array, ...]} — already-decoded numpy masks.
    task_images: {task_id: image_bytes} downloaded from LS (preferred) — can be empty.
    crops_source: fallback local directory with source images.
    """
    images_out = output_dir / "images"
    masks_out  = output_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # Build suffix index of local crops for fuzzy matching
    # e.g. "s8_gt0042_fiber.png" → full path "07492891-s8_gt0042_fiber.png"
    suffix_index: dict[str, Path] = {}
    if crops_source.exists():
        for fp in crops_source.glob("*.png"):
            # Strip UUID prefix (8hex-rest)
            parts = fp.name.split("-", 1)
            if len(parts) == 2:
                suffix_index[parts[1]] = fp
            suffix_index[fp.name] = fp  # exact match too

    ann_lookup = {}
    processed = skipped = no_mask = 0

    for task_id, orig_name in task_id_to_name.items():
        img_bytes = task_images.get(task_id)
        src_path = None

        if img_bytes:
            # Decode the image we downloaded from Label Studio
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            # Try to find locally: exact match → suffix match → parent dirs
            candidates = [
                crops_source / orig_name,
                suffix_index.get(orig_name),
                crops_source.parent / orig_name,
                crops_source.parent / "images" / orig_name,
            ]
            src_path = next((p for p in candidates if p is not None and p.exists()), None)
            if src_path is None:
                print(f"  SKIP (src not found): {orig_name}")
                skipped += 1
                continue
            img = cv2.imread(str(src_path))

        if img is None:
            print(f"  SKIP (unreadable): {orig_name}")
            skipped += 1
            continue
        h, w = img.shape[:2]

        # Merge all mask layers for this task
        merged = np.zeros((h, w), dtype=np.uint8)
        mask_arrays = task_id_to_masks.get(task_id, [])

        if not mask_arrays:
            no_mask += 1  # task exists but no brush annotation — still copy image

        for m in mask_arrays:
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            merged = np.maximum(merged, m)

        _, merged = cv2.threshold(merged, 127, 255, cv2.THRESH_BINARY)

        # Save image → images/
        cv2.imwrite(str(images_out / orig_name), img)

        # Save mask → masks/
        mask_filename = f"{Path(orig_name).stem}_mask.png"
        cv2.imwrite(str(masks_out / mask_filename), merged)

        ann_lookup[orig_name] = {
            "mask_file":  mask_filename,
            "class_name": mp_type,
            "class_id":   TYPE_TO_CLASS_ID.get(mp_type, 0),
        }
        processed += 1

    with open(output_dir / "annotations.json", "w") as f:
        json.dump(ann_lookup, f, indent=2)

    print(f"  ✅ {processed} tasks converted"
          + (f", {skipped} skipped" if skipped else "")
          + (f", {no_mask} without brush mask" if no_mask else ""))
    print(f"     images/  → {images_out}  ({processed} files)")
    print(f"     masks/   → {masks_out}   ({processed} files)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export Label Studio projects → training images/ + masks/ structure."
    )
    parser.add_argument("--api-key", default=None,
                        help="Label Studio API token (optional — will prompt for login if not provided or rejected)")
    parser.add_argument("--email", default=None,
                        help="Label Studio login email (prompted if needed)")
    parser.add_argument("--password", default=None,
                        help="Label Studio login password (prompted if needed)")
    parser.add_argument("--url", default=LS_URL,
                        help=f"Label Studio URL (default: {LS_URL})")
    parser.add_argument("--crops-root", default=str(CROPS_ROOT),
                        help="Root dir with fiber/, film/, fragment/ source images")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT),
                        help="Root output dir (default: data/)")
    parser.add_argument("--types", nargs="+", default=["fiber", "film", "fragment"],
                        choices=["fiber", "film", "fragment"],
                        help="Which types to export (default: all 3)")
    args = parser.parse_args()

    ls_url     = args.url.rstrip("/")
    crops_root = Path(args.crops_root)
    out_root   = Path(args.output_root)

    print(f"\nConnecting to Label Studio at {ls_url} ...")
    session = get_auth_session(ls_url, args.api_key, args.email, args.password)
    projects = get_projects(session, ls_url)
    if not projects:
        print("ERROR: No projects found. Check --api-key and --url.")
        sys.exit(1)
    print(f"Found {len(projects)} project(s): {list(projects.keys())}\n")

    for mp_type in args.types:
        title = TYPE_TO_TITLE[mp_type]
        if title not in projects:
            print(f"⚠️  Project '{title}' not found — skipping {mp_type}.")
            continue

        proj_id    = projects[title]
        output_dir = out_root / f"crops_{mp_type}_sam"

        print(f"{'='*55}")
        print(f"Exporting: {title}  (id={proj_id})")
        print(f"  → {output_dir}")
        print(f"{'='*55}")

        # 1. JSON export → filenames + RLE masks decoded in one pass
        json_bytes = export_download(session, ls_url, proj_id, "JSON")
        task_id_to_name, task_id_to_masks = parse_json_export(json_bytes)
        n_with_mask = sum(1 for v in task_id_to_masks.values() if v)
        print(f"  Tasks in export: {len(task_id_to_name)}, with brush masks: {n_with_mask}")

        # 2. Download task images from Label Studio
        task_images: dict[int, bytes] = {}
        for task_id, orig_name in task_id_to_name.items():
            try:
                task_resp = session.get(f"{ls_url}/api/tasks/{task_id}", timeout=10)
                if task_resp.status_code == 200:
                    img_url = task_resp.json().get("data", {}).get("image", "")
                    if img_url:
                        full_url = f"{ls_url}{img_url}" if img_url.startswith("/") else img_url
                        img_resp = session.get(full_url, timeout=30)
                        if img_resp.status_code == 200:
                            task_images[task_id] = img_resp.content
            except Exception as e:
                print(f"  WARN: Could not download image for task {task_id}: {e}")
        print(f"  Downloaded {len(task_images)}/{len(task_id_to_name)} images from LS")

        # 3. Build training directory
        build_training_dir(
            mp_type=mp_type,
            task_id_to_name=task_id_to_name,
            task_id_to_masks=task_id_to_masks,
            task_images=task_images,
            crops_source=crops_root / mp_type,
            output_dir=output_dir,
        )
        print()

    print("="*55)
    print("Export complete ✅")
    print("Training directories:")
    for mp_type in args.types:
        p = out_root / f"crops_{mp_type}_sam"
        n_img = len(list((p / "images").glob("*.png"))) if (p / "images").exists() else 0
        print(f"  {mp_type:>10}: {p}  ({n_img} images)")
    print("="*55)


if __name__ == "__main__":
    main()
