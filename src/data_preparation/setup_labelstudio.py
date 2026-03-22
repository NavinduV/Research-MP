"""
setup_labelstudio.py
--------------------
Creates 3 independent Label Studio projects for microplastic annotation:
  - Fiber    → BrushLabels  (thin/elongated shapes)
  - Film     → PolygonLabels (flat irregular shapes)
  - Fragment → PolygonLabels (irregular chunks)

Each project imports image tasks from data/crops/<type>/ via local storage
so the file-count upload limit is never hit.

Prerequisites
─────────────
1. Label Studio running (locally or on AWS):
       label-studio start --port 8080

2. Label Studio must allow local file serving. Add these env vars BEFORE
   starting Label Studio:
       LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
       LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<absolute path to data/crops>

   Example (PowerShell):
       $env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
       $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="D:\\Research_Dev\\mp-detect\\data\\crops"
       label-studio start --port 8080

3. Install the SDK:
       pip install label-studio-sdk

Usage
─────
    python src/data_preparation/setup_labelstudio.py \
        --url http://localhost:8080 \
        --api-key <YOUR_API_KEY> \
        --crops-root data/crops

Find your API key at: http://localhost:8080/user/account  →  Access Token
"""

import argparse
import sys
import os
import requests
from pathlib import Path

try:
    from label_studio_sdk.client import LabelStudio
except ImportError:
    print("ERROR: label-studio-sdk not installed.")
    print("       Run: pip install label-studio-sdk")
    sys.exit(1)


# ── Labeling interface configs ────────────────────────────────────────────────

# Fiber: BrushLabels — better for thin/elongated shapes
FIBER_CONFIG = """<View>
  <Header value="Fiber Microplastic — Brush the full fiber body"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"/>
  <BrushLabels name="label" toName="image">
    <Label value="fiber" background="#FF6B6B" showInline="true"/>
  </BrushLabels>
</View>"""

# Film: PolygonLabels — flat, sheet-like irregular outlines
FILM_CONFIG = """<View>
  <Header value="Film Microplastic — Outline the full film boundary"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2">
    <Label value="film" background="#4ECDC4" showInline="true"/>
  </PolygonLabels>
</View>"""

# Fragment: PolygonLabels — irregular solid chunks
FRAGMENT_CONFIG = """<View>
  <Header value="Fragment Microplastic — Outline the full fragment boundary"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2">
    <Label value="fragment" background="#45B7D1" showInline="true"/>
  </PolygonLabels>
</View>"""


PROJECTS = [
    {
        "type": "fiber",
        "title": "MP Labelling — Fiber",
        "description": "Binary segmentation of fiber microplastics. "
                       "Use the brush tool to paint the full fiber body.",
        "config": FIBER_CONFIG,
        "color": "#FF6B6B",
    },
    {
        "type": "film",
        "title": "MP Labelling — Film",
        "description": "Binary segmentation of film microplastics. "
                       "Use polygons to trace the film boundary.",
        "config": FILM_CONFIG,
        "color": "#4ECDC4",
    },
    {
        "type": "fragment",
        "title": "MP Labelling — Fragment",
        "description": "Binary segmentation of fragment microplastics. "
                       "Use polygons to trace the fragment boundary.",
        "config": FRAGMENT_CONFIG,
        "color": "#45B7D1",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_image_paths(crops_root: Path, mp_type: str) -> list[str]:
    """Return sorted list of absolute PNG paths for a given type directory."""
    type_dir = crops_root / mp_type
    if not type_dir.exists():
        print(f"  WARNING: {type_dir} not found — no tasks will be imported.")
        return []
    paths = sorted(type_dir.glob("*.png"))
    return [str(p.resolve()) for p in paths]


def build_tasks(image_paths: list[str], crops_root: Path) -> list[dict]:
    """
    Build Label Studio task dicts using /data/local-files/?d= URL scheme.
    Paths are made relative to crops_root (= LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT).
    """
    tasks = []
    crops_root_abs = crops_root.resolve().as_posix()
    for p in image_paths:
        rel = Path(p).resolve().as_posix()
        if rel.startswith(crops_root_abs + "/"):
            rel = rel[len(crops_root_abs) + 1:]
        url = f"/data/local-files/?d={rel}"
        tasks.append({"data": {"image": url}})
    return tasks


def create_or_get_project(ls: LabelStudio, title: str, description: str,
                           label_config: str, color: str) -> object:
    """Create a new project (skip if title already exists)."""
    for proj in ls.projects.list():
        if proj.title == title:
            print(f"  → Project already exists (id={proj.id}), reusing.")
            ls.projects.update(proj.id, label_config=label_config)
            return proj

    proj = ls.projects.create(
        title=title,
        description=description,
        label_config=label_config,
        color=color,
    )
    print(f"  → Created project id={proj.id}")
    return proj


def import_tasks(api_key: str, ls_url: str, project_id: int,
                 image_paths: list[str]) -> None:
    """
    Upload image files directly to Label Studio via multipart POST.
    This is the most reliable method — no local-file-serving env vars needed.
    Each uploaded file becomes one task automatically.
    """
    if not image_paths:
        return

    headers = {"Authorization": f"Token {api_key}"}
    BATCH = 50  # files per request
    total = 0

    for i in range(0, len(image_paths), BATCH):
        batch = image_paths[i : i + BATCH]
        files = []
        handles = []
        try:
            for p in batch:
                fh = open(p, "rb")
                handles.append(fh)
                files.append(("file", (Path(p).name, fh, "image/png")))

            resp = requests.post(
                f"{ls_url}/api/projects/{project_id}/import",
                headers=headers,
                files=files,
                timeout=120,
            )
        finally:
            for fh in handles:
                fh.close()

        if resp.status_code in (200, 201):
            imported = resp.json().get("task_count", len(batch))
            total += imported
            print(f"  → Uploaded {total}/{len(image_paths)} ...")
        else:
            print(f"  ERROR HTTP {resp.status_code}: {resp.text[:300]}")
            return

    print(f"  → Upload complete: {total} tasks")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create 3 Label Studio projects for MP type labelling."
    )
    parser.add_argument("--url", default="http://localhost:8080",
                        help="Label Studio URL (default: http://localhost:8080)")
    parser.add_argument("--api-key", required=True,
                        help="Label Studio API key (from Account → Access Token)")
    parser.add_argument("--crops-root", default="data/crops",
                        help="Path to crops root dir with fiber/, film/, fragment/ subfolders")
    parser.add_argument("--projects-only", action="store_true",
                        help="Only create projects, skip image import (upload manually later)")
    args = parser.parse_args()

    projects_only = args.projects_only
    crops_root = Path(args.crops_root).resolve()
    if not projects_only and not crops_root.exists():
        print(f"ERROR: crops-root not found: {crops_root}")
        print("       Use --projects-only to skip image import.")
        sys.exit(1)

    ls_url  = args.url.rstrip("/")
    api_key = args.api_key

    print(f"\nConnecting to Label Studio at {ls_url} ...")
    ls = LabelStudio(base_url=ls_url, api_key=api_key)

    try:
        ls.users.whoami()
        print("Connection OK ✅\n")
    except Exception as e:
        print(f"ERROR: Could not connect — {e}")
        print("Check --url and --api-key, and ensure Label Studio is running.")
        sys.exit(1)

    for cfg in PROJECTS:
        mp_type = cfg["type"]
        print(f"{'='*55}")
        print(f"Setting up project: {cfg['title']}")
        print(f"{'='*55}")

        # 1. Create / reuse project
        proj = create_or_get_project(
            ls,
            title=cfg["title"],
            description=cfg["description"],
            label_config=cfg["config"],
            color=cfg["color"],
        )

        if projects_only:
            print(f"  ✅ {cfg['title']} ready (projects-only mode, no import)\n")
            continue

        # 2. Check existing task count — skip if already populated
        try:
            existing_count = ls.tasks.list(project=proj.id, page_size=1)
            existing_n = getattr(existing_count, 'total', None)
            if existing_n is None:
                existing_n = len(list(ls.tasks.list(project=proj.id)))
            if existing_n > 0:
                print(f"  → Already has {existing_n} tasks, skipping import.")
                print(f"  ✅ {cfg['title']} ready\n")
                continue
        except Exception:
            pass

        # 3. Scan images on disk
        image_paths = get_image_paths(crops_root, mp_type)
        print(f"  Found {len(image_paths)} images in {crops_root / mp_type}")

        if len(image_paths) == 0:
            print("  Skipping task import (no images found).")
            continue

        # 4. Upload images directly
        import_tasks(api_key, ls_url, proj.id, image_paths)

        print(f"  ✅ {cfg['title']} ready\n")

    print("=" * 55)
    print("All 3 projects ready ✅")
    print(f"Open: {ls_url}/projects/")
    print("=" * 55)
    print()
    print("NOTE: Images are served as local files.")
    print("      Label Studio MUST be started with:")
    print("        $env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED='true'")
    print(f"        $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT='{crops_root}'")


if __name__ == "__main__":
    main()
