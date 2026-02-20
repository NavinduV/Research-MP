# Microplastic Detection Dashboard

A full-stack pipeline dashboard — FastAPI ML backend + React/Vite frontend.

## Architecture

```
mp-detect/
├── src/
│   ├── api/
│   │   └── pipeline_api.py   ← FastAPI backend (port 8000)
│   ├── pipeline_inference.py ← ML pipeline (YOLO + EfficientNet + Mask R-CNN)
│   └── app/                  ← Legacy stitching app
├── frontend/                 ← React/Vite dashboard (port 5173)
│   └── src/
│       ├── pages/
│       │   ├── UploadPage.jsx   ← Image upload + pipeline config
│       │   ├── ResultsPage.jsx  ← Full results dashboard
│       │   └── HistoryPage.jsx  ← Past job browser
│       ├── api/detect.js        ← API client
│       └── components/
├── start_dashboard.ps1       ← One-click launcher (Windows)
└── requirements.txt
```

## Quick Start

### 1. Start both servers (Windows PowerShell)

```powershell
# From the repo root
.\start_dashboard.ps1
```

Or manually:

```powershell
# Terminal 1 – Backend
cd D:\Research_Dev\mp-detect
.\venv\Scripts\activate
uvicorn src.api.pipeline_api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 – Frontend
cd D:\Research_Dev\mp-detect\frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Features

### Detect Page
- Drag-and-drop image upload (multiple files)
- Toggle EfficientNet classification on/off
- Toggle Mask R-CNN segmentation on/off
- Sliders for YOLO confidence, mask threshold, pixel→micron ratio, crop padding
- Advanced panel for custom model weight paths

### Results Page (post-run)
All sections are collapsible:

| Section | What it shows |
|---|---|
| Overview tiles | Total count, per-class counts with %, avg confidences, reclassification rate, avg length |
| Detection Counts donut | Pie/donut chart of fiber / film / fragment |
| Length Distribution | Histogram of particle lengths |
| Per-Class Size Comparison | Bar charts: mean length, area, circularity per class |
| Shape Analysis | Scatter (length vs circularity, bubble = area) + Radar chart |
| Confidence Analysis | Stacked bar: YOLO confidence buckets per class |
| Visualizations | Annotated / mask overlay / original image viewer with multi-image tabs |
| Detailed Statistics Table | Mean / median / min / max for every metric × every class |
| All Detections Table | Sortable, filterable per-detection table with all metrics |
| Pipeline Config | Summary of parameters used |
| Export JSON | Download full report |

### History Page
- Lists all in-session jobs with status and timestamp
- Click to reload any past result into Results page

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET  | `/api/health` | Backend status & device |
| POST | `/api/detect` | Run pipeline on uploaded image(s) |
| GET  | `/api/results/{job_id}` | Fetch stored report JSON |
| GET  | `/api/image/{job_id}/{idx}` | Annotated visualization |
| GET  | `/api/mask/{job_id}/{idx}` | Mask overlay |
| GET  | `/api/original/{job_id}/{idx}` | Original upload |
| GET  | `/api/jobs` | List all in-memory jobs |

Full interactive docs: **http://localhost:8000/docs**
