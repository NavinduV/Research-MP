# Microplastic Detection Pipeline - Getting Started Guide

## Overview

This project detects microplastics (fibers, films, fragments) in macro images of filter papers using a three-stage deep learning pipeline:

1. **YOLO** - Fast object detection (finds bounding boxes)
2. **Mask R-CNN** - Instance segmentation (precise boundaries)
3. **EfficientNet** - Classification (fiber/film/fragment)

---

## Quick Start: Test Pipeline WITHOUT Training

You can test the entire pipeline using pretrained models to validate everything works:

```powershell
# Activate environment
.\venv\Scripts\activate

# Install dependencies (if not done)
pip install -r requirements.txt

# Test with your stitched image
python src/test_pipeline.py --image "dev-test/stitched/s7.png" --model all
```

This will:
- ✅ Validate your environment is set up correctly
- ✅ Test YOLO, Mask R-CNN, and EfficientNet loading
- ✅ Save visualization results to `experiments/pipeline_test/`
- ❌ Won't detect actual microplastics (needs fine-tuning)

---

## Step-by-Step: Label Your Data

### Step 1: Start Label Studio

```powershell
# In a separate terminal
.\venv\Scripts\activate
label-studio
```

Open browser: http://localhost:8080

### Step 2: Create Project

1. Click **"Create Project"**
2. Name it: `Microplastic Detection`
3. Go to **"Labeling Setup"** tab
4. Select **"Object Detection with Bounding Boxes"**
5. Replace the config with:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Fiber" background="red"/>
    <Label value="Film" background="green"/>
    <Label value="Fragment" background="blue"/>
  </RectangleLabels>
</View>
```

### Step 3: Import Your Stitched Images

1. Go to project settings → **"Cloud Storage"** or **"Import"**
2. Upload your stitched macro images from `data/stitched/` or `dev-test/stitched/`

### Step 4: Label Microplastics

For each image:
1. Draw bounding boxes around each microplastic
2. Select the class: **Fiber**, **Film**, or **Fragment**
3. Click **Submit** when done

**Tips:**
- **Fibers**: Long, thin, thread-like particles
- **Films**: Flat, sheet-like, often transparent
- **Fragments**: Irregular chunks, broken pieces

### Step 5: Export Annotations

1. Go to project → **Export**
2. Choose **"JSON"** format
3. Save as `data/labelstudio_export.json`

---

## Step-by-Step: Train Models

### Step 1: Convert Labels to YOLO Format

```powershell
# Setup YOLO directory structure
python src/train_yolo.py --mode setup

# Convert Label Studio export to YOLO format
python src/convert_labels.py --input data/labelstudio_export.json --output data/yolo --format yolo --images data/stitched
```

### Step 2: Train YOLO

```powershell
# Train YOLO (start with small model for testing)
python src/train_yolo.py --mode train --data data/yolo/dataset.yaml --epochs 50 --model-size n

# For better accuracy (takes longer):
python src/train_yolo.py --mode train --data data/yolo/dataset.yaml --epochs 100 --model-size s
```

### Step 3: Test YOLO Predictions

```powershell
# Predict on a new image
python src/train_yolo.py --mode predict --image "dev-test/stitched/s7.png"
```

### Step 4: Convert Labels for Mask R-CNN

```powershell
# Convert to COCO format for Mask R-CNN
python src/convert_labels.py --input data/labelstudio_export.json --output data/annotations --format coco --images data/stitched
```

### Step 5: Extract Patches for EfficientNet

```powershell
# Extract classification patches
python src/convert_labels.py --input data/labelstudio_export.json --output data/patches --format patches --images data/stitched
```

### Step 6: Train EfficientNet Classifier

```powershell
python src/train_effnet.py
```

### Step 7: Train Mask R-CNN

```powershell
python src/train_maskrcnn.py
```

---

## Recommended Labeling Strategy for 10 Samples

With only 10 samples, here's the optimal approach:

| Step | Samples | Purpose |
|------|---------|---------|
| 1 | Label 5 | Train YOLO (minimum viable) |
| 2 | Label 2 | Validation set |
| 3 | Label 3 | Test set (final evaluation) |

### Minimum Labels Needed:
- **YOLO**: 5-10 images with ~50 total objects
- **EfficientNet**: 30+ patches per class (extracted from detections)
- **Mask R-CNN**: 10+ images with polygon annotations

---

## Directory Structure After Setup

```
mp-detect/
├── data/
│   ├── labelstudio_export.json    # Your exported annotations
│   ├── yolo/                       # YOLO format data
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── labels/train/
│   │   ├── labels/val/
│   │   └── dataset.yaml
│   ├── annotations/                # COCO format for Mask R-CNN
│   │   └── annotations.json
│   └── patches/                    # Classification patches
│       ├── fiber/
│       ├── film/
│       └── fragment/
├── experiments/
│   ├── pipeline_test/              # Pretrained test results
│   ├── microplastic_yolo/          # YOLO training results
│   └── predictions/                # Inference outputs
└── src/
    ├── test_pipeline.py            # Test with pretrained models
    ├── train_yolo.py               # YOLO training
    ├── train_maskrcnn.py           # Mask R-CNN training
    ├── train_effnet.py             # EfficientNet training
    └── convert_labels.py           # Format converter
```

---

## Pipeline Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT: Macro Image (stitched filter paper photo)             │
│              │                                                 │
│              ▼                                                 │
│  ┌─────────────────────┐                                      │
│  │   YOLO Detection    │  Fast detection of microplastics     │
│  │   (Bounding Boxes)  │  Output: boxes + confidence          │
│  └──────────┬──────────┘                                      │
│              │                                                 │
│              ▼                                                 │
│  ┌─────────────────────┐                                      │
│  │   Mask R-CNN        │  Precise boundary segmentation       │
│  │   (Segmentation)    │  Output: pixel masks                 │
│  └──────────┬──────────┘                                      │
│              │                                                 │
│              ▼                                                 │
│  ┌─────────────────────┐                                      │
│  │   EfficientNet      │  Classify each detection             │
│  │   (Classification)  │  Output: fiber/film/fragment         │
│  └──────────┬──────────┘                                      │
│              │                                                 │
│              ▼                                                 │
│  OUTPUT: Microplastic Statistics                              │
│  - Count per class (fibers: X, films: Y, fragments: Z)        │
│  - Sizes and areas                                            │
│  - Visualization with masks                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Micro Images - Do You Need Them?

Based on your description:

| Use Case | Need to Label? | Need to Stitch? |
|----------|---------------|-----------------|
| **Training** | No | No |
| **Validation** | No | No |
| **Ground Truth Comparison** | Optional | No |

**Micro images are useful for:**
- Validating that macro detections are in correct positions
- Confirming classification (higher detail helps verify fiber vs film)
- Not needed for training if macro images have good quality

---

## Troubleshooting

### CUDA/GPU Issues
```powershell
# Check if CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Model Download Issues
Models download automatically on first run. Ensure internet connection.

### Memory Errors
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `--model-size n`

---

## Next Steps

1. ✅ Run `python src/test_pipeline.py --image "dev-test/stitched/s7.png" --model all`
2. ✅ Start Label Studio and create project
3. ✅ Label 5 images with bounding boxes
4. ✅ Export and convert labels
5. ✅ Train YOLO first (fastest feedback)
6. ⏳ Evaluate and iterate
