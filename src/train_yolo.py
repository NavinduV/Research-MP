"""
Train YOLOv8 for microplastic detection.

This script trains a YOLO model on labeled microplastic data exported from Label Studio.
YOLO is used for fast object detection - finding bounding boxes around microplastics.

Classes:
    0: fiber
    1: film  
    2: fragment

Usage:
    python src/train_yolo.py --data data/yolo/dataset.yaml --epochs 100
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import os
import numpy as np
import cv2


def class_agnostic_nms(boxes, iou_threshold=0.3):
    """
    Class-agnostic Non-Maximum Suppression.
    
    Keeps only the highest-confidence detection per physical location,
    regardless of predicted class.  This ensures each real microplastic
    produces exactly ONE bounding box.
    
    Args:
        boxes: list/array  [[x1, y1, x2, y2, cls_id, conf], ...]
        iou_threshold: IoU above which the lower-confidence box is suppressed.
                       0.3 is aggressive enough to remove scale-variants.
    Returns:
        kept: list of boxes that survived NMS (same format as input rows)
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 5]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]          # highest confidence first

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # IoU of box i with every remaining box
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        # Also suppress if smaller box is mostly inside larger box
        # (containment ratio — catches scale-variant duplicates)
        min_area = np.minimum(areas[i], areas[order[1:]])
        containment = inter / np.maximum(min_area, 1e-6)

        # Suppress if IoU > threshold OR containment > 0.6
        inds = np.where((iou <= iou_threshold) & (containment <= 0.6))[0]
        order = order[inds + 1]

    return boxes[keep].tolist()


def merge_overlapping_boxes(boxes, iou_threshold=0.0):
    """
    Merge overlapping detection boxes into larger bounding boxes.
    
    Only merges boxes that have actual geometric overlap - this prevents
    the chain effect where distant boxes get merged together.
    
    Args:
        boxes: List of boxes in format [[x1, y1, x2, y2, cls_id, conf], ...]
        iou_threshold: IoU threshold above which boxes are considered overlapping
                       Use 0.0 to merge any boxes that overlap at all
        
    Returns:
        merged_boxes: List of merged bounding boxes [[x1, y1, x2, y2], ...]
        box_assignments: List mapping original box index to merged box index
    """
    if len(boxes) == 0:
        return [], []
    
    boxes = np.array(boxes)
    n_boxes = len(boxes)
    
    def boxes_overlap(box1, box2):
        """Check if two boxes have any overlap (intersection area > 0)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Check for actual overlap
        if x2 > x1 and y2 > y1:
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0
            return iou > iou_threshold
        return False
    
    # Build adjacency based ONLY on actual overlap
    adjacency = np.zeros((n_boxes, n_boxes), dtype=bool)
    for i in range(n_boxes):
        for j in range(i + 1, n_boxes):
            if boxes_overlap(boxes[i][:4], boxes[j][:4]):
                adjacency[i, j] = True
                adjacency[j, i] = True
    
    # Find connected components using iterative BFS (avoid recursion limit)
    visited = np.zeros(n_boxes, dtype=bool)
    clusters = []
    
    for start in range(n_boxes):
        if not visited[start]:
            cluster = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if not visited[node]:
                    visited[node] = True
                    cluster.append(node)
                    for neighbor in range(n_boxes):
                        if adjacency[node, neighbor] and not visited[neighbor]:
                            queue.append(neighbor)
            clusters.append(cluster)
    
    # Create merged bounding boxes for each cluster WITH class labels
    merged_boxes = []
    box_assignments = np.zeros(n_boxes, dtype=int)
    
    for cluster_idx, cluster in enumerate(clusters):
        cluster_boxes = boxes[cluster]
        
        # Get the bounding box that encompasses all boxes in the cluster
        x1 = np.min(cluster_boxes[:, 0])
        y1 = np.min(cluster_boxes[:, 1])
        x2 = np.max(cluster_boxes[:, 2])
        y2 = np.max(cluster_boxes[:, 3])
        
        # Determine dominant class in this cluster (most common class)
        # boxes format: [[x1, y1, x2, y2, cls_id, conf], ...]
        class_ids = cluster_boxes[:, 4].astype(int)
        class_counts = np.bincount(class_ids, minlength=3)
        dominant_class = np.argmax(class_counts)
        
        # Get max confidence in cluster
        max_conf = np.max(cluster_boxes[:, 5])
        
        # Add padding (10% on each side)
        width = x2 - x1
        height = y2 - y1
        padding_x = width * 0.1
        padding_y = height * 0.1
        
        merged_boxes.append({
            'x1': x1 - padding_x,
            'y1': y1 - padding_y, 
            'x2': x2 + padding_x,
            'y2': y2 + padding_y,
            'class_id': int(dominant_class),
            'confidence': float(max_conf),
            'num_detections': len(cluster)
        })
        
        for box_idx in cluster:
            box_assignments[box_idx] = cluster_idx
    
    return merged_boxes, box_assignments.tolist()


def create_dataset_yaml(data_dir: str, output_path: str):
    """
    Create YOLO dataset configuration file.
    
    Args:
        data_dir: Path to the YOLO format dataset directory
        output_path: Path to save the dataset.yaml file
    """
    dataset_config = {
        'path': os.path.abspath(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'fiber',
            1: 'film',
            2: 'fragment'
        },
        'nc': 3  # number of classes
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created dataset config: {output_path}")
    return output_path


def train(data_yaml: str, epochs: int = 100, imgsz: int = 640, batch: int = 16, 
          model_size: str = 'n', resume: bool = False, high_accuracy: bool = True, lr0: float = None):
    """
    Train YOLOv8 model on microplastic dataset with optimized settings for accuracy.
    
    Args:
        data_yaml: Path to dataset.yaml configuration
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        model_size: YOLO model size (n, s, m, l, x)
        resume: Resume training from last checkpoint
        high_accuracy: Use high accuracy settings (more augmentation, slower training)
        lr0: Initial learning rate (overrides default if provided)
    """
    # Select model size
    model_name = f'yolov8{model_size}.pt'
    
    print(f"\n{'='*60}")
    print("YOLO TRAINING - MICROPLASTIC DETECTION")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"High Accuracy Mode: {high_accuracy}")
    print(f"{'='*60}\n")
    
    # Load model (pretrained on COCO)
    model = YOLO(model_name)
    
    # Base training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'name': 'microplastic_yolo',
        'project': 'experiments',
        'exist_ok': True,
        'pretrained': True,
        'save': True,
        'plots': True,
        'resume': resume,
        'verbose': True,
    }
    
    if high_accuracy:
        # ============================================================
        # HIGH ACCURACY SETTINGS FOR SMALL DATASETS
        # ============================================================
        
        # Optimizer settings - lower LR for better fine-tuning
        train_args.update({
            'optimizer': 'AdamW',          # AdamW often better than Adam
            'lr0': 0.0005,                 # Lower initial LR for fine-tuning (default)
            'lrf': 0.01,                   # Final LR = lr0 * lrf
            'momentum': 0.937,             # SGD momentum / Adam beta1
            'weight_decay': 0.0005,        # L2 regularization
            'warmup_epochs': 5.0,          # Longer warmup for small datasets
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
        })
        
        # Override lr0 if provided via CLI
        if lr0 is not None:
            train_args['lr0'] = lr0
            print(f">>> Custom learning rate: lr0={lr0}")
        
        # Aggressive Data Augmentation - CRITICAL for small datasets
        train_args.update({
            # Mosaic & Mixup - creates synthetic training samples
            'mosaic': 1.0,                 # Mosaic augmentation probability
            'mixup': 0.2,                  # Mixup augmentation probability
            'copy_paste': 0.3,             # Copy-paste augmentation (great for detection)
            
            # Geometric augmentations
            'degrees': 15.0,               # Rotation (+/- degrees)
            'translate': 0.2,              # Translation (+/- fraction)
            'scale': 0.5,                  # Scale (+/- gain)
            'shear': 5.0,                  # Shear (+/- degrees)
            'perspective': 0.0005,         # Perspective (+/- fraction)
            'flipud': 0.5,                 # Vertical flip probability
            'fliplr': 0.5,                 # Horizontal flip probability
            
            # Color/HSV augmentations
            'hsv_h': 0.015,                # Hue augmentation
            'hsv_s': 0.7,                  # Saturation augmentation
            'hsv_v': 0.4,                  # Value/brightness augmentation
            
            # Other augmentations
            'erasing': 0.4,                # Random erasing probability
            'crop_fraction': 1.0,          # Classification crop fraction
        })
        
        # Training settings for better convergence
        train_args.update({
            'patience': 50,                # Longer patience for small datasets
            'cos_lr': True,                # Cosine LR scheduler (better than linear)
            'close_mosaic': 20,            # Disable mosaic for final epochs
            'amp': True,                   # Mixed precision training
            'fraction': 1.0,               # Use all data
            'cache': True,                 # Cache images in RAM for faster training
            'workers': 4,                  # DataLoader workers
            'seed': 42,                    # Reproducibility
        })
        
        # Detection-specific settings
        train_args.update({
            'box': 7.5,                    # Box loss gain
            'cls': 0.5,                    # Classification loss gain
            'dfl': 1.5,                    # Distribution focal loss gain
            'label_smoothing': 0.1,        # Label smoothing for regularization
            'nbs': 64,                     # Nominal batch size for autoscaling
            'overlap_mask': True,          # Masks overlap during training
            'mask_ratio': 4,               # Mask downsample ratio
            'dropout': 0.1,                # Dropout for regularization (small datasets)
        })
        
        print(">>> High Accuracy Mode Enabled:")
        print("    - Aggressive data augmentation (mosaic, mixup, copy-paste)")
        print("    - Lower learning rate with cosine scheduler")
        print("    - Extended patience and warmup")
        print("    - Label smoothing and dropout regularization")
        print()
    else:
        # Standard training settings
        train_args.update({
            'optimizer': 'Adam',
            'lr0': 0.001,
            'patience': 20,
        })
    
    # Train the model
    results = model.train(**train_args)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best model saved to: experiments/microplastic_yolo/weights/best.pt")
    print(f"{'='*60}")
    
    return results


def validate(model_path: str, data_yaml: str):
    """
    Validate trained model on test set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset.yaml
    """
    print(f"\nValidating model: {model_path}")
    
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return metrics


def predict(model_path: str, image_path: str, save_dir: str = "experiments/predictions-max-accuracy", 
            conf: float = 0.1, filter_only: bool = True, effnet_path: str = None):
    """
    Run inference on an image.
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to input image
        save_dir: Directory to save predictions
        conf: Confidence threshold (default 0.1 for undertrained models)
        filter_only: If True, only show detections on the filter paper (not background)
        effnet_path: Path to EfficientNet checkpoint for classification refinement.
                     If provided, each YOLO crop is reclassified by EfficientNet for
                     better accuracy.
    """
    print(f"\nRunning inference on: {image_path}")
    print(f"Confidence threshold: {conf}")
    print(f"Filter-only mode: {filter_only}")
    
    # ------------------------------------------------------------------
    # Load EfficientNet for classification refinement (if requested)
    # ------------------------------------------------------------------
    effnet_model = None
    effnet_transform = None
    if effnet_path and Path(effnet_path).exists():
        import timm
        import torch
        from torchvision import transforms
        from PIL import Image as PILImage
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        effnet_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=3)
        ckpt = torch.load(effnet_path, map_location=device)
        effnet_model.load_state_dict(ckpt['model_state_dict'])
        effnet_model.to(device)
        effnet_model.eval()
        
        effnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        print(f"EfficientNet loaded from: {effnet_path}  (device={device})")
    elif effnet_path:
        print(f"WARNING: EfficientNet checkpoint not found: {effnet_path}")
    
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        save=not filter_only,  # Don't save if we need to filter
        save_txt=True,
        project=save_dir,
        name='inference',
        exist_ok=True,
        conf=conf
    )
    
    class_names = ['fiber', 'film', 'fragment']
    
    # Process detections and draw results
    if len(results) > 0:
        import cv2
        import numpy as np
        
        # Process each result with its corresponding image
        for result in results:
            # Get the actual image path from the result
            actual_image_path = result.path
            
            # Load image to get dimensions and detect filter
            img = cv2.imread(actual_image_path)
            if img is None:
                print(f"Warning: Could not read image {actual_image_path}, skipping...")
                continue
                
            height, width = img.shape[:2]
            min_dim = min(height, width)
            
            # ============================================================
            # FILTER DETECTION (only needed if filter_only=True)
            # ============================================================
            cx, cy, radius = None, None, None
            detection_method = None
            
            if filter_only:
                skip_filtering = False
                
                # --- Strategy 0: Check if filter fills entire image ---
                # If there's minimal dark background, the filter covers the whole image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Check for dark pixels (background)
                dark_threshold = 100
                dark_pixels = np.sum(gray < dark_threshold)
                dark_ratio = dark_pixels / (height * width)
                
                # Also check the edges of the image for background
                edge_margin = 20
                top_edge = gray[:edge_margin, :].mean()
                bottom_edge = gray[-edge_margin:, :].mean()
                left_edge = gray[:, :edge_margin].mean()
                right_edge = gray[:, -edge_margin:].mean()
                edge_brightness = np.mean([top_edge, bottom_edge, left_edge, right_edge])
                
                # If less than 5% dark pixels AND edges are bright, filter fills image
                if dark_ratio < 0.05 and edge_brightness > 150:
                    skip_filtering = True
                    detection_method = "full-coverage"
                    print(f"[{Path(actual_image_path).name}] Filter fills entire image (dark_ratio={dark_ratio:.2%}, edge_brightness={edge_brightness:.0f}) - including all detections")
                
                # --- Strategy 1: Edge-based Hough Circle Detection ---
                # This works best for circular filter papers regardless of staining
                if not skip_filtering:
                    # Apply bilateral filter to reduce noise while keeping edges
                    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
                    
                    # Apply Canny edge detection
                    edges = cv2.Canny(blurred, 30, 100)
                    
                    # Dilate edges to connect broken circles
                    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    edges = cv2.dilate(edges, kernel_dilate, iterations=1)
                    
                    # Try multiple parameter combinations for Hough Circles
                    param_sets = [
                        {'dp': 1.2, 'param1': 50, 'param2': 30, 'minRadius': int(min_dim * 0.30), 'maxRadius': int(min_dim * 0.50)},
                        {'dp': 1.5, 'param1': 80, 'param2': 40, 'minRadius': int(min_dim * 0.25), 'maxRadius': int(min_dim * 0.55)},
                        {'dp': 1.0, 'param1': 100, 'param2': 25, 'minRadius': int(min_dim * 0.35), 'maxRadius': int(min_dim * 0.48)},
                        {'dp': 2.0, 'param1': 50, 'param2': 50, 'minRadius': int(min_dim * 0.30), 'maxRadius': int(min_dim * 0.52)},
                    ]
                    
                    best_circle = None
                    for params in param_sets:
                        circles = cv2.HoughCircles(
                            blurred, cv2.HOUGH_GRADIENT,
                            dp=params['dp'],
                            minDist=min_dim // 2,
                            param1=params['param1'],
                            param2=params['param2'],
                            minRadius=params['minRadius'],
                            maxRadius=params['maxRadius']
                        )
                        
                        if circles is not None:
                            circles = np.round(circles[0, :]).astype("int")
                            # Find the circle closest to the image center with largest radius
                            img_cx, img_cy = width // 2, height // 2
                            for c in circles:
                                dist_to_center = np.sqrt((c[0] - img_cx)**2 + (c[1] - img_cy)**2)
                                # Prefer circles near center and large
                                if dist_to_center < min_dim * 0.3:
                                    if best_circle is None or c[2] > best_circle[2]:
                                        best_circle = c
                            
                            if best_circle is not None:
                                break
                    
                    if best_circle is not None:
                        cx, cy, radius = best_circle
                        detection_method = "edge-Hough"
                    
                    # --- Strategy 2: Background detection (invert to find filter) ---
                    if cx is None:
                        # Detect dark wooden background (low value, brownish hue)
                        # Dark areas: V < 100
                        lower_dark = np.array([0, 0, 0])
                        upper_dark = np.array([180, 255, 120])
                        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
                        
                        # Also detect the characteristic brown/wood color
                        lower_wood = np.array([5, 30, 20])
                        upper_wood = np.array([25, 200, 150])
                        wood_mask = cv2.inRange(hsv, lower_wood, upper_wood)
                        
                        # Combine background masks
                        bg_mask = cv2.bitwise_or(dark_mask, wood_mask)
                        
                        # Invert to get filter paper region
                        filter_mask = cv2.bitwise_not(bg_mask)
                        
                        # Clean up the mask
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                        filter_mask = cv2.morphologyEx(filter_mask, cv2.MORPH_CLOSE, kernel)
                        filter_mask = cv2.morphologyEx(filter_mask, cv2.MORPH_OPEN, kernel)
                        
                        # Fill holes in the filter region
                        contours, _ = cv2.findContours(filter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            # Find the largest contour (should be the filter paper)
                            largest_contour = max(contours, key=cv2.contourArea)
                            contour_area = cv2.contourArea(largest_contour)
                            
                            # Must be at least 10% of image area to be valid
                            if contour_area > (height * width * 0.10):
                                # Fit minimum enclosing circle
                                (cx_float, cy_float), radius_float = cv2.minEnclosingCircle(largest_contour)
                                cx, cy, radius = int(cx_float), int(cy_float), int(radius_float)
                                
                                # Verify circularity
                                circle_area = np.pi * radius * radius
                                circularity = contour_area / circle_area if circle_area > 0 else 0
                                
                                if circularity > 0.6 and radius > min_dim * 0.20:
                                    detection_method = "background-invert"
                                else:
                                    cx, cy, radius = None, None, None
                    
                    # --- Strategy 3: Default fallback ---
                    if cx is None:
                        # Use image center with reasonable radius
                        cx, cy = width // 2, height // 2
                        radius = int(min_dim * 0.42)
                        detection_method = "default-center"
                    
                    # Add small margin inward to avoid edge artifacts
                    radius = int(radius * 0.98)
                    
                    print(f"[{Path(actual_image_path).name}] Filter: center=({cx}, {cy}), r={radius}, method={detection_method}")
            else:
                # Not filtering - set dummy values
                skip_filtering = True
            
            # Filter boxes to those inside the circle (or include all if not filtering)
            filtered_indices = []
            
            if not filter_only or skip_filtering:
                # Include all detections
                filtered_indices = list(range(len(result.boxes)))
            else:
                for i, box in enumerate(result.boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    box_cx = (xyxy[0] + xyxy[2]) / 2
                    box_cy = (xyxy[1] + xyxy[3]) / 2
                    
                    # Check if box center is inside the filter circle
                    dist = np.sqrt((box_cx - cx)**2 + (box_cy - cy)**2)
                    if dist <= radius:
                        filtered_indices.append(i)
            
            
            # Draw detections
            output_img = img.copy()
            
            # Draw filter circle boundary only if we detected one
            if not skip_filtering and cx is not None:
                cv2.circle(output_img, (cx, cy), radius, (128, 128, 128), 2)
            
            # Collect all filtered boxes
            raw_filtered = []
            for i in filtered_indices:
                box = result.boxes[i]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                raw_filtered.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], cls_id, conf_val])
            
            # ============================================================
            # CLASS-AGNOSTIC NMS — one box per physical microplastic
            # ============================================================
            # IoU 0.3 is aggressive: if two boxes overlap >30% the weaker
            # one is removed.  Also suppresses containment (small box
            # inside a larger one covering the same MP).
            nms_boxes = class_agnostic_nms(raw_filtered, iou_threshold=0.3)
            
            filtered_count = len(nms_boxes)
            print(f"[{Path(actual_image_path).name}] NMS: {len(raw_filtered)} raw -> {filtered_count} unique MPs")
            
            # ============================================================
            # EfficientNet reclassification of each NMS crop
            # ============================================================
            if effnet_model is not None and len(nms_boxes) > 0:
                import torch
                from PIL import Image as PILImage
                
                device = next(effnet_model.parameters()).device
                reclassified = 0
                for idx, bx in enumerate(nms_boxes):
                    x1c, y1c = max(0, int(bx[0]) - 10), max(0, int(bx[1]) - 10)
                    x2c, y2c = min(width, int(bx[2]) + 10), min(height, int(bx[3]) + 10)
                    crop_bgr = img[y1c:y2c, x1c:x2c]
                    if crop_bgr.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    pil_crop = PILImage.fromarray(crop_rgb)
                    inp = effnet_transform(pil_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = effnet_model(inp)
                        probs = torch.softmax(out, dim=1)[0]
                        pred_cls = probs.argmax().item()
                        pred_conf = probs[pred_cls].item()
                    
                    old_cls = int(bx[4])
                    # Store EfficientNet result in extra columns
                    # bx: [x1, y1, x2, y2, yolo_cls, yolo_conf, effnet_cls, effnet_conf]
                    bx.extend([pred_cls, pred_conf])
                    if pred_cls != old_cls:
                        reclassified += 1
                
                print(f"[{Path(actual_image_path).name}] EfficientNet reclassified {reclassified}/{filtered_count} detections")
            
            # Draw surviving detections
            det_colors = [(255, 0, 0), (0, 255, 255), (0, 255, 0)]  # fiber=Blue, film=Cyan, fragment=Green
            for bx in nms_boxes:
                x1b, y1b, x2b, y2b = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
                yolo_cls = int(bx[4])
                yolo_conf = float(bx[5])
                
                # Use EfficientNet class if available, else YOLO
                if len(bx) >= 8:
                    cls_id = int(bx[6])
                    effnet_conf = float(bx[7])
                    color = det_colors[cls_id]
                    cv2.rectangle(output_img, (x1b, y1b), (x2b, y2b), color, 2)
                    label = f"{class_names[cls_id]} {effnet_conf:.2f}"
                    # Smaller YOLO tag below
                    yolo_tag = f"YOLO: {class_names[yolo_cls]} {yolo_conf:.2f}"
                    cv2.putText(output_img, label, (x1b, y1b - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(output_img, yolo_tag, (x1b, y2b + 18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cls_id = yolo_cls
                    conf_val = yolo_conf
                    color = det_colors[cls_id]
                    cv2.rectangle(output_img, (x1b, y1b), (x2b, y2b), color, 2)
                    label = f"{class_names[cls_id]} {conf_val:.2f}"
                    cv2.putText(output_img, label, (x1b, y1b - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # ============================================================
            # Save one crop per NMS-surviving detection for Mask R-CNN
            # ============================================================
            if len(nms_boxes) > 0:
                crops_dir = Path(save_dir) / 'crops'
                (crops_dir / 'images').mkdir(parents=True, exist_ok=True)
                for cls_name in class_names:
                    (crops_dir / cls_name).mkdir(parents=True, exist_ok=True)
                
                # Load existing annotations (accumulate across images)
                ann_file = crops_dir / 'annotations.json'
                if ann_file.exists():
                    import json as _json
                    with open(ann_file) as _f:
                        crop_annotations = _json.load(_f)
                else:
                    crop_annotations = {}
                
                crop_padding = 15  # pixels of context around the detection
                for i, bx in enumerate(nms_boxes):
                    det_x1, det_y1 = int(bx[0]), int(bx[1])
                    det_x2, det_y2 = int(bx[2]), int(bx[3])
                    yolo_cls_id = int(bx[4])
                    yolo_conf_val = float(bx[5])
                    
                    # Use EfficientNet class if available
                    if len(bx) >= 8:
                        final_cls_id = int(bx[6])
                        final_conf = float(bx[7])
                    else:
                        final_cls_id = yolo_cls_id
                        final_conf = yolo_conf_val
                    
                    cls_name = class_names[final_cls_id]
                    
                    # Crop with padding, clipped to image bounds
                    x1 = max(0, det_x1 - crop_padding)
                    y1 = max(0, det_y1 - crop_padding)
                    x2 = min(width, det_x2 + crop_padding)
                    y2 = min(height, det_y2 + crop_padding)
                    
                    crop_img = img[y1:y2, x1:x2]
                    if crop_img.size == 0:
                        continue
                    
                    crop_filename = f"{Path(actual_image_path).stem}_crop{i+1}_{cls_name}.png"
                    cv2.imwrite(str(crops_dir / cls_name / crop_filename), crop_img)
                    cv2.imwrite(str(crops_dir / 'images' / crop_filename), crop_img)
                    
                    # Relative box within the crop
                    rel_x1 = det_x1 - x1
                    rel_y1 = det_y1 - y1
                    rel_x2 = det_x2 - x1
                    rel_y2 = det_y2 - y1
                    
                    ann_entry = {
                        'source_image': Path(actual_image_path).name,
                        'class_id': final_cls_id,
                        'class_name': cls_name,
                        'yolo_class_id': yolo_cls_id,
                        'yolo_class_name': class_names[yolo_cls_id],
                        'yolo_confidence': yolo_conf_val,
                        'rel_box': [rel_x1, rel_y1, rel_x2, rel_y2],
                        'crop_size': [crop_img.shape[1], crop_img.shape[0]]
                    }
                    if len(bx) >= 8:
                        ann_entry['effnet_class_id'] = int(bx[6])
                        ann_entry['effnet_class_name'] = class_names[int(bx[6])]
                        ann_entry['effnet_confidence'] = float(bx[7])
                    
                    crop_annotations[crop_filename] = ann_entry
                
                import json as _json
                with open(ann_file, 'w') as _f:
                    _json.dump(crop_annotations, _f, indent=2)
                
                print(f"[{Path(actual_image_path).name}] Saved {len(nms_boxes)} crops for Mask R-CNN")
                print(f"[{Path(actual_image_path).name}] Crops saved to: {crops_dir}")
                print(f"[{Path(actual_image_path).name}] annotations.json updated ({len(crop_annotations)} total crops)")
            
            # Save result
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output_path = Path(save_dir) / 'inference' / f"{Path(actual_image_path).stem}_filtered.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), output_img)
            
            if skip_filtering:
                print(f"[{Path(actual_image_path).name}] All detections: {filtered_count}, Saved: {output_path.name}")
            else:
                print(f"[{Path(actual_image_path).name}] Filtered: {filtered_count}, Removed: {len(result.boxes) - filtered_count}, Saved: {output_path.name}")
            
            # Print class-wise summary
            from collections import Counter
            yolo_counts = Counter()
            final_counts = Counter()
            for bx in nms_boxes:
                yolo_counts[class_names[int(bx[4])]] += 1
                if len(bx) >= 8:
                    final_counts[class_names[int(bx[6])]] += 1
                else:
                    final_counts[class_names[int(bx[4])]] += 1
            
            print(f"\n  Class breakdown (final): {dict(final_counts)}")
            if effnet_model is not None:
                print(f"  YOLO original:          {dict(yolo_counts)}")
                changes = sum(1 for bx in nms_boxes if len(bx) >= 8 and int(bx[4]) != int(bx[6]))
                print(f"  EfficientNet changed: {changes}/{filtered_count} classifications")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO for microplastic detection')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'predict', 'setup'], 
                        default='train', help='Mode: train, val, predict, or setup')
    parser.add_argument('--data', type=str, default='data/yolo/dataset.yaml',
                        help='Path to dataset.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--model-size', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--model', type=str, help='Path to trained model (for val/predict)')
    parser.add_argument('--image', type=str, help='Image path for prediction')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold for predictions')
    parser.add_argument('--no-filter', action='store_true', help='Disable filter-only mode (detect on entire image)')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--fast', action='store_true', help='Use fast training mode (less augmentation)')
    parser.add_argument('--lr0', type=float, default=None, help='Initial learning rate (overrides default)')
    parser.add_argument('--effnet', type=str, default=None,
                        help='Path to EfficientNet checkpoint for classification refinement in predict mode')
    
    args = parser.parse_args()
    
    if args.mode == 'setup':
        # Create dataset structure
        data_dir = Path('data/yolo')
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (data_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (data_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (data_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (data_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        (data_dir / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        
        create_dataset_yaml('data/yolo', 'data/yolo/dataset.yaml')
        print("\nDataset structure created! Now export from Label Studio in YOLO format.")
        
    elif args.mode == 'train':
        if not Path(args.data).exists():
            print(f"Error: Dataset config not found: {args.data}")
            print("Run with --mode setup first, then export data from Label Studio")
            return
        train(args.data, args.epochs, args.imgsz, args.batch, args.model_size, args.resume, high_accuracy=not args.fast, lr0=args.lr0)
        
    elif args.mode == 'val':
        model_path = args.model or 'experiments/microplastic_yolo/weights/best.pt'
        validate(model_path, args.data)
        
    elif args.mode == 'predict':
        model_path = args.model or 'experiments/microplastic_yolo/weights/best.pt'
        if not args.image:
            print("Error: --image required for predict mode")
            return
        predict(model_path, args.image, conf=args.conf, filter_only=not args.no_filter,
               effnet_path=args.effnet)


if __name__ == "__main__":
    main()
