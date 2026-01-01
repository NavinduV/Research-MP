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
          model_size: str = 'n', resume: bool = False):
    """
    Train YOLOv8 model on microplastic dataset.
    
    Args:
        data_yaml: Path to dataset.yaml configuration
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        model_size: YOLO model size (n, s, m, l, x)
        resume: Resume training from last checkpoint
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
    print(f"{'='*60}\n")
    
    # Load model (pretrained on COCO)
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='microplastic_yolo',
        project='experiments',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        patience=20,  # Early stopping patience
        save=True,
        plots=True,
        resume=resume
    )
    
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


def predict(model_path: str, image_path: str, save_dir: str = "experiments/predictions"):
    """
    Run inference on an image.
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to input image
        save_dir: Directory to save predictions
    """
    print(f"\nRunning inference on: {image_path}")
    
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        project=save_dir,
        name='inference',
        exist_ok=True
    )
    
    # Print detection summary
    for result in results:
        boxes = result.boxes
        print(f"\nDetected {len(boxes)} microplastics:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_names = ['fiber', 'film', 'fragment']
            print(f"  - {class_names[cls]}: {conf:.2f}")
    
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
    parser.add_argument('--resume', action='store_true', help='Resume training')
    
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
        train(args.data, args.epochs, args.imgsz, args.batch, args.model_size, args.resume)
        
    elif args.mode == 'val':
        model_path = args.model or 'experiments/microplastic_yolo/weights/best.pt'
        validate(model_path, args.data)
        
    elif args.mode == 'predict':
        model_path = args.model or 'experiments/microplastic_yolo/weights/best.pt'
        if not args.image:
            print("Error: --image required for predict mode")
            return
        predict(model_path, args.image)


if __name__ == "__main__":
    main()
