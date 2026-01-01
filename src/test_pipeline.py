"""
Test microplastic detection pipeline using pretrained models.
No custom training data required - validates pipeline architecture.

Usage:
    python src/test_pipeline.py --image "dev-test/stitched/s7.png" --model all
    python src/test_pipeline.py --image "dev-test/stitched/s7.png" --model yolo
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import os

# Create output directory
OUTPUT_DIR = Path("experiments/pipeline_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PipelineTester:
    """Test detection pipeline with pretrained models."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pretrained models
        self.yolo_model = None
        self.maskrcnn_model = None
        self.effnet_model = None
        
    def load_yolo(self):
        """Load pretrained YOLOv8 model."""
        print("Loading YOLOv8 pretrained model...")
        from ultralytics import YOLO
        self.yolo_model = YOLO('yolov8n.pt')  # nano version for speed
        print("✓ YOLOv8 loaded successfully!")
        return self.yolo_model
    
    def load_maskrcnn(self):
        """Load pretrained Mask R-CNN model."""
        print("Loading Mask R-CNN pretrained model...")
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.maskrcnn_model = maskrcnn_resnet50_fpn(weights=weights)
        self.maskrcnn_model.to(self.device)
        self.maskrcnn_model.eval()
        print("✓ Mask R-CNN loaded successfully!")
        return self.maskrcnn_model
    
    def load_efficientnet(self):
        """Load pretrained EfficientNet model."""
        print("Loading EfficientNet-B0 pretrained model...")
        weights = EfficientNet_B0_Weights.DEFAULT
        self.effnet_model = efficientnet_b0(weights=weights)
        self.effnet_model.to(self.device)
        self.effnet_model.eval()
        print("✓ EfficientNet loaded successfully!")
        return self.effnet_model
    
    def test_yolo(self, image_path: str, save_output: bool = True):
        """
        Test YOLO detection on an image.
        Pretrained YOLO won't detect microplastics specifically,
        but will show if the pipeline works.
        """
        if self.yolo_model is None:
            self.load_yolo()
        
        print(f"\n--- Testing YOLO on: {image_path} ---")
        results = self.yolo_model(image_path)
        
        # Get detection info
        for result in results:
            boxes = result.boxes
            print(f"  Detected {len(boxes)} objects")
            
            if save_output:
                output_path = OUTPUT_DIR / (Path(image_path).stem + "_yolo_result.jpg")
                result.save(filename=str(output_path))
                print(f"  Saved visualization to: {output_path}")
        
        return results
    
    def test_maskrcnn(self, image_path: str, confidence_threshold: float = 0.5):
        """
        Test Mask R-CNN segmentation on an image.
        Returns masks that could potentially capture microplastic shapes.
        """
        if self.maskrcnn_model is None:
            self.load_maskrcnn()
        
        print(f"\n--- Testing Mask R-CNN on: {image_path} ---")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.maskrcnn_model(image_tensor)
        
        pred = predictions[0]
        masks = pred['masks']
        scores = pred['scores']
        boxes = pred['boxes']
        
        # Filter by confidence
        high_conf_mask = scores > confidence_threshold
        filtered_masks = masks[high_conf_mask]
        filtered_scores = scores[high_conf_mask]
        filtered_boxes = boxes[high_conf_mask]
        
        print(f"  Detected {len(filtered_masks)} objects (confidence > {confidence_threshold})")
        
        # Visualize masks
        if len(filtered_masks) > 0:
            self._visualize_masks(image_path, filtered_masks, filtered_scores, filtered_boxes)
        else:
            print("  No high-confidence detections (this is expected with pretrained model)")
        
        return predictions
    
    def _visualize_masks(self, image_path: str, masks, scores, boxes):
        """Visualize Mask R-CNN output."""
        image = cv2.imread(image_path)
        overlay = image.copy()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (mask, score, box) in enumerate(zip(masks[:5], scores[:5], boxes[:5])):
            mask_np = mask[0].cpu().numpy()
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            color = colors[i % len(colors)]
            overlay[mask_binary == 1] = color
            
            # Draw bounding box
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{score:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        output_path = OUTPUT_DIR / (Path(image_path).stem + "_maskrcnn_result.jpg")
        cv2.imwrite(str(output_path), result)
        print(f"  Saved visualization to: {output_path}")
    
    def test_efficientnet_features(self, image_path: str):
        """
        Test EfficientNet feature extraction.
        Shows how the model extracts features that could classify microplastic types.
        """
        if self.effnet_model is None:
            self.load_efficientnet()
        
        print(f"\n--- Testing EfficientNet on: {image_path} ---")
        
        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(self.device)
        
        # Get predictions (ImageNet classes)
        with torch.no_grad():
            outputs = self.effnet_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        print("  Top 5 ImageNet predictions (won't match microplastics, shows model works):")
        weights = EfficientNet_B0_Weights.DEFAULT
        categories = weights.meta["categories"]
        for i in range(5):
            print(f"    {categories[top5_catid[i]]}: {top5_prob[i].item():.3f}")
        
        return outputs
    
    def run_full_pipeline_test(self, image_path: str):
        """
        Run complete pipeline test on a single image.
        Demonstrates the flow: Detection -> Segmentation -> Classification
        """
        print("\n" + "="*60)
        print("FULL PIPELINE TEST - PRETRAINED MODELS")
        print("="*60)
        print(f"Image: {image_path}")
        print("NOTE: Pretrained models won't detect microplastics specifically")
        print("This validates the pipeline architecture works correctly")
        print("="*60)
        
        # Step 1: YOLO Detection
        print("\n[STEP 1/3] YOLO Object Detection")
        yolo_results = self.test_yolo(image_path)
        
        # Step 2: Mask R-CNN Segmentation
        print("\n[STEP 2/3] Mask R-CNN Instance Segmentation")
        maskrcnn_results = self.test_maskrcnn(image_path, confidence_threshold=0.3)
        
        # Step 3: EfficientNet Classification
        print("\n[STEP 3/3] EfficientNet Classification")
        effnet_results = self.test_efficientnet_features(image_path)
        
        print("\n" + "="*60)
        print("PIPELINE TEST COMPLETE")
        print(f"Results saved in: {OUTPUT_DIR}")
        print("="*60)
        
        return {
            'yolo': yolo_results,
            'maskrcnn': maskrcnn_results,
            'efficientnet': effnet_results
        }


def main():
    """Main function to test the pipeline."""
    parser = argparse.ArgumentParser(description='Test microplastic detection pipeline')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to test image (e.g., dev-test/stitched/s7.png)')
    parser.add_argument('--model', type=str, choices=['yolo', 'maskrcnn', 'effnet', 'all'], 
                        default='all', help='Which model to test')
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    tester = PipelineTester()
    
    if args.model == 'yolo':
        tester.test_yolo(args.image)
    elif args.model == 'maskrcnn':
        tester.test_maskrcnn(args.image)
    elif args.model == 'effnet':
        tester.test_efficientnet_features(args.image)
    else:
        tester.run_full_pipeline_test(args.image)


if __name__ == "__main__":
    main()
