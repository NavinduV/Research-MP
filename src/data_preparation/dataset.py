import torch
from torch.utils.data import Dataset
import cv2
import json
import numpy as np

class MicroplasticDataset(Dataset):
    """
    Dataset for Mask R-CNN using COCO-style annotations.
    """

    def __init__(self, image_dir, annotation_file):
        with open(annotation_file) as f:
            self.coco = json.load(f)

        self.image_dir = image_dir
        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_path = f"{self.image_dir}/{img_info['file_name']}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0

        anns = [a for a in self.annotations if a["image_id"] == img_info["id"]]

        boxes = []
        masks = []
        labels = []

        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])

            mask = np.zeros((img_info["height"], img_info["width"]))
            for poly in ann["segmentation"]:
                pts = np.array(poly).reshape(-1, 2)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            masks.append(mask)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels),
            "masks": torch.tensor(masks, dtype=torch.uint8)
        }

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image, target
