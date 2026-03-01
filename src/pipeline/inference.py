"""Lightweight inference utilities.

Delay heavy ML imports until functions are called so the ASGI app can be
imported without having ML dependencies installed at startup. If a required
package is missing when you actually run inference, a clear RuntimeError is
raised.
"""

def load_models():
    try:
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        import timm
    except Exception as e:
        raise RuntimeError(
            "Required ML packages are not installed. Install `torch`, `torchvision`, and `timm` to load models: "
            + str(e)
        )

    seg_model = maskrcnn_resnet50_fpn(num_classes=4)
    seg_model.load_state_dict(torch.load("experiments/maskrcnn.pth"))
    seg_model.eval()

    cls_model = timm.create_model("efficientnet_b0", num_classes=3)
    cls_model.load_state_dict(torch.load("experiments/efficientnet.pth"))
    cls_model.eval()

    return seg_model, cls_model


def analyze_image(image_path):
    try:
        import cv2
        import numpy as np
        import torch
    except Exception as e:
        raise RuntimeError(
            "Required runtime packages are not installed. Install `opencv-python`, `numpy`, and `torch`: "
            + str(e)
        )

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)

    seg_model, cls_model = load_models()

    with torch.no_grad():
        output = seg_model(tensor)[0]

    results = {"fiber": 0, "fragment": 0, "film": 0}
    total_area = image.shape[0] * image.shape[1]

    for mask in output["masks"]:
        area = mask.sum().item()
        ratio = area / total_area

        if ratio < 0.001:
            results["fiber"] += 1
        elif ratio < 0.01:
            results["fragment"] += 1
        else:
            results["film"] += 1

    return results


if __name__ == "__main__":
    try:
        print(analyze_image("data/stitched/stitched_filter.png"))
    except Exception as e:
        print("Error running analysis:", e)
