import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from dataset import MicroplasticDataset

def train():
    dataset = MicroplasticDataset(
        image_dir="data/patches",
        annotation_file="data/annotations/annotations.json"
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.roi_heads.box_predictor.cls_score.out_features = 4
    model.roi_heads.mask_predictor.mask_fcn_logits.out_channels = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

    torch.save(model.state_dict(), "experiments/maskrcnn.pth")


if __name__ == "__main__":
    train()
