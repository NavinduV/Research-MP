import torch
import timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class ParticleDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for cls in os.listdir(root):
            for f in os.listdir(os.path.join(root, cls)):
                self.samples.append((os.path.join(root, cls, f), int(cls)))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label


def train():
    dataset = ParticleDataset("data/particles")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

    torch.save(model.state_dict(), "experiments/efficientnet.pth")


if __name__ == "__main__":
    train()
