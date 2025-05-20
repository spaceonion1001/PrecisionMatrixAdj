import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

torch.manual_seed(42)  # For reproducibility

class MNISTCNN(nn.Module):
    def __init__(self, feature_dim=256, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # [B, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [B, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # [B, 64, 7, 7]
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, feature_dim)  # Penultimate layer
        self.fc2 = nn.Linear(feature_dim, num_classes) # Final classifier layer

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        features = self.fc1(x)         # 256-dim penultimate representation
        logits = self.fc2(features)    # Class scores
        return logits, features        # Return both if needed
    
def train_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Model
    model = MNISTCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch=epoch, num_epochs=num_epochs)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Test Loss={test_loss:.4f}, Acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "./models/mnist_cnn_best.pth")
            print("✅ Saved best model")

    print(f"🏁 Final Test Accuracy: {best_acc:.4f}")

# Training loop
def train(model, loader, optimizer, criterion, device, epoch=1, num_epochs=20):
    model.train()
    total_loss, correct = 0, 0
    for x, y in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy
    