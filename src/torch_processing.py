import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from pytorch_ood.dataset.img.ninco import NINCO
import argparse
from PIL import Image
import json

np.random.seed(42)


class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    
def get_args():
    parser = argparse.ArgumentParser(description="Feature Extraction from Datasets")
    parser.add_argument("--datasets", type=str, default='NINCO', help="Batch size for DataLoader")
    args = parser.parse_args()

    return args

def get_dataloader(dataset_name, batch_size=64, sample_size=10000):
    transform = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),  # Resize for MobileNetV3
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) if dataset_name == "FashionMNIST" else transforms.Lambda(lambda x: x),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if dataset_name == "FashionMNIST" else
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    elif dataset_name == "FashionMNIST":
        dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    elif dataset_name == "OxfordIIITPet":
        dataset = datasets.OxfordIIITPet(root="./data", split='trainval', target_types="category", transform=transform, download=True)
    elif dataset_name == "NINCO":
        ninco_class_root = os.path.join("./data", "NINCO/NINCO_OOD_classes")
        if not os.path.exists(ninco_class_root):
            raise FileNotFoundError(f"Dataset not found at {ninco_class_root}. Please download the dataset.")
        dataset = ImageFolder(root=ninco_class_root, transform=transform)
        label_to_name = {v: k for k, v in dataset.class_to_idx.items()}
        with open("ninco_label_to_name.json", "w") as f:
            json.dump(label_to_name, f)
        exit()
        #dataset = NINCO(root="./data", transform=transform, download=False)
    elif dataset_name == 'TinyImageNet':
        tiny_path = os.path.join("./data/TinyImageNet", "tiny-imagenet-200/train")
        if not os.path.exists(tiny_path):
            raise FileNotFoundError(f"Dataset not found at {tiny_path}. Please download the dataset.")
        dataset = ImageFolder(root=tiny_path, transform=transform)
        idx_to_wnid = {i: wnid for i, wnid in enumerate(dataset.classes)}
        wnid_to_label = dataset.class_to_idx  # e.g., {'n01443537': 0, ...}

        # Step 2: Reverse it: label_index -> wnid
        label_to_wnid = {v: k for k, v in wnid_to_label.items()}

        # Step 3: Load wnid -> human-readable name from words.txt
        wnid_to_name = {}
        with open("./data/TinyImageNet/tiny-imagenet-200/words.txt") as f:
            for line in f:
                wnid, name = line.strip().split('\t')
                wnid_to_name[wnid] = name

        # Step 4: Final mapping: label_index -> human name
        label_to_name = {
            label: wnid_to_name.get(label_to_wnid[label], "Unknown")
            for label in label_to_wnid
        }
        with open("tinyimagenet_label_to_name.json", "w") as f:
            json.dump(label_to_name, f)
    else:
        raise ValueError("Unsupported dataset")
    
    # Downsample dataset to 5000 samples
    print(dataset.classes)
    dataset = Subset(dataset, np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def extract_features(model, dataloader, device, dataset_name):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Processing {dataset_name}"):
            images, targets = images.to(device), targets.to(device)
            feats = model(images).cpu().numpy()
            features.append(feats)
            labels.append(targets.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    # Save as CSV
    #features_flat = features.reshape(features.shape[0], -1)  # Flatten features
    features = features.squeeze()
    df = pd.DataFrame(features)
    df['label'] = labels
    csv_path = f"features_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved features to {csv_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(pretrained=True)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove classifier
    feature_extractor = feature_extractor.to(device)
    # mobilenet = models.mobilenet_v3_small(pretrained=True)
    # feature_extractor = mobilenet.features  # Extract features directly
    # feature_extractor = feature_extractor.to(device)
    
    #for dataset_name in ["CIFAR10", "FashionMNIST", "OxfordIIITPet"]:
    for dataset_name in ["NINCO"]:
    #for dataset_name in ['TinyImageNet']:
        dataloader = get_dataloader(dataset_name)
        extract_features(feature_extractor, dataloader, device, dataset_name)

if __name__ == "__main__":
    main()
