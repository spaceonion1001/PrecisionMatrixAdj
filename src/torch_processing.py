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
# from pytorch_ood.dataset.img.ninco import NINCO
import argparse
from PIL import Image
import json

from msl import MSLLabeledDataset, get_finetuned_resnet18, get_optimizer, train_model, MSLLabeledDatasetFilteredRemapped, CustomAlexNet, get_alexnet_optimizer, get_finetuned_resnet50, MSLLabeledDatasetAllSplits, train_model
from mnist import train_mnist, MNISTCNN

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
    if dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),  # repeat channels to get 3
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                std=[0.229, 0.224, 0.225])   # ImageNet std
        ])
    else:
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
    elif dataset_name == 'MSL':
        msl_path = "./data/msl-labeled-data-set-v2.1"
        #dataset = MSLLabeledDataset(msl_path, split='val')
        dataset = MSLLabeledDatasetAllSplits(msl_path)
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    elif dataset_name == 'KMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.KMNIST(root="./data", train=True, transform=transform, download=True)
    elif dataset_name == 'SVHN':
        dataset = datasets.SVHN(root="./data", split='train', transform=transform, download=True)
    else:
        raise ValueError("Unsupported dataset")
    
    # Downsample dataset if necessary
    # print(dataset.classes)
    if dataset_name == 'MSL':
        print(dataset.class_to_idx)
    
    dataset = Subset(dataset, np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def extract_features(model, dataloader, device, dataset_name):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Processing {dataset_name}"):
            images, targets = images.to(device), targets.to(device)
            if dataset_name in ['MNIST', 'KMNIST']:
                _, feats = model(images)
                feats = feats.cpu().numpy()
            else:
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
    
    # for dataset_name in ["CIFAR10", "FashionMNIST"]:
    for dataset_name in ["FashionMNIST"]:
    #for dataset_name in ["NINCO"]:
    #for dataset_name in ['TinyImageNet']:
    #for dataset_name in ['MSL']:
    #for dataset_name in ['MNIST', 'KMNIST']:
    # for dataset_name in ['SVHN']:
        dataloader = get_dataloader(dataset_name, sample_size=np.inf)
        if dataset_name == 'MSL':
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Rebuild the model with the correct number of classes
            # model = get_finetuned_resnet18(num_classes=len(dataloader.dataset.dataset.classes) - 1) # we threw out training on artifact class
            #model = get_finetuned_resnet18(num_classes=len(dataloader.dataset.dataset.classes)) # or maybe we didn't...
            model = get_finetuned_resnet50(num_classes=len(dataloader.dataset.dataset.classes) - 1) # or maybe we did....
            print("Loading Pretrained Model...")
            model.load_state_dict(torch.load('./models/best_model_mslv2.pth', map_location=device))
            old_fc = model.fc
            # Keep only the dropout and first linear layer (trained)
            model.fc = nn.Sequential(
                old_fc[0],  # Dropout(p=0.5)
                old_fc[1]   # Linear(2048 -> 512) with pretrained weights
            )
            #model = nn.Sequential(*list(model.children())[:-1])
            model.to(device)
            model.eval()
            extract_features(model, dataloader, device, dataset_name)
        elif dataset_name == 'MNIST' or dataset_name == 'KMNIST':
            print("Loading Pretrained Model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MNISTCNN().to(device)
            model.load_state_dict(torch.load("./models/mnist_cnn_best.pth", map_location=device))
            model.eval()
            extract_features(model, dataloader, device, dataset_name)
        else:
            extract_features(feature_extractor, dataloader, device, dataset_name)

def main_train_msl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    msl_path = "./data/msl-labeled-data-set-v2.1"
    #train_dataset = MSLLabeledDatasetFilteredRemapped(msl_path, split='train', exclude_labels={2})
    #val_dataset = MSLLabeledDatasetFilteredRemapped(msl_path, split='val', exclude_labels={2})
    #test_dataset = MSLLabeledDatasetFilteredRemapped(msl_path, split='test', exclude_labels={2})
    train_dataset = MSLLabeledDatasetAllSplits(msl_path, exclude_labels={2})
    val_dataset = MSLLabeledDatasetAllSplits(msl_path, exclude_labels={2})
    test_dataset = MSLLabeledDatasetAllSplits(msl_path, exclude_labels={2})
    # train_dataset = MSLLabeledDataset(msl_path, split='train')
    # val_dataset = MSLLabeledDataset(msl_path, split='val')
    # test_dataset = MSLLabeledDataset(msl_path, split='test')
    
    print("Train Size {} Val Size {} Test Size {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    print(train_dataset.class_to_idx)
    # Assuming you've already created train_dataset and test_dataset
    #model = get_finetuned_resnet18(num_classes=len(train_dataset.classes))
    model = get_finetuned_resnet50(num_classes=len(train_dataset.classes))
    # for param in model.named_parameters():
    #     print(param)
    # exit()
    trained_model = train_model(model, train_dataset, val_dataset, optimizer=None, num_epochs=50, batch_size=32)
    #model = CustomAlexNet(num_classes=len(train_dataset.classes)).to(device)
    #optimizer = get_alexnet_optimizer(model)
    #trained_model = train_model(model, train_dataset, test_dataset, optimizer=optimizer, num_epochs=1000)

def process_imagenet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tiny_path = os.path.join("./data/TinyImageNet", "tiny-imagenet-200/train")
    tiny_path_val = os.path.join("./data/TinyImageNet", "tiny-imagenet-200/val/images_fixed")
    if not os.path.exists(tiny_path) or not os.path.exists(tiny_path_val):
        raise FileNotFoundError(f"Dataset not found at {tiny_path}. Please download the dataset.")
    train_dataset = ImageFolder(root=tiny_path, transform=transform)
    val_dataset = ImageFolder(root=tiny_path_val, transform=transform)
    print("Train Size {} Val Size {}".format(len(train_dataset), len(val_dataset)))
    idx_to_wnid = {i: wnid for i, wnid in enumerate(train_dataset.classes)}
    wnid_to_label = train_dataset.class_to_idx  # e.g., {'n01443537': 0, ...}

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

    model = get_finetuned_resnet18(num_classes=len(train_dataset.classes), dropout_p=0.2)
    trained_model = train_model(model, train_dataset, val_dataset, optimizer=None, num_epochs=50, batch_size=64)

    
    
    



if __name__ == "__main__":
    main()
    # main_train_msl()
    # train_mnist()
    # process_imagenet()

