import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import csv

import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy

from tqdm import tqdm

torch.manual_seed(42)


class CustomAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomAlexNet, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        
        # Keep the original features (convs)
        self.features = alexnet.features  # conv1–conv5
        
        # Replace classifier with custom structure
        self.fc6 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc7 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.extra_fc = nn.Linear(4096, 256)  # new 256-dim layer
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.extra_fc(x)
        x = self.classifier(x)
        return x
    
class MSLLabeledDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.split_file = os.path.join(root_dir, f'{split}-set-v2.1.txt')
        self.class_map_path = os.path.join(root_dir, 'class_map.csv')

        # Load samples
        with open(self.split_file, 'r') as f:
            lines = f.readlines()
        self.samples = [line.strip().split() for line in lines]
        self.samples = [(fname, int(label)) for fname, label in self.samples]

        # Load class map
        self.class_to_name = self._load_class_map()
        self.classes = [self.class_to_name[i] for i in sorted(self.class_to_name)]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Define transform
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_class_map(self):
        class_map = {}
        with open(self.class_map_path, 'r') as f:
            reader = csv.reader(f)
            #next(reader)  # Skip header
            for row in reader:
                idx, name = int(row[0]), row[1]
                class_map[idx] = name
        return class_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    

class MSLLabeledDatasetFilteredRemapped(Dataset):
    def __init__(self, root_dir, split='train', transform=None, exclude_labels=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.split_file = os.path.join(root_dir, f'{split}-set-v2.1.txt')
        self.class_map_path = os.path.join(root_dir, 'class_map.csv')
        self.exclude_labels = set(exclude_labels) if exclude_labels else set()

        # Load class map and keep only allowed classes
        full_class_map = self._load_class_map()
        kept_class_ids = sorted(set(full_class_map.keys()) - self.exclude_labels)

        # Build remapping from old class index → new index
        self.original_to_new = {old: new for new, old in enumerate(kept_class_ids)}
        self.new_to_name = {self.original_to_new[old]: full_class_map[old] for old in kept_class_ids}
        self.class_to_idx = {name: idx for idx, name in self.new_to_name.items()}
        self.classes = [self.new_to_name[i] for i in range(len(self.new_to_name))]

        # Load and filter samples, remap labels
        with open(self.split_file, 'r') as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            fname, label = line.strip().split()
            label = int(label)
            if label not in self.exclude_labels:
                new_label = self.original_to_new[label]
                self.samples.append((fname, new_label))

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_class_map(self):
        class_map = {}
        with open(self.class_map_path, 'r') as f:
            reader = csv.reader(f)
            #next(reader)  # skip header
            for row in reader:
                idx, name = int(row[0]), row[1]
                class_map[idx] = name
        return class_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class MSLLabeledDatasetAllSplits(Dataset):
    def __init__(self, root_dir, transform=None, exclude_labels=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.class_map_path = os.path.join(root_dir, 'class_map.csv')
        self.exclude_labels = set(exclude_labels) if exclude_labels else set()

        # Load class map and keep only allowed classes
        full_class_map = self._load_class_map()
        kept_class_ids = sorted(set(full_class_map.keys()) - self.exclude_labels)

        # Build remapping from old class index → new index
        self.original_to_new = {old: new for new, old in enumerate(kept_class_ids)}
        self.new_to_name = {self.original_to_new[old]: full_class_map[old] for old in kept_class_ids}
        self.class_to_idx = {name: idx for idx, name in self.new_to_name.items()}
        self.classes = [self.new_to_name[i] for i in range(len(self.new_to_name))]

        # Load and filter all split files (train, val, test)
        self.samples = []
        for split in ['train', 'val', 'test']:
            split_file = os.path.join(root_dir, f'{split}-set-v2.1.txt')
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    for line in f:
                        fname, label = line.strip().split()
                        label = int(label)
                        if label not in self.exclude_labels:
                            new_label = self.original_to_new[label]
                            self.samples.append((fname, new_label))

        # Set transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_class_map(self):
        class_map = {}
        with open(self.class_map_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                idx, name = int(row[0]), row[1]
                class_map[idx] = name
        return class_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_finetuned_resnet18(num_classes, dropout_p=0.5):
    model = models.resnet18(pretrained=True)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last two conv blocks
    for name, child in model.named_children():
        if name in ['layer3', 'layer4', 'fc']:
            for param in child.parameters():
                param.requires_grad = True

    # Replace the FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )

    return model

def get_finetuned_resnet50(num_classes, dropout_p=0.5):
    # Load pretrained ResNet-50
    model = models.resnet50(pretrained=True)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # # Unfreeze last two convolutional blocks:
    # for name, param in model.named_parameters():
    #     # if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
    #     if name.startswith("layer4") or name.startswith("fc"):
    #         param.requires_grad = True

    # Modify the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, 512),
        #nn.Dropout(p=dropout_p),
        nn.Linear(512, num_classes)
    )

    # Unfreeze last two convolutional blocks:
    for name, param in model.named_parameters():
        # if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True

    return model

def get_optimizer(model, base_lr=1e-3, fc_lr_multiplier=1):
    # Separate parameters
    fc_params = list(model.fc.parameters())
    conv_params = [p for name, p in model.named_parameters()
                   if p.requires_grad and not name.startswith('fc')]

    optimizer = torch.optim.Adam([
        {'params': conv_params, 'lr': base_lr},
        {'params': fc_params, 'lr': base_lr * fc_lr_multiplier}
    ])

    return optimizer

def get_alexnet_optimizer(model, base_lr=1e-4):
    # Freeze conv1–conv4 (named features.0 to features.8)
    conv_params_0x = []
    conv5_params_1x = []
    for idx, module in enumerate(model.features):
        for param in module.parameters():
            if idx <= 8:
                param.requires_grad = False
                conv_params_0x.append(param)
            else:
                conv5_params_1x.append(param)

    # fc6 + fc7 + extra_fc + classifier → 20×
    fc_params_20x = list(model.fc6.parameters()) + \
                    list(model.fc7.parameters()) + \
                    list(model.extra_fc.parameters()) + \
                    list(model.classifier.parameters())

    return torch.optim.Adam([
        {'params': conv5_params_1x, 'lr': base_lr},
        {'params': fc_params_20x, 'lr': base_lr * 20}
    ])

    
def train_model(model, train_dataset, test_dataset, optimizer, num_epochs=10, batch_size=64, device='cuda', savename='mslv2'):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if not optimizer: # means we are using resnet18
        optimizer = get_optimizer(model)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Test Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), './models/best_model_{}.pth'.format(savename))
            print("✅ Best model updated and saved!")

    print(f"Best test accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


