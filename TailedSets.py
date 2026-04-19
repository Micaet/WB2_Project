import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from datasets import Dataset

# --- 1. KONFIGURACJA ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 45390
DATA_DIR = "Mao/tomas-gajarsky___cifar100-lt/r-50/0.0.0/63527001400d9b287a751876f1c70f5e4152e917"
REPO_NAME = "ssl-data-curation"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(SEED)

# --- 2. IMPORTY Z REPOZYTORIUM ---
repo_path = os.path.abspath(REPO_NAME)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from src.clusters import HierarchicalCluster
from src import hierarchical_kmeans_gpu as hkmg
from src import hierarchical_sampling as hs


# --- 3. DANE ---
class CifarArrowDataset(torch.utils.data.Dataset):
    def __init__(self, arrow_file, transform=None):
        self.ds = Dataset.from_file(arrow_file)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img, label = item['img'], item['fine_label']
        if self.transform:
            img = self.transform(img)
        return img, label


# --- 4. FUNKCJA TRENINGOWA (Wspiera oba tryby) ---
def train_experiment(train_ds, test_loader, device, title, pretrained=True, epochs=10):
    print(f"\n[START] {title} | Pretrained={pretrained}")

    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        lr = 0.0001
    else:
        model = models.resnet18(weights=None)
        lr = 0.001

    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoka {epoch + 1}/{epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total
    print(f"Wynik: {acc:.2f}%")
    return acc


# --- 5. GŁÓWNY PROCES ---
def main():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = CifarArrowDataset(os.path.join(DATA_DIR, "cifar100-lt-train.arrow"), transform=transform)
    test_ds = CifarArrowDataset(os.path.join(DATA_DIR, "cifar100-lt-test.arrow"), transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # A. Ekstrakcja cech (ResNet50 dla lepszej separacji klastrów)
    print("--- KROK 1: Ekstrakcja embeddingów (ResNet50) ---")
    extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    extractor.fc = nn.Identity()
    extractor.eval()

    feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(DataLoader(train_ds, batch_size=128), desc="Ekstrakcja"):
            feats.append(extractor(imgs.to(device)).cpu())
    data_tensor = torch.cat(feats)

    # B. Klastrowanie (TWOJE NOWE PARAMETRY)
    print("\n--- KROK 2: Hierarchiczny k-means (300, 50) ---")
    clusters_dict = hkmg.hierarchical_kmeans_with_resampling(
        data=data_tensor.to(device),
        n_clusters=[450, 150],
        n_levels=2,
        sample_sizes=[10, 2],
        verbose=False,
    )
    cl = HierarchicalCluster.from_dict(clusters_dict)

    # C. Pętla Eksperymentów
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_ft = {}  # Fine-tuning
    results_scratch = {}  # From scratch

    for p in percentages:
        label = f"{int(p * 100)}%"
        if p < 1.0:
            target_size = int(len(train_ds) * p)
            indices = hs.hierarchical_sampling(cl, target_size=target_size)
            current_ds = Subset(train_ds, indices)
        else:
            current_ds = train_ds

        # 1. Test Fine-tuning
        acc_ft = train_experiment(current_ds, test_loader, device, f"FT {label}", pretrained=True)
        results_ft[label] = acc_ft

        # 2. Test Scratch
        acc_sc = train_experiment(current_ds, test_loader, device, f"SCRATCH {label}", pretrained=False)
        results_scratch[label] = acc_sc

    # --- 6. GENEROWANIE 2 WYKRESÓW ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Wykres 1: Fine-tuning
    bars1 = ax1.bar(results_ft.keys(), results_ft.values(), color='skyblue', edgecolor='black')
    ax1.set_title('Fine-tuning ResNet18 (Wagi ImageNet)', fontsize=14)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, max(results_ft.values()) + 10)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{bar.get_height():.1f}%', ha='center',
                 fontweight='bold')

    # Wykres 2: Scratch
    bars2 = ax2.bar(results_scratch.keys(), results_scratch.values(), color='salmon', edgecolor='black')
    ax2.set_title('Trening ResNet18 od zera (Random Init)', fontsize=14)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, max(results_scratch.values()) + 10)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{bar.get_height():.1f}%', ha='center',
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig('porownanie_podejsc.png')
    plt.show()


if __name__ == "__main__":
    main()