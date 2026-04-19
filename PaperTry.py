import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import tarfile
import urllib.request
from transformers import AutoImageProcessor, AutoModel

# --- 1. KONFIGURACJA ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 45390
# Konfiguracja Imagenette (z testw.py)
DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DATA_DIR = "imagenette2-320"
REPO_NAME = "ssl-data-curation"
DINO_MODEL = "facebook/dinov2-large"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(SEED)


# --- 2. PRZYGOTOWANIE DANYCH I REPOZYTORIUM ---
def prepare_data():
    if not os.path.exists(DATA_DIR):
        print("--- Pobieranie zbioru Imagenette (320px)... ---")
        if not os.path.exists("imagenette.tgz"):
            urllib.request.urlretrieve(DATA_URL, "imagenette.tgz")
        with tarfile.open("imagenette.tgz", "r:gz") as tar:
            tar.extractall()
    print("--- Dane Imagenette gotowe ---")


def setup_repo():
    if not os.path.exists(REPO_NAME):
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/ssl-data-curation.git"], check=True)
    sys.path.insert(0, os.path.abspath(REPO_NAME))


prepare_data()
setup_repo()

from src.clusters import HierarchicalCluster
from src import hierarchical_kmeans_gpu as hkmg
from src import hierarchical_sampling as hs


# --- 3. EKSTRAKCJA CECH DINOv2 ---
def get_dino_embeddings(dataset):
    print(f"\n--- KROK 1: Ekstrakcja cech SSL ({DINO_MODEL}) ---")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL).to(device)
    model.eval()

    def dino_trans(img):
        return processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0)

    # Tymczasowy dataset z transformacją DINO
    original_transform = dataset.transform
    dataset.transform = dino_trans
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    features = []
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for imgs, _ in tqdm(loader, desc="DINO Extraction"):
            outputs = model(imgs.to(device))
            emb = outputs.last_hidden_state[:, 0, :]
            features.append(emb.cpu())

    dataset.transform = original_transform
    return torch.cat(features)


# --- 4. SILNIK TRENINGOWY ---
def train_model(train_ds, test_loader, device, title, pretrained=True, epochs=10):
    print(f"\n[Trening] {title} (Pretrained={pretrained})")

    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    else:
        model = models.resnet18(weights=None)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Imagenette ma 10 klas
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

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
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Wynik {title}: {accuracy:.2f}%")
    return accuracy


# --- 5. GŁÓWNY EKSPERYMENT ---
def main():
    res_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Ładowanie Imagenette przy użyciu ImageFolder (jak w testw.py)
    full_train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=res_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=res_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 1. SSL Embeddings
    embeddings = get_dino_embeddings(full_train_dataset)

    # 2. Hierarchical K-Means (500x5)
    print("\n--- KROK 2: Klastrowanie Meta FAIR Style (500x5) ---")
    clusters_dict = hkmg.hierarchical_kmeans_with_resampling(
        data=embeddings.to(device),
        n_clusters=[500, 5],
        n_levels=2,
        sample_sizes=[15, 2],
        verbose=False
    )
    cl = HierarchicalCluster.from_dict(clusters_dict)

    # 3. Pętla testowa
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results_ft, results_scratch = {}, {}

    for p in percentages:
        label = f"{int(p * 100)}%"
        if p < 1.0:
            target_size = int(len(full_train_dataset) * p)
            idx = hs.hierarchical_sampling(cl, target_size=target_size)
            current_ds = Subset(full_train_dataset, idx)
        else:
            current_ds = full_train_dataset

        results_ft[label] = train_model(current_ds, test_loader, device, f"FT {label}", True)
        results_scratch[label] = train_model(current_ds, test_loader, device, f"Scratch {label}", False)

    # 4. Wykresy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    def plot_res(ax, data, title, color):
        ax.bar(data.keys(), data.values(), color=color, edgecolor='black')
        ax.set_title(title)
        ax.set_ylabel("Accuracy %")
        for i, v in enumerate(data.values()):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')

    plot_res(ax1, results_ft, "Imagenette: Fine-tuning ResNet18 (Pretrained)", "skyblue")
    plot_res(ax2, results_scratch, "Imagenette: Training ResNet18 from Scratch", "salmon")

    plt.tight_layout()
    plt.savefig('meta_curation_imagenette.png')
    plt.show()


if __name__ == "__main__":
    main()