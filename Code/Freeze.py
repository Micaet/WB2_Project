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


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 45390
DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DATA_DIR = "imagenette2-320"
REPO_NAME = "ssl-data-curation"
DINO_MODEL = "facebook/dinov2-large"
NUM_CLASSES = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(SEED)


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



from src.clusters import HierarchicalCluster
from src import hierarchical_kmeans_gpu as hkmg
from src import hierarchical_sampling as hs


def get_dino_embeddings(dataset):
    print(f"\n--- KROK 1: Ekstrakcja cech SSL ({DINO_MODEL}) do samplingu ---")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL).to(device)
    model.eval()

    def dino_trans(img):
        return processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0)

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


def train_resnet50_comparison(train_ds, test_loader, mode="fine_tune", epochs=5):

    print(f"\n[Trening] Mode: {mode.upper()} | Rozmiar danych: {len(train_ds)}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if mode == "linear_probe":
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-2)
    else:
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
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

    acc = 100 * correct / total
    print(f"-> Wynik {mode}: {acc:.2f}%")
    return acc


# --- 5. GŁÓWNY EKSPERYMENT ---
def main():
    res_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=res_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=res_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    embeddings = get_dino_embeddings(full_train_dataset)

    print("\n--- KROK 2: Klastrowanie hierarchiczne ---")
    clusters_dict = hkmg.hierarchical_kmeans_with_resampling(
        data=embeddings.to(device),
        n_clusters=[500, 5],
        n_levels=2,
        sample_sizes=[15, 2],
        verbose=False
    )
    cl = HierarchicalCluster.from_dict(clusters_dict)


    percentages = [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]
    results_lp = {}
    results_ft = {}

    for p in percentages:
        label = f"{int(p * 100)}%"
        target_size = int(len(full_train_dataset) * p)

        if p < 1.0:
            idx = hs.hierarchical_sampling(cl, target_size=target_size)
            current_ds = Subset(full_train_dataset, idx)
        else:
            current_ds = full_train_dataset

        results_lp[label] = train_resnet50_comparison(current_ds, test_loader, mode="linear_probe")
        results_ft[label] = train_resnet50_comparison(current_ds, test_loader, mode="fine_tune")

    # 4. Wykres porównawczy
    plt.figure(figsize=(10, 6))
    x = np.arange(len(percentages))
    width = 0.35

    plt.bar(x - width / 2, results_lp.values(), width, label='Linear Probe (Frozen)', color='gray')
    plt.bar(x + width / 2, results_ft.values(), width, label='Fine-tuning (Unfrozen)', color='blue')

    plt.xlabel('Procent danych treningowych')
    plt.ylabel('Accuracy %')
    plt.title('Porównanie Strategii Trenowania ResNet50 (Weights V2)')
    plt.xticks(x, results_lp.keys())
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig('resnet50_comparison.png')
    plt.show()


if __name__ == "__main__":
    main()