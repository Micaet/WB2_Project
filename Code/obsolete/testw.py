import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import tarfile
import urllib.request

# --- 1. ROZWIĄZANIE KONFLIKTÓW SYSTEMOWYCH ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 2. KONFIGURACJA REPOZYTORIUM ---
REPO_NAME = "ssl-data-curation"
REPO_URL = "https://github.com/facebookresearch/ssl-data-curation.git"


#
def setup_repo():
    if not os.path.exists(REPO_NAME):
        print(f"--- Klonowanie repozytorium {REPO_NAME}... ---")
        subprocess.run(["git", "clone", REPO_URL], check=True)

    # Dodanie katalogów do sys.path, aby Python widział folder 'src'
    repo_path = os.path.abspath(REPO_NAME)
    src_path = os.path.join(repo_path, "src")

    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    print("--- Ścieżki repozytorium skonfigurowane ---")


#
setup_repo()
# # --- 3. IMPORTY Z REPOZYTORIUM (PO USTAWIENIU ŚCIEŻEK) ---
try:
    from src.clusters import HierarchicalCluster
    from src import hierarchical_kmeans_gpu as hkmg
    from src import hierarchical_sampling as hs

    print("--- Moduły Facebook Research załadowane pomyślnie ---")
except ImportError as e:
    print(f"Błąd importu: {e}")
    sys.exit(1)
#
# # --- 4. PRZYGOTOWANIE DANYCH (IMAGENETTE) ---
DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DATA_DIR = "imagenette2-320"


#
#
# def prepare_data():
#     if not os.path.exists(DATA_DIR):
#         print("--- Pobieranie zbioru Imagenette (320px)... ---")
#         urllib.request.urlretrieve(DATA_URL, "imagenette.tgz")
#         with tarfile.open("imagenette.tgz", "r:gz") as tar:
#             tar.ext


# --- 5. GŁÓWNA FUNKCJA TRENINGOWA ---
def train_model(train_loader, test_loader, device, title, epochs=5):
    print(f"\n[TRENING] Start: {title}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoka {epoch + 1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Ewaluacja
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"[WYNIK] {title} Accuracy: {accuracy:.2f}%")
    return accuracy


# --- 6. URUCHOMIENIE EKSPERYMENTU ---
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Praca na urządzeniu: {device}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"--- TRYB PEŁNY: Zbiór treningowy liczy {len(full_train_dataset)} obrazów ---")

    # A. EKSTRAKCJA CECH (Dla wszystkich obrazów w zbiorze)
    print("\n--- KROK 1: Ekstrakcja cech dla całego zbioru (ResNet18 Pretrained) ---")
    extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    extractor.fc = nn.Identity()
    extractor.eval()

    all_features = []
    with torch.no_grad():

        feature_loader = DataLoader(full_train_dataset, batch_size=128, shuffle=False)
        for imgs, _ in tqdm(feature_loader, desc="Ekstrakcja cech"):
            feat = extractor(imgs.to(device))
            all_features.append(feat.cpu())

    data_tensor = torch.cat(all_features)

    # B. SAMPLING (Twoja metoda na pełnych danych)
    print("\n--- KROK 2: Ranking i Sampling (Wybór 1000 najlepszych obrazów) ---")
    clusters = hkmg.hierarchical_kmeans_with_resampling(
        data=data_tensor.to(device),
        n_clusters=[120, 50],
        n_levels=2,
        sample_sizes=[15, 2],
        verbose=False,
    )

    cl = HierarchicalCluster.from_dict(clusters)

    target_subset_size = 8000
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_subset_size)

    sampled_dataset = Subset(full_train_dataset, sampled_indices)
    print(f"Wyselekcjonowano {len(sampled_indices)} obrazów za pomocą Twojej metody.")

    num_epochs = 10

    acc_sampled = train_model(
        DataLoader(sampled_dataset, batch_size=64, shuffle=True),
        test_loader,
        device,
        f"RANKING SUBSET ({target_subset_size} images)",
        epochs=num_epochs
    )


    acc_full = train_model(
        DataLoader(full_train_dataset, batch_size=64, shuffle=True),
        test_loader,
        device,
        f"FULL DATASET ({len(full_train_dataset)} images)",
        epochs=num_epochs
    )

    print("\n" + "=" * 50)
    print(f"FINALNE PODSUMOWANIE PO {num_epochs} EPOKACH:")
    print(f"Accuracy - Pełny zbiór: {acc_full:.2f}%")
    print(f"Accuracy - Twoja metoda (tylko {target_subset_size} zdjęć): {acc_sampled:.2f}%")

    efektywnosc = acc_sampled / acc_full * 100
    print(f"Twoja metoda osiągnęła {efektywnosc:.1f}% jakości pełnego zbioru, "
          f"używając jedynie {target_subset_size / len(full_train_dataset) * 100:.1f}% danych!")
    print("=" * 50)


if __name__ == "__main__":
    main()
