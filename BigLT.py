import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import random
import matplotlib.pyplot as plt
from datasets import load_dataset


REPO_NAME = "ssl-data-curation"
if os.path.exists(REPO_NAME):
    sys.path.insert(0, os.path.abspath(REPO_NAME))

try:
    from src.clusters import HierarchicalCluster
    from src import hierarchical_kmeans_gpu as hkmg
    from src import hierarchical_sampling as hs
except ImportError:
    print("BŁĄD: Nie można zaimportować modułów klastrowania z ssl-data-curation.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_CLASSES = 1000
MAX_EPOCHS = 10
PERCENTAGES = [30, 40, 70,80, 90, 100]

os.makedirs('wykresy', exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class HFImageDataset(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.hf_ds = hf_ds
        self.transform = transform
        self.targets = self.hf_ds['label']

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item['image'].convert('RGB')
        label = item['label']

        if self.transform:
            img = self.transform(img)

        return img, label


def create_frozen_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model.to(device)


def train_model_on_images(train_loader, val_loader, test_loader, title, y_train_labels):
    print(f"\n Start treningu: {title}")

    model = create_frozen_resnet()

    classes_present = np.unique(y_train_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y_train_labels)
    class_weights = torch.ones(NUM_CLASSES).to(device)
    for c, w in zip(classes_present, weights):
        class_weights[c] = w

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= train_total
        train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        print(
            f"  -> Epoka [{epoch + 1:02d}/{MAX_EPOCHS:02d}] | Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)",
            flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f" Zakończono. TEST ACCURACY: {test_acc:.2f}%", flush=True)
    return test_acc


def plot_class_distributions(y_res, y_din, percentage):
    counts_res = np.bincount(y_res, minlength=NUM_CLASSES)
    counts_din = np.bincount(y_din, minlength=NUM_CLASSES)


    p_res = (counts_res + 1) / np.sum(counts_res + 1)
    p_din = (counts_din + 1) / np.sum(counts_din + 1)


    kl_div = np.sum(p_res * np.log(p_res / p_din))

    x = np.arange(NUM_CLASSES)
    width = 0.4

    fig, ax = plt.subplots(figsize=(22, 6))


    ax.bar(x - width / 2, counts_res, width, label='ResNet Selection', color='teal', alpha=0.9, align='center')
    ax.bar(x + width / 2, counts_din, width, label='DINO Selection', color='indigo', alpha=0.9, align='center')

    ax.set_ylabel('Liczba próbek')
    ax.set_xlabel('Indeks Klasy (0 - 999)')
    ax.set_title(f'Rozkład klas - Podzbiór {percentage}% | KL Divergence (ResNet||DINO): {kl_div:.4f}')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_xticks(np.arange(0, NUM_CLASSES, 50))
    ax.set_xlim(-1, NUM_CLASSES)

    plt.tight_layout()
    plt.savefig(f'wykresy/rozklad_klas_{percentage}_procent.png', dpi=300)
    plt.close()
    print(f"   [Wizualizacja] Zapisano wykres: wykresy/rozklad_klas_{percentage}_procent.png (KL: {kl_div:.4f})",
          flush=True)


def plot_final_accuracy(results_res, results_din, percentages):
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, results_res, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8,
             label='ResNet Selection')
    plt.plot(percentages, results_din, marker='s', linestyle='-', color='indigo', linewidth=2, markersize=8,
             label='DINO Selection')

    plt.title('Porównanie skuteczności (Test Accuracy)', fontsize=14)
    plt.xlabel('Procent użytych danych treningowych (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(percentages)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('wykresy/podsumowanie_accuracy.png', dpi=300)
    plt.close()
    print("\n   [Wizualizacja] Zapisano wykres ostateczny: wykresy/podsumowanie_accuracy.png", flush=True)


def main():
    print("--- Pobieranie/Ładowanie zbioru z Hugging Face ---", flush=True)
    hf_dataset = load_dataset("inria-chile/imagenet-lt-v2")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_train_dataset = HFImageDataset(hf_dataset['train'], transform=transform)
    val_test_dataset = HFImageDataset(hf_dataset['val'], transform=transform)

    print("--- Ładowanie embeddingów do klasteryzacji... ---", flush=True)
    data = torch.load('embeddings_imagenet_dino.pt', map_location=device)
    train_resnet_emb = data['train_resnet'].to(device)
    train_dino_emb = data['train_dino'].to(device)

    all_train_labels = np.array(full_train_dataset.targets)
    total_train_samples = len(all_train_labels)

    val_loader = DataLoader(val_test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(val_test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    N_CLUSTERS = [120, 50]
    N_LEVELS = 2

    final_acc_resnet = []
    final_acc_dino = []


    for p in PERCENTAGES:
        print(f"\n{'*' * 40}\nEKSPERYMENT: {p}% DANYCH TRENINGOWYCH\n{'*' * 40}", flush=True)
        num_samples = int(total_train_samples * (p / 100.0))

        if p == 100:
            idx_res = list(range(total_train_samples))
            idx_din = list(range(total_train_samples))
        else:
            sz = [max(1, int(num_samples / n)) for n in N_CLUSTERS]

            set_seed(SEED)
            c_res = hkmg.hierarchical_kmeans_with_resampling(
                data=train_resnet_emb.float(), n_clusters=N_CLUSTERS, n_levels=N_LEVELS, sample_sizes=sz, verbose=False
            )
            idx_res = hs.hierarchical_sampling(HierarchicalCluster.from_dict(c_res), target_size=num_samples).astype(
                int).tolist()

            set_seed(SEED)
            c_din = hkmg.hierarchical_kmeans_with_resampling(
                data=train_dino_emb.float(), n_clusters=N_CLUSTERS, n_levels=N_LEVELS, sample_sizes=sz, verbose=False
            )
            idx_din = hs.hierarchical_sampling(HierarchicalCluster.from_dict(c_din), target_size=num_samples).astype(
                int).tolist()

        train_subset_res = Subset(full_train_dataset, idx_res)
        train_subset_din = Subset(full_train_dataset, idx_din)

        y_labels_res = all_train_labels[idx_res]
        y_labels_din = all_train_labels[idx_din]

        plot_class_distributions(y_labels_res, y_labels_din, p)

        train_loader_res = DataLoader(train_subset_res, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
        train_loader_din = DataLoader(train_subset_din, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

        set_seed(SEED)
        acc_resnet = train_model_on_images(
            train_loader=train_loader_res,
            val_loader=val_loader,
            test_loader=test_loader,
            title=f"Podzbiór ResNet ({p}%)",
            y_train_labels=y_labels_res
        )

        set_seed(SEED)
        acc_dino = train_model_on_images(
            train_loader=train_loader_din,
            val_loader=val_loader,
            test_loader=test_loader,
            title=f"Podzbiór DINO ({p}%)",
            y_train_labels=y_labels_din
        )

        final_acc_resnet.append(acc_resnet)
        final_acc_dino.append(acc_dino)

        print(f"\n WYNIKI DLA {p}%:", flush=True)
        print(f"   -> ResNet Selection: {acc_resnet:.2f}%", flush=True)
        print(f"   -> DINO Selection:   {acc_dino:.2f}%\n", flush=True)

    plot_final_accuracy(final_acc_resnet, final_acc_dino, PERCENTAGES)


if __name__ == "__main__":
    main()