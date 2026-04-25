"""
Hierarchical k-means SSL data curation experiment on ImageNet1K.
"""

import os
import sys
import subprocess
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------------------------------------
# ── USER CONFIGURATION ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

IMAGENET_TRAIN_DIR = "/home/sirko/wb-data/train"
IMAGENET_VAL_DIR = "/home/sirko/wb_2026/WB2_Project/imagenette2-320/val"

SELECTED_SYNSETS = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]

SYNSET_NAMES = [
    "tench", "springer", "cassette", "chainsaw", "church",
    "Fr. horn", "garbage truck", "gas pump", "golf ball", "parachute",
]

TARGET_TRAIN_SIZE = 13000
ALPHA_VALUES      = [0.5]
SEEDS             = [42]
PERCENTAGES       = [0.8]
DINO_MODEL        = "facebook/dinov2-large"
NUM_CLASSES       = 10
EPOCHS            = 50
OUTPUT_CSV        = "results.csv"
EMBEDDINGS       = "resnet"  # "resnet" or "dino"

# ---------------------------------------------------------------------------

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

REPO_NAME = "ssl-data-curation"

def setup_repo():
    if not os.path.exists(REPO_NAME):
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/ssl-data-curation.git"],
            check=True,
        )
    sys.path.insert(0, os.path.abspath(REPO_NAME))

def build_skewed_train_dataset(imagenet_train_dir: str, synsets: list[str], target_size: int, alpha: float, seed: int, transform) -> datasets.ImageFolder:
    rng = np.random.default_rng(seed)
    full_ds = datasets.ImageFolder(imagenet_train_dir, transform=transform)
    class_to_idx = full_ds.class_to_idx
    synset_indices = {s: class_to_idx[s] for s in synsets if s in class_to_idx}
    
    if len(synset_indices) < len(synsets):
        missing = set(synsets) - set(synset_indices)
        raise ValueError(f"Synsets not found in {imagenet_train_dir}: {missing}")

    idx_by_class: dict[int, list[int]] = {v: [] for v in synset_indices.values()}
    for sample_idx, (_, label) in enumerate(full_ds.samples):
        if label in idx_by_class:
            idx_by_class[label].append(sample_idx)

    proportions = rng.dirichlet(alpha=np.full(NUM_CLASSES, alpha))
    counts = {
        cls_idx: min(max(1, round(prop * target_size)), len(idx_by_class[cls_idx]))
        for cls_idx, prop in zip(idx_by_class.keys(), proportions)
    }

    actual_total = sum(counts.values())
    print(f"  Dirichlet(alpha={alpha}) class sizes [min={min(counts.values())}, max={max(counts.values())}, total={actual_total}]")

    selected: list[int] = []
    for cls_idx, n in counts.items():
        chosen = rng.choice(idx_by_class[cls_idx], size=n, replace=False).tolist()
        selected.extend(chosen)

    return Subset(full_ds, selected)

def build_val_dataset(imagenet_val_dir: str, synsets: list[str], transform) -> Subset:
    full_val = datasets.ImageFolder(imagenet_val_dir, transform=transform)
    class_to_idx = full_val.class_to_idx
    target_labels = {class_to_idx[s] for s in synsets if s in class_to_idx}
    indices = [i for i, (_, lbl) in enumerate(full_val.samples) if lbl in target_labels]
    return Subset(full_val, indices)

def get_dino_embeddings(dataset, dino_transform) -> torch.Tensor:
    print(f"\n  STEP 1: SSL feature extraction ({DINO_MODEL})")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL).to(device)
    model.eval()

    inner = dataset.dataset if isinstance(dataset, Subset) else dataset
    original_transform = inner.transform

    def dino_trans(img):
        return processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0)

    inner.transform = dino_trans
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    features = []
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for imgs, _ in tqdm(loader, desc="  DINO"):
            out = model(imgs.to(device))
            features.append(out.last_hidden_state[:, 0, :].cpu())

    inner.transform = original_transform
    embeddings = torch.cat(features)
    return F.normalize(embeddings, p=2, dim=1)

def get_resnet_embeddings(dataset, device) -> torch.Tensor:
    print("\n  STEP 1: SSL feature extraction (ResNet-50)")
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    features = []

    with torch.inference_mode():
        for imgs, _ in tqdm(loader, desc="  ResNet"):
            out = backbone(imgs.to(device))
            features.append(out.cpu())

    embeddings = torch.cat(features)
    return F.normalize(embeddings, p=2, dim=1)

def train_resnet50(train_ds, test_loader: DataLoader, mode: str = "fine_tune", epochs: int = EPOCHS) -> float:
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
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, pred = torch.max(model(imgs), 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"    [{mode:>14s}]  acc = {acc:.2f}%  (n_train={len(train_ds)})")
    return acc

def get_top_level_labels(cl, n_samples: int) -> np.ndarray:
    levels = sorted(cl.clusters.keys())
    
    labels = np.full(n_samples, -1, dtype=int)
    for fine_id, sample_indices in enumerate(cl.clusters[levels[0]]):
        for idx in sample_indices:
            labels[idx] = fine_id
            
    for lvl in levels[1:]:
        parent_clusters = cl.clusters[lvl]
        child_to_parent = {}
        for parent_id, child_ids in enumerate(parent_clusters):
            for child_id in child_ids:
                child_to_parent[child_id] = parent_id
                
        for i in range(n_samples):
            if labels[i] != -1:
                labels[i] = child_to_parent.get(labels[i], -1)
                
    return labels

def get_true_labels_for_subset(subset: Subset) -> np.ndarray:
    inner = subset.dataset
    raw = np.array([inner.samples[i][1] for i in subset.indices])
    unique = sorted(set(raw.tolist()))
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[v] for v in raw])

def visualize_clusters_umap(embeddings: torch.Tensor, true_labels: np.ndarray, coarse_labels: np.ndarray, alpha: float, seed: int, class_names: list) -> tuple:
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn:  pip install umap-learn")

    print(f"  UMAP projection (alpha={alpha}, seed={seed}) ...", flush=True)
    reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=1, verbose=False)
    emb_2d = reducer.fit_transform(embeddings.numpy())

    n_classes   = len(class_names)
    n_coarse    = int(coarse_labels.max()) + 1
    class_cmap  = plt.cm.get_cmap("tab10", n_classes)
    coarse_cmap = plt.cm.get_cmap("Set1",  max(n_coarse, 10))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    for cls_id in range(n_classes):
        mask = true_labels == cls_id
        ax_l.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=[class_cmap(cls_id)], s=4, alpha=0.6, linewidths=0,
            label=f"{class_names[cls_id]}  (n={mask.sum()})",
        )
    ax_l.set_title("Coloured by true class", fontsize=11)
    ax_l.set_xlabel("UMAP-1"); ax_l.set_ylabel("UMAP-2")
    ax_l.legend(markerscale=3, fontsize=7, loc="best", title="class", title_fontsize=8, framealpha=0.7, ncol=2)

    for c_id in range(n_coarse):
        mask = coarse_labels == c_id
        ax_r.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=[coarse_cmap(c_id)], s=4, alpha=0.6, linewidths=0,
            label=f"cluster {c_id}  (n={mask.sum()})",
        )
    ax_r.set_title(f"Coloured by coarse cluster  (K={n_coarse})", fontsize=11)
    ax_r.set_xlabel("UMAP-1")
    ax_r.legend(markerscale=3, fontsize=7, loc="best", title="cluster", title_fontsize=8, framealpha=0.7)

    fig.suptitle(f"DINOv2 embeddings -- Dirichlet alpha={alpha},  seed={seed}\n({len(true_labels):,} training samples)", fontsize=12, y=1.01)
    plt.tight_layout()
    fname = f"clusters_alpha{alpha}_seed{seed}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {fname}")

    return emb_2d

def build_combined_cluster_figure(records: list, seed: int):
    n_rows = len(records)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :] 

    for row_i, rec in enumerate(records):
        emb_2d        = rec["emb_2d"]
        true_labels   = rec["true_labels"]
        coarse_labels = rec["coarse_labels"]
        alpha         = rec["alpha"]
        class_names   = rec["class_names"]

        n_classes   = len(class_names)
        n_coarse    = int(coarse_labels.max()) + 1
        class_cmap  = plt.cm.get_cmap("tab10", n_classes)
        coarse_cmap = plt.cm.get_cmap("Set1",  max(n_coarse, 10))

        ax_l, ax_r = axes[row_i, 0], axes[row_i, 1]

        for cls_id in range(n_classes):
            mask = true_labels == cls_id
            ax_l.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[class_cmap(cls_id)], s=3, alpha=0.5, linewidths=0, label=f"{class_names[cls_id]} (n={mask.sum()})")
        ax_l.set_title(f"alpha={alpha}  |  true class", fontsize=10)
        ax_l.set_xlabel("UMAP-1"); ax_l.set_ylabel("UMAP-2")
        ax_l.legend(markerscale=3, fontsize=6, loc="best", ncol=2, framealpha=0.7)

        for c_id in range(n_coarse):
            mask = coarse_labels == c_id
            ax_r.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[coarse_cmap(c_id)], s=3, alpha=0.5, linewidths=0, label=f"cluster {c_id}  n={mask.sum()}")
        ax_r.set_title(f"alpha={alpha}  |  coarse cluster  (K={n_coarse})", fontsize=10)
        ax_r.set_xlabel("UMAP-1")
        ax_r.legend(markerscale=3, fontsize=6, loc="best", framealpha=0.7)

    fig.suptitle(f"DINOv2 UMAP projections across Dirichlet alpha values  (seed={seed})\nLeft: true class distribution  |  Right: coarse SSL cluster assignment", fontsize=12, y=1.01)
    plt.tight_layout()
    fname = f"clusters_combined_seed{seed}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCombined cluster overview saved -> {fname}")

def main():
    setup_repo()

    from src.clusters import HierarchicalCluster
    import src.hierarchical_kmeans_gpu as hkmg
    import src.hierarchical_sampling as hs

    res_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_ds = build_val_dataset(IMAGENET_VAL_DIR, SELECTED_SYNSETS, res_transform)
    test_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    csv_fields = ["seed", "alpha", "percentage", "n_samples", "linear_probe_acc", "fine_tune_acc"]
    csv_file = open(OUTPUT_CSV, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    VIZ_SEED = SEEDS[0]          
    viz_records: list[dict] = [] 

    for seed in SEEDS:
        set_seed(seed)
        print(f"\n{'='*70}")
        print(f"  SEED = {seed}")
        print(f"{'='*70}")

        for alpha in ALPHA_VALUES:
            print(f"\n  alpha = {alpha}")

            skewed_ds = build_skewed_train_dataset(IMAGENET_TRAIN_DIR, SELECTED_SYNSETS, TARGET_TRAIN_SIZE, alpha, seed, res_transform)
            if EMBEDDINGS == "resnet":
                embeddings = get_resnet_embeddings(skewed_ds, device)
            else:
                embeddings = get_dino_embeddings(skewed_ds, res_transform)

            print("\n  STEP 2: Hierarchical k-means clustering")
            clusters_dict = hkmg.hierarchical_kmeans_with_resampling(
                data=embeddings.to(device),
                n_clusters=[100, 30, 10], 
                n_levels=3,
                sample_sizes=[65, 216, 650], 
                verbose=False,
            )
            cl = HierarchicalCluster.from_dict(clusters_dict)

            if seed == VIZ_SEED:
                print("\n  STEP 2b: Cluster visualisation (UMAP)")
                true_labels   = get_true_labels_for_subset(skewed_ds)
                coarse_labels = get_top_level_labels(cl, len(skewed_ds))
                emb_2d = visualize_clusters_umap(embeddings, true_labels, coarse_labels, alpha, seed, SYNSET_NAMES)
                viz_records.append(dict(alpha=alpha, emb_2d=emb_2d, true_labels=true_labels, coarse_labels=coarse_labels, class_names=SYNSET_NAMES))

            print(f"\n  STEP 3: Training at {len(PERCENTAGES)} data budgets")
            for p in PERCENTAGES:
                label = f"{int(p * 100)}%"
                target_size = int(len(skewed_ds) * p)

                if p < 1.0:
                    idx = hs.hierarchical_sampling(cl, target_size=target_size)
                    current_ds = Subset(skewed_ds, idx)
                else:
                    current_ds = skewed_ds

                lp_acc = train_resnet50(current_ds, test_loader, mode="linear_probe", epochs=50)
                ft_acc = train_resnet50(current_ds, test_loader, mode="fine_tune", epochs=25)

                row = dict(seed=seed, alpha=alpha, percentage=label, n_samples=len(current_ds), linear_probe_acc=round(lp_acc, 4), fine_tune_acc=round(ft_acc, 4))
                writer.writerow(row)
                csv_file.flush()          

    if viz_records:
        build_combined_cluster_figure(viz_records, VIZ_SEED)

    csv_file.close()
    print(f"\n✓  All results saved to  {OUTPUT_CSV}")

    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)

    fig, axes = plt.subplots(1, len(ALPHA_VALUES), figsize=(6 * len(ALPHA_VALUES), 5), sharey=True)
    if len(ALPHA_VALUES) == 1:
        axes = [axes]
        
    for ax, alpha in zip(axes, ALPHA_VALUES):
        sub = df[df["alpha"] == alpha].groupby("percentage")[["linear_probe_acc", "fine_tune_acc"]].mean().reindex([f"{int(p*100)}%" for p in PERCENTAGES])
        x = np.arange(len(PERCENTAGES))
        w = 0.35
        ax.bar(x - w / 2, sub["linear_probe_acc"], w, label="Linear Probe", color="gray")
        ax.bar(x + w / 2, sub["fine_tune_acc"], w, label="Fine-tune", color="steelblue")
        ax.set_title(f"alpha = {alpha}")
        ax.set_xlabel("% of training data")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("ResNet-50 on Dirichlet-skewed ImageNet subset\n(mean over 5 seeds, hierarchical SSL sampling)", fontsize=13)
    plt.tight_layout()
    plt.savefig("results_summary.png", dpi=150)
    print("✓  Summary plot saved to  results_summary.png")

if __name__ == "__main__":
    main()