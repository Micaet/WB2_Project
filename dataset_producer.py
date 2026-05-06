import argparse
import shutil
import numpy as np
from pathlib import Path
from torchvision import datasets

def prepare_skewed_train(data_dir: Path, out_dir: Path, target_size: int, alpha: float, min_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    ds = datasets.ImageFolder(str(data_dir))
    
    idx_by_class = {i: [] for i in range(len(ds.classes))}
    for sample_idx, (_, label) in enumerate(ds.samples):
        idx_by_class[label].append(sample_idx)
        
    n_classes = len(ds.classes)
    proportions = rng.dirichlet(np.full(n_classes, alpha))
    raw_counts = proportions * target_size
    
    counts = np.maximum(raw_counts, min_per_class)
    counts = np.round(counts / counts.sum() * target_size).astype(int)
    counts = np.maximum(counts, min_per_class)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    total_copied = 0

    for cls_idx in range(n_classes):
        counts[cls_idx] = min(counts[cls_idx], len(idx_by_class[cls_idx]))
        chosen = rng.choice(idx_by_class[cls_idx], size=counts[cls_idx], replace=False)
        class_name = ds.classes[cls_idx]
        
        class_out_dir = out_dir / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_idx in chosen:
            src_path, _ = ds.samples[sample_idx]
            shutil.copy(src_path, class_out_dir / Path(src_path).name)
            total_copied += 1
            
        print(f"Copied {counts[cls_idx]} images for class {class_name}")

    print(f"\nPrepared skewed dataset at {out_dir.resolve()} with {total_copied} total images.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Path to original balanced train dir (e.g. imagenet/train)")
    p.add_argument("--out_dir", required=True, help="Path to save skewed train dir (e.g. imagenet_skewed/train)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--size", type=int, default=13000)
    p.add_argument("--min_per_class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prepare_skewed_train(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        target_size=args.size,
        alpha=args.alpha,
        min_per_class=args.min_per_class,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()