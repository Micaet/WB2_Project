import argparse
import shutil
import numpy as np
from pathlib import Path
from torchvision import datasets


def prepare_skewed_train(
    data_dir: Path,
    out_dir: Path,
    alpha: float,
    min_per_class: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    ds = datasets.ImageFolder(str(data_dir))

    idx_by_class = {i: [] for i in range(len(ds.classes))}
    for sample_idx, (_, label) in enumerate(ds.samples):
        idx_by_class[label].append(sample_idx)

    n_classes = len(ds.classes)
    pool_sizes = np.array([len(idx_by_class[i]) for i in range(n_classes)])

    # Sample Dirichlet proportions, then scale total size so the dominant
    # class exactly fills its pool — no budget wasted, skew preserved.
    proportions = rng.dirichlet(np.full(n_classes, alpha))
    max_pool = pool_sizes[np.argmax(proportions)]
    scaled_size = int(max_pool / proportions.max())

    raw_counts = proportions * scaled_size
    counts = np.maximum(raw_counts, min_per_class)
    counts = np.round(counts / counts.sum() * scaled_size).astype(int)
    counts = np.maximum(counts, min_per_class)

    out_dir.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for cls_idx in range(n_classes):
        # Never request more images than the pool actually has
        counts[cls_idx] = min(counts[cls_idx], len(idx_by_class[cls_idx]))
        chosen = rng.choice(idx_by_class[cls_idx], size=counts[cls_idx], replace=False)

        class_name = ds.classes[cls_idx]
        class_out_dir = out_dir / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        for sample_idx in chosen:
            src_path, _ = ds.samples[sample_idx]
            shutil.copy(src_path, class_out_dir / Path(src_path).name)

        total_copied += 1
        print(f"Copied {counts[cls_idx]:>5d} images for class '{class_name}'")

    print(f"\nPrepared skewed dataset at {out_dir.resolve()}")
    print(f"Total images : {sum(counts[i] for i in range(n_classes))}")
    print(f"Min class    : {min(counts[i] for i in range(n_classes))}")
    print(f"Max class    : {max(counts[i] for i in range(n_classes))}")
    print(f"Imbalance    : {max(counts[i] for i in range(n_classes)) / max(min(counts[i] for i in range(n_classes)), 1):.1f}x")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Create a Dirichlet-skewed subset of an ImageFolder dataset. "
            "The total size is scaled automatically so the dominant class "
            "fills its entire pool."
        )
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Path to source ImageFolder train dir (e.g. places365/train)"
    )
    p.add_argument(
        "--out_dir", required=True,
        help="Path to save skewed train dir (e.g. data/skewed_places365_alpha05/train)"
    )
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="Dirichlet concentration parameter. Lower = more skewed. (default: 0.5)"
    )
    p.add_argument(
        "--min_per_class", type=int, default=200,
        help="Hard floor on images per class after sampling. (default: 200)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. (default: 42)"
    )
    args = p.parse_args()

    prepare_skewed_train(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        alpha=args.alpha,
        min_per_class=args.min_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()