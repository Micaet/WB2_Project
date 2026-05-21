"""
Download 10 selected Places365 classes from the official dataset and save them
in torchvision ImageFolder format:

    places365/
    └── train/
        ├── airport_terminal/
        ├── beach/
        └── ...

⚠ The full Places365-Standard (small, 256×256) train split is ~24 GB.
  On Colab, point --download_dir at a mounted Google Drive path so the
  download survives session restarts.

Prerequisites
-------------
    pip install torch torchvision tqdm

Usage
-----
    python download_places365_classes.py

    # On Colab with Drive mounted:
    python download_places365_classes.py \\
        --download_dir /content/drive/MyDrive/places365_raw \\
        --output_dir   /content/drive/MyDrive/places365/train
"""

import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm


# ── Class selection ───────────────────────────────────────────────────────────
# 10 visually distinct Places365 classes, each with ~4 000–5 000 train images.
# The dataset uses the format '/a/airport_terminal'; we match on the final part.

SELECTED_CLASSES = [
    "airport_terminal",
    "beach",
    "bedroom",
    "canyon",
    "classroom",
    "forest_path",
    "kitchen",
    "mountain",
    "office",
    "restaurant",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_short_to_idx(dataset):
    """Map 'airport_terminal' -> int label from '/a/airport_terminal' keys."""
    return {k.split("/")[-1]: v for k, v in dataset.class_to_idx.items()}


def resolve_img_path(raw_file: str, dataset) -> Path:
    """
    torchvision Places365 stores img paths relative to <root>/<dir_365>/.
    __getitem__ does: os.path.join(self.root, self.dir_365, file)
    We replicate that here.
    """
    if os.path.isabs(raw_file):
        return Path(raw_file)
    dir_365 = getattr(dataset, "dir_365", None)
    if dir_365 is not None:
        return Path(dataset.root) / dir_365 / raw_file
    # Fallback: search common sub-directories
    for subdir in ("data_256", "data", ""):
        candidate = Path(dataset.root) / subdir / raw_file
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve image path '{raw_file}' under {dataset.root}. "
        "Check that the dataset downloaded and extracted correctly."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download 10 Places365 classes into ImageFolder format."
    )
    parser.add_argument(
        "--download_dir",
        default="places365_raw",
        help="Where to store the full downloaded dataset (~24 GB).",
    )
    parser.add_argument(
        "--output_dir",
        default="places365/train",
        help="Output directory in ImageFolder format (only the 10 classes).",
    )
    args = parser.parse_args()

    try:
        from torchvision.datasets import Places365
    except ImportError:
        raise SystemExit("Run:  pip install torch torchvision")

    # ── Download (skipped automatically if already present) ───────────────────
    print("Loading Places365 small train split (~24 GB download if not cached)…")
    dataset = Places365(
        root=args.download_dir,
        split="train-standard",
        small=True,       # 256×256; set False for original-resolution (~100 GB)
        download=True,
    )
    print(f"  Dataset loaded: {len(dataset):,} total images, "
          f"{len(dataset.classes)} classes.\n")

    # ── Validate selected classes ─────────────────────────────────────────────
    short_to_idx = build_short_to_idx(dataset)
    missing = [c for c in SELECTED_CLASSES if c not in short_to_idx]
    if missing:
        available_sample = sorted(short_to_idx.keys())[:40]
        raise ValueError(
            f"The following classes were not found in Places365:\n  {missing}\n"
            f"Sample of available class names:\n  {available_sample}"
        )

    selected_idx_set = {short_to_idx[c] for c in SELECTED_CLASSES}
    idx_to_name      = {short_to_idx[c]: c for c in SELECTED_CLASSES}

    # ── Create output directories ─────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    for cls_name in SELECTED_CLASSES:
        (output_dir / cls_name).mkdir(parents=True, exist_ok=True)

    # Count already-copied images to allow resuming
    already_saved = {
        cls_name: len(list((output_dir / cls_name).glob("*")))
        for cls_name in SELECTED_CLASSES
    }
    if any(v > 0 for v in already_saved.values()):
        print("Resuming — already saved:", {k: v for k, v in already_saved.items() if v > 0})

    # ── Copy matching images ──────────────────────────────────────────────────
    print("Scanning dataset and copying selected classes…")
    counts = {c: already_saved[c] for c in SELECTED_CLASSES}

    with tqdm(total=len(dataset), desc="Scanning", unit="img") as pbar:
        for raw_file, label in dataset.imgs:
            pbar.update(1)
            if label not in selected_idx_set:
                continue

            cls_name = idx_to_name[label]
            src = resolve_img_path(raw_file, dataset)
            dst = output_dir / cls_name / src.name

            if not dst.exists():
                shutil.copy(src, dst)

            counts[cls_name] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nDone. Images per class:")
    total = 0
    for cls_name in SELECTED_CLASSES:
        n = len(list((output_dir / cls_name).glob("*")))
        print(f"  {cls_name:<25s}  {n:>5d} images")
        total += n
    print(f"  {'TOTAL':>25s}  {total:>5d} images")
    print(f"\nSaved to: {output_dir.resolve()}")
    print(
        f"\nNext step — create a skewed dataset:\n"
        f"  python prepare_skewed_dataset.py "
        f"--data_dir {output_dir.resolve()} "
        f"--out_dir places365_skewed/train "
        f"--alpha 0.1 --size 13000 --min_per_class 200"
    )


if __name__ == "__main__":
    main()