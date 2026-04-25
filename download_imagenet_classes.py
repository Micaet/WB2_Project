"""
Download the 10 Imagenette synsets from the ILSVRC/imagenet-1k dataset on
Hugging Face and save them to disk in torchvision ImageFolder format:

    imagenet/
    └── train/
        ├── n01440764/   (tench)
        ├── n02102040/   (English springer)
        └── ...

Prerequisites (run these in your terminal first)
------------------------------------------------
    pip install datasets huggingface_hub Pillow tqdm

    # Create a free account at huggingface.co, then accept the license at:
    # huggingface.co/datasets/ILSVRC/imagenet-1k
    # Then authenticate:
    huggingface-cli login

Usage
-----
    python download_imagenet_classes.py

Expected download size: ~1.5 GB  (≈1 300 images × 10 classes)
Expected time:          5–15 min depending on your connection
"""

import os
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("imagenet/train")

# (synset_id, short_name) — same 10 classes as the experiment
SYNSETS = [
    ("n01440764", "tench"),
    ("n02102040", "English springer"),
    ("n02979186", "cassette player"),
    ("n03000684", "chain saw"),
    ("n03028079", "church"),
    ("n03394916", "French horn"),
    ("n03417042", "garbage truck"),
    ("n03425413", "gas pump"),
    ("n03445777", "golf ball"),
    ("n03888257", "parachute"),
]

SYNSET_IDS = {s for s, _ in SYNSETS}

# ── Download ─────────────────────────────────────────────────────────────────

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run:  pip install datasets huggingface_hub Pillow tqdm")

    # Build synset → folder name mapping from the HF dataset's class labels.
    # HF imagenet-1k uses integer labels; we need to map those to synset IDs.
    print("Loading label mapping from HF metadata ...")
    # The feature info is available without downloading images
    ds_info = load_dataset(
        "ILSVRC/imagenet-1k",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    features = ds_info.features
    # features["label"].names is a list of 1000 synset IDs ordered by int label
    label_names: list[str] = features["label"].names   # e.g. ["n01440764", ...]
    target_int_labels = {
        i for i, name in enumerate(label_names) if name in SYNSET_IDS
    }
    int_to_synset = {i: label_names[i] for i in target_int_labels}

    print(f"Targeting {len(target_int_labels)} classes: "
          f"{[label_names[i] for i in sorted(target_int_labels)]}\n")

    # Create output directories
    for synset_id, _ in SYNSETS:
        (OUTPUT_DIR / synset_id).mkdir(parents=True, exist_ok=True)

    # Count already-saved images to allow resuming
    saved = {
        synset_id: len(list((OUTPUT_DIR / synset_id).glob("*.JPEG")))
        for synset_id, _ in SYNSETS
    }
    print("Already saved:", {k: v for k, v in saved.items() if v > 0} or "none")

    # Stream the training split and save matching images
    print("\nStreaming ImageNet train split (only saves 10 classes) ...")
    ds_stream = load_dataset(
        "ILSVRC/imagenet-1k",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    counters = dict(saved)
    with tqdm(desc="Images saved", unit="img") as pbar:
        for sample in ds_stream:
            int_label: int = sample["label"]
            if int_label not in target_int_labels:
                continue

            synset_id = int_to_synset[int_label]
            idx = counters[synset_id]
            out_path = OUTPUT_DIR / synset_id / f"{synset_id}_{idx:05d}.JPEG"

            if not out_path.exists():
                img: Image.Image = sample["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(out_path, format="JPEG", quality=95)

            counters[synset_id] += 1
            pbar.update(1)

    print("\nDone. Images per class:")
    total = 0
    for synset_id, name in SYNSETS:
        n = len(list((OUTPUT_DIR / synset_id).glob("*.JPEG")))
        print(f"  {synset_id}  ({name:<20s})  {n:>5d} images")
        total += n
    print(f"  {'TOTAL':>30s}  {total:>5d} images")
    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
