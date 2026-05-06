from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

OUTPUT_ROOT = Path("imagenet")
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
    ("n01491361", "tiger shark"),
    ("n01514859", "hen"),
    ("n01608432", "kite"),
    ("n02123045", "tabby cat"),
    ("n02504458", "African elephant"),
    ("n03085013", "computer keyboard"),
    ("n01630670", "common newt"),
    ("n03792782", "mountain bike"),
    ("n04285008", "sports car"),
    ("n04507155", "umbrella"),
]
SYNSET_IDS = {s for s, _ in SYNSETS}

def main():
    print("Loading label mapping from HF metadata...")
    ds_info = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
    label_names = ds_info.features["label"].names
    
    target_int_labels = {i for i, name in enumerate(label_names) if name in SYNSET_IDS}
    int_to_synset = {i: label_names[i] for i in target_int_labels}

    splits = {"train": "train", "validation": "val", "test": "test"}

    for hf_split, out_split in splits.items():
        print(f"\n--- Processing split: {hf_split} -> {out_split} ---")
        split_dir = OUTPUT_ROOT / out_split
        
        for synset_id, _ in SYNSETS:
            (split_dir / synset_id).mkdir(parents=True, exist_ok=True)
            
        ds_stream = load_dataset("ILSVRC/imagenet-1k", split=hf_split, streaming=True, trust_remote_code=True)
        
        saved = {synset_id: len(list((split_dir / synset_id).glob("*.JPEG"))) for synset_id in SYNSET_IDS}
        
        with tqdm(desc=f"Downloading {hf_split}", unit="img") as pbar:
            for sample in ds_stream:
                int_label = sample.get("label", -1)
                if int_label not in target_int_labels:
                    continue
                
                synset_id = int_to_synset[int_label]
                idx = saved[synset_id]
                out_path = split_dir / synset_id / f"{synset_id}_{hf_split}_{idx:05d}.JPEG"
                
                if not out_path.exists():
                    img = sample["image"]
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(out_path, format="JPEG", quality=95)
                    
                saved[synset_id] += 1
                pbar.update(1)

if __name__ == "__main__":
    main()