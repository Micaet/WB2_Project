import warnings
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

OUTPUT_ROOT = Path("imagenet")

TARGET_LABELS = {
    0: "n01440764", 3: "n01491361", 8: "n01514859", 21: "n01608432", 
    26: "n01630670", 217: "n02102040", 281: "n02123045", 386: "n02504458", 
    480: "n02979186", 491: "n03000684", 497: "n03028079", 508: "n03085013", 
    566: "n03394916", 569: "n03417042", 571: "n03425413", 574: "n03445777", 
    671: "n03792782", 701: "n03888257", 817: "n04285008", 879: "n04507155"
}

def main():
    splits = {"train": "train", "validation": "val", "test": "test"}

    for hf_split, out_split in splits.items():
        print(f"\n--- Processing split: {hf_split} -> {out_split} ---")
        split_dir = OUTPUT_ROOT / out_split
        
        for synset_id in TARGET_LABELS.values():
            (split_dir / synset_id).mkdir(parents=True, exist_ok=True)
            
        ds_stream = load_dataset("ILSVRC/imagenet-1k", split=hf_split, streaming=True)
        saved = {synset_id: len(list((split_dir / synset_id).glob("*.JPEG"))) for synset_id in TARGET_LABELS.values()}
        
        with tqdm(desc=f"Downloading {hf_split}", unit="img") as pbar:
            for sample in ds_stream:
                int_label = sample.get("label", -1)
                
                if int_label not in TARGET_LABELS:
                    continue
                
                synset_id = TARGET_LABELS[int_label]
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