"""Step 1: split raw images into train / val / test folders WITHOUT data leakage.

Input layout (you create this):
    dataset_raw/
        rohu/        img001.jpg ...
        katla/       ...
        (one folder per species; folder name = class_id in species_info.csv)

Output layout:
    dataset/train/<class>/...   dataset/val/<class>/...   dataset/test/<class>/...

Leakage protection: exact duplicate files (same MD5 hash) are removed BEFORE
splitting, so the same photo can never be in both train and test.
IMPORTANT: also use only ORIGINAL photos here. If a downloaded dataset contains
pre-augmented copies of a photo, delete those copies, because near-identical
copies would still leak between train and test. (Augmentation is done
automatically during training instead.)

Run:  python prepare_dataset.py --raw dataset_raw --out dataset --overwrite
"""
import argparse
import hashlib
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def md5_of_file(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="dataset_raw", help="folder with one sub-folder per species")
    p.add_argument("--out", default="dataset", help="output folder")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="delete the output folder first if it exists")
    args = p.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    if not raw.is_dir():
        sys.exit(f"ERROR: raw folder '{raw}' not found.")
    class_dirs = sorted(d for d in raw.iterdir() if d.is_dir() and not d.name.startswith("."))
    if len(class_dirs) < 2:
        sys.exit("ERROR: need at least 2 species folders inside the raw folder.")

    if out.exists():
        if not args.overwrite:
            sys.exit(f"ERROR: '{out}' already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(out)

    rng = random.Random(args.seed)
    seen_hashes = {}          # hash -> first file path (across ALL classes)
    report = []

    for cdir in class_dirs:
        files = sorted(f for f in cdir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        unique, dupes = [], 0
        for f in files:
            try:
                h = md5_of_file(f)
            except OSError as e:
                print(f"  skipping unreadable file {f}: {e}")
                continue
            if h in seen_hashes:
                dupes += 1
                if seen_hashes[h].parent != f.parent:
                    print(f"  WARNING: {f} is identical to {seen_hashes[h]} (different class folder!)")
                continue
            seen_hashes[h] = f
            unique.append(f)

        rng.shuffle(unique)
        n = len(unique)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        if n >= 10:                       # make sure val and test are never empty
            n_train = min(n_train, n - 2)
            n_val = max(n_val, 1)
        splits = {
            "train": unique[:n_train],
            "val": unique[n_train:n_train + n_val],
            "test": unique[n_train + n_val:],
        }
        for split, items in splits.items():
            dest = out / split / cdir.name
            dest.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(items):
                shutil.copy2(f, dest / f"{cdir.name}_{i:04d}{f.suffix.lower()}")

        report.append({"class": cdir.name, "found": len(files), "duplicates_removed": dupes,
                       "train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"])})
        if n < 100:
            print(f"  NOTE: '{cdir.name}' has only {n} unique images - aim for 200+ for reliable results.")

    df = pd.DataFrame(report)
    print("\n", df.to_string(index=False))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "split_report.csv", index=False)
    print(f"\nDone. Dataset written to '{out}'.")


if __name__ == "__main__":
    main()
