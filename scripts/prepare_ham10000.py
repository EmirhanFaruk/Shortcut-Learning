"""
prepare_ham10000.py
-------------------
Turn the raw HAM10000 download into the binary folder layout train.py expects:

    out/
      train/{benign,malignant}/*.jpg
      val/{benign,malignant}/*.jpg
      test/{benign,malignant}/*.jpg

Binary mapping (standard):
    malignant = {mel, bcc, akiec}
    benign    = {nv, bkl, df, vasc}

Splitting is done by LESION id (not image), so multiple images of the same
lesion never straddle train/test -- a real leakage trap worth getting right and
worth a sentence in your write-up.

Usage:
    python scripts/prepare_ham10000.py \
        --images-dir /path/to/HAM10000_images \
        --metadata   /path/to/HAM10000_metadata.csv \
        --out        data/ham_binary
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

MALIGNANT = {"mel", "bcc", "akiec"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="HAM10000",
                    help="folder containing the HAM10000 jpgs (searched recursively)")
    ap.add_argument("--metadata", default="HAM10000/HAM10000_metadata.csv")
    ap.add_argument("--out", default="data/ham_binary")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--link", action="store_true", help="symlink instead of copy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    img_index = {p.stem: p for p in images_dir.rglob("*.jpg")}

    # group image_ids by lesion_id, and record label per image
    lesions = defaultdict(list)
    label_of = {}
    with open(args.metadata) as f:
        for row in csv.DictReader(f):
            iid, lid, dx = row["image_id"], row["lesion_id"], row["dx"]
            if iid not in img_index:
                continue
            lesions[lid].append(iid)
            label_of[iid] = "malignant" if dx in MALIGNANT else "benign"

    lesion_ids = list(lesions)
    random.Random(args.seed).shuffle(lesion_ids)
    n = len(lesion_ids)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    split_of = {}
    for i, lid in enumerate(lesion_ids):
        split_of[lid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    counts = defaultdict(int)
    for lid, iids in lesions.items():
        split = split_of[lid]
        for iid in iids:
            label = label_of[iid]
            dst_dir = Path(args.out) / split / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{iid}.jpg"
            src = img_index[iid]
            if args.link:
                if not dst.exists():
                    dst.symlink_to(src.resolve())
            else:
                shutil.copy(src, dst)
            counts[(split, label)] += 1

    print("Done. Image counts:")
    for split in ["train", "val", "test"]:
        b = counts[(split, "benign")]
        m = counts[(split, "malignant")]
        print(f"  {split:5s}  benign={b:5d}  malignant={m:5d}")


if __name__ == "__main__":
    main()
