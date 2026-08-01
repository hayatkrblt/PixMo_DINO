"""
Extracts the first N image-caption pairs out of img2dataset-format .tar
shards (confirmed format for this PixMo-Cap download: each shard has
NNNNNNNNN.jpg + NNNNNNNNN.txt + NNNNNNNNN.json per sample, where the
.txt is already the plain caption text -- no JSON parsing needed).

Usage:
    python3 extract_from_tars.py --tar-dir /media/pinas/datasets/PixMoCap/img2dataset_raw \
        --image-dir /media/pinas/datasets/PixMoCap/img2dataset_raw/extracted_images \
        --manifest pixmo_manifest.jsonl \
        --limit 20000
Usage (next batch, e.g.pairs 40000-60000):
    python3 extract_from_tars.py --tar-dir /media/pinas/datasets/PixMoCap/img2dataset_raw \
        --image-dir /media/pinas/datasets/karabulut/extracted_images[the number of the batch] \
        --manifest pixmo_manifest_part[the number of the batch].jsonl \
        --skip 40000 \
        --limit 20000
"""

import argparse
import json
import tarfile
from collections import defaultdict
from pathlib import Path

IMAGE_EXT = ".jpg"
CAPTION_EXT = ".txt"


def extract(tar_dir: str, image_dir: str, manifest_path: str, limit: int, skip: int):
    tar_dir = Path(tar_dir)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(tar_dir.glob("*.tar"))
    print(f"Found {len(tar_files)} tar shards.")
    seen = 0      # total valid pairs encountered so far (across all shards)
    written = 0
    with open(manifest_path, "w") as manifest_f:
        for tar_path in tar_files:
            if written >= limit:
                break

            print(f"Reading {tar_path.name} ({written}/{limit} pairs so far)...")
            with tarfile.open(tar_path, "r") as tf:
                # Group members by shared basename (the 9-digit key, e.g.
                # "000000020"), since each sample has a .jpg and a .txt
                # (and a .json we don't need) sharing that same stem.
                members_by_stem = defaultdict(dict)
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    path = Path(member.name)
                    stem, ext = path.stem, path.suffix.lower()
                    if ext == IMAGE_EXT:
                        members_by_stem[stem]["image"] = member
                    elif ext == CAPTION_EXT:
                        members_by_stem[stem]["caption"] = member

                for stem, pair in members_by_stem.items():
                    if written >= limit:
                        break
                    if "image" not in pair or "caption" not in pair:
                        # Incomplete pair -- skip rather than guess.
                        continue

                    if seen < skip: #necessary for generalising for the nth batch of 20k images (e.g. if you want to extract 20k-40k, you need to skip the first 20k)
                        seen += 1
                        continue
                    seen += 1

                    image_member = pair["image"]
                    caption_member = pair["caption"]

                    # Extract image bytes to disk. Prefixed with the shard
                    # name so keys don't collide across different shards.
                    out_image_path = image_dir / f"{tar_path.stem}_{stem}{IMAGE_EXT}"
                    with tf.extractfile(image_member) as img_f:
                        out_image_path.write_bytes(img_f.read())

                    # .txt is already plain caption text -- just decode it.
                    with tf.extractfile(caption_member) as cap_f:
                        caption_text = cap_f.read().decode("utf-8").strip()

                    manifest_f.write(json.dumps({
                        "image_path": str(out_image_path),
                        "caption": caption_text,
                    }) + "\n")
                    written += 1

    print(f"Done. Wrote {written} image-caption pairs to {manifest_path}")
    print(f"Images extracted to {image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar-dir", required=True, help="Directory containing .tar shards")
    parser.add_argument("--image-dir", required=True, help="Where to write extracted images")
    parser.add_argument("--manifest", required=True, help="Output manifest.jsonl path")
    parser.add_argument("--limit", type=int, default=20000, help="Number of pairs to extract")
    parser.add_argument("--skip", type=int, default=0, help="Number of valid pairs to skip before extracting (for resuming)")
    args = parser.parse_args()

    extract(args.tar_dir, args.image_dir, args.manifest, args.limit, args.skip)
