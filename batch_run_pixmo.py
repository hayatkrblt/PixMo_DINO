"""
Batch runner for pixmo_cap_filter_pipeline.filter_caption() over a real
PixMo-Cap manifest, meant to run ON the machine where the images live
(e.g. via `ssh yourserver`, then run this script there directly --
much simpler than trying to stream remote images over the network).

What this adds on top of your snippet:
  1. Reads records from a file instead of a hardcoded list, so it scales
     to the whole dataset.
  2. RESUME support: if the job crashes or you disconnect, rerunning
     this script skips everything already processed instead of
     starting over.
  3. Writes results one line at a time (not held in memory, not lost
     if the job dies halfway through 712k images).
  4. try/except per record: one corrupt image or missing file doesn't
     kill the whole run.
  5. Progress bar so you can see it's alive.

Usage:
    python3 batch_run_pixmo.py --manifest pixmo_manifest.jsonl --output r[insert number].jsonl (I went from 1 to 3)
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from pixmo_cap_filter_pipeline import filter_caption


def load_manifest(manifest_path: str):
    """
    Reads a JSONL file where each line is: {"image_path": ..., "caption": ...}

    If your PixMo-Cap data lives in a different format (e.g. a HuggingFace
    `datasets` parquet file, or a CSV), convert it to this JSONL shape
    first -- one small one-off script, much simpler than teaching every
    downstream tool a new format. Example conversion if you have a HF
    dataset object called `ds`:

        with open("pixmo_manifest.jsonl", "w") as f:
            for row in ds:
                f.write(json.dumps({
                    "image_path": row["image_path"],
                    "caption": row["caption"],
                }) + "\\n")
    """
    records = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_already_done(output_path: str) -> set:
    """
    Reads the output file (if it exists from a previous run) and
    returns the set of image_paths already processed, so we can skip
    them this time. This is what makes the job resumable -- kill it,
    rerun the exact same command, it picks up where it left off.
    """
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["image_path"])
            except json.JSONDecodeError:
                # Last line of a previous run may be cut off mid-write
                # if the job was killed at exactly the wrong moment --
                # ignore it, that record will just get reprocessed.
                continue
    return done


def run_batch(manifest_path: str, output_path: str, limit: int = None):
    records = load_manifest(manifest_path)
    already_done = load_already_done(output_path)

    if already_done:
        print(f"Resuming: {len(already_done)} images already processed, skipping those.")

    remaining = [r for r in records if r["image_path"] not in already_done]
    if limit:
        remaining = remaining[:limit]

    print(f"Processing {len(remaining)} images...")

    error_count = 0
    # Open in APPEND mode, and write+flush after every single record --
    # this is what guarantees you never lose progress even if the
    # process is killed without warning (out of memory, ssh drop, etc.)
    with open(output_path, "a") as out_f:
        for record in tqdm(remaining):
            image_path = record["image_path"]

            if not Path(image_path).exists():
                error_count += 1
                out_f.write(json.dumps({
                    "image_path": image_path,
                    "caption": record["caption"],
                    "error": "image_file_not_found",
                }) + "\n")
                out_f.flush()
                continue

            try:
                result = filter_caption(image_path, record["caption"])
                out_f.write(json.dumps({
                    "image_path": image_path,
                    "caption": record["caption"],
                    **result,
                }) + "\n")
            except Exception as e:
                # Catch-all so one bad image (corrupt file, decode
                # error, unexpected model output) doesn't kill a job
                # that might otherwise run for many hours unattended.
                error_count += 1
                out_f.write(json.dumps({
                    "image_path": image_path,
                    "caption": record["caption"],
                    "error": str(e),
                }) + "\n")
            out_f.flush()

    print(f"Done. {error_count} errors out of {len(remaining)} images. See {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest file")
    parser.add_argument("--output", required=True, help="Path to write JSONL results")
    parser.add_argument("--limit", type=int, default=None, help="Only process N images (for testing)")
    args = parser.parse_args()

    run_batch(args.manifest, args.output, limit=args.limit)
