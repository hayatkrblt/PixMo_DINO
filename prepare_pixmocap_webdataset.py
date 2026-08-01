#!/usr/bin/env python3
"""
Takes a JSONL dataset containing 'image_path', 'caption', and 'kept' text phrases,
runs Grounding DINO to find bounding boxes, and packages cropped parent images 
into WebDataset .tar shards.

CUDA_VISIBLE_DEVICES=1 /opt/miniconda/envs/openclip_ft/bin/python prepare_pixmocap_webdataset.py \
    --jsonl_path r[insert number of the batch>3].jsonl \
    --output_dir /media/pinas/datasets/PixMoCap/output_shards/part[insert number of the batch] \
    --shard_prefix shard_p[insert number of the batch] \
    --samples_per_tar 1000 
"""

import argparse
import io
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, GroundingDinoForObjectDetection

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def deduplicate_phrases(kept_raw: List[str]) -> List[str]:
    """Removes duplicate phrases (case-insensitive) while preserving order."""
    seen = set()
    unique_kept = []
    for phrase in kept_raw:
        cleaned = phrase.strip()
        norm = cleaned.lower()
        if norm and norm not in seen:
            seen.add(norm)
            unique_kept.append(cleaned)
    return unique_kept


def synthesize_fluent_caption(kept_phrases: List[str], raw_fallback: str) -> str:
    """Synthesizes a fluent, natural English sentence for child.txt."""
    if not kept_phrases:
        return raw_fallback

    phrases = [p.rstrip(" .") for p in kept_phrases]

    if len(phrases) == 1:
        phrase = phrases[0]
        prefix = "Photo of " if phrase.lower().startswith(("a ", "an ", "the ")) else "A photo of "
        return f"{prefix}{phrase}."

    if len(phrases) == 2:
        return f"A photo featuring {phrases[0]} and {phrases[1]}."

    body = ", ".join(phrases[:-1])
    return f"A photo featuring {body}, and {phrases[-1]}."


class PhraseGrounder:
    """Handles Grounding DINO object detection on CUDA."""

    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        logging.info(f"Loading Grounding DINO ({model_id}) on device: {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def ground_phrases(self, image: Image.Image, phrases: List[str]) -> Dict[str, List[int]]:
        if not phrases:
            return {}

        text_prompt = ". ".join(phrases) + "."

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            box_threshold=0.28,
            text_threshold=0.22,
            target_sizes=[image.size[::-1]],
        )[0]

        scores = results["scores"].cpu().numpy()
        labels = [l.lower().strip() for l in results["labels"]]
        boxes = results["boxes"].cpu().numpy()

        grounded_boxes = {}
        for phrase in phrases:
            norm_phrase = phrase.lower().strip()
            best_score = -1.0
            best_box = None

            for score, label, box in zip(scores, labels, boxes):
                if norm_phrase in label or label in norm_phrase:
                    if score > best_score:
                        best_score = score
                        best_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

            if best_box is not None:
                grounded_boxes[phrase] = best_box

        return grounded_boxes


def process_jsonl_to_shards(args):
    os.makedirs(args.output_dir, exist_ok=True)
    grounder = PhraseGrounder(model_id=args.model_id, device=args.device)

    logging.info(f"Loading input JSONL from: '{args.jsonl_path}'...")
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    logging.info(f"Loaded {len(records)} entries. Beginning WebDataset shard generation...")

    shard_idx = 0
    current_tar = None
    valid_count = 0
    skipped_count = 0

    for record in tqdm(records, desc="Building WebDataset Shards"):
        image_path = record.get("image_path")
        raw_caption = record.get("caption", "")
        kept_raw = record.get("kept", [])

        if not image_path or not Path(image_path).exists():
            skipped_count += 1
            continue

        try:
            raw_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.warning(f"Could not open image '{image_path}': {e}")
            skipped_count += 1
            continue

        kept_phrases = deduplicate_phrases(kept_raw)
        synthesized_caption = synthesize_fluent_caption(kept_phrases, raw_fallback=raw_caption)
        boxes_dict = grounder.ground_phrases(raw_image, kept_phrases)

        if valid_count % args.samples_per_tar == 0:
            if current_tar:
                current_tar.close()
            tar_name = f"{args.shard_prefix}_{shard_idx:05d}.tar"
            tar_path = os.path.join(args.output_dir, tar_name)
            current_tar = tarfile.open(tar_path, "w")
            shard_idx += 1

        sample_key = f"{valid_count:09d}"

        # Write child.jpg
        child_bytes_io = io.BytesIO()
        raw_image.save(child_bytes_io, format="JPEG", quality=95)
        child_bytes = child_bytes_io.getvalue()

        img_info = tarfile.TarInfo(name=f"{sample_key}.child.jpg")
        img_info.size = len(child_bytes)
        current_tar.addfile(img_info, io.BytesIO(child_bytes))

        # Write child.txt (Synthesized sentence)
        cap_bytes = synthesized_caption.encode("utf-8")
        cap_info = tarfile.TarInfo(name=f"{sample_key}.child.txt")
        cap_info.size = len(cap_bytes)
        current_tar.addfile(cap_info, io.BytesIO(cap_bytes))

        # Write numparents.txt
        num_bytes = str(len(kept_phrases)).encode("utf-8")
        num_info = tarfile.TarInfo(name=f"{sample_key}.numparents.txt")
        num_info.size = len(num_bytes)
        current_tar.addfile(num_info, io.BytesIO(num_bytes))

        # Write parent entries
        for p_idx, phrase in enumerate(kept_phrases):
            parent_prefix = f"parent{p_idx:03d}"

            ptxt_bytes = phrase.encode("utf-8")
            ptxt_info = tarfile.TarInfo(name=f"{sample_key}.{parent_prefix}.txt")
            ptxt_info.size = len(ptxt_bytes)
            current_tar.addfile(ptxt_info, io.BytesIO(ptxt_bytes))

            box = boxes_dict.get(phrase)
            if box and len(box) == 4:
                xmin, ymin, xmax, ymax = box
                width, height = raw_image.size

                xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
                xmax, ymax = min(width, int(xmax)), min(height, int(ymax))

                if (xmax - xmin) > 10 and (ymax - ymin) > 10:
                    cropped_img = raw_image.crop((xmin, ymin, xmax, ymax))
                    crop_io = io.BytesIO()
                    cropped_img.save(crop_io, format="JPEG", quality=95)
                    parent_bytes = crop_io.getvalue()
                else:
                    parent_bytes = child_bytes
            else:
                parent_bytes = child_bytes

            pimg_info = tarfile.TarInfo(name=f"{sample_key}.{parent_prefix}.jpg")
            pimg_info.size = len(parent_bytes)
            current_tar.addfile(pimg_info, io.BytesIO(parent_bytes))

        valid_count += 1

    if current_tar:
        current_tar.close()

    logging.info(f"Done! Saved {valid_count} samples into {shard_idx} shards in '{args.output_dir}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSONL into HyCoCLIP WebDataset TAR Shards")
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--shard_prefix", type=str, default="pixmocap_shard")
    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--samples_per_tar", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_jsonl_to_shards(args)
