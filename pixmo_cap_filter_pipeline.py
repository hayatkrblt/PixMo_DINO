"""
End-to-end PixMo-Cap noun-chunk filtering pipeline for HyCoCLIP.

Per image/caption:
    1. Extract noun chunks from the RAW caption (spaCy).
    2. Hard-filter chunks that unambiguously describe the annotation/
       medium rather than the scene (cheap set lookup, no model call).
    3. Run everything that survives step 2 through local Grounding DINO.
       Keep rule: grounded OR on the protected abstract-noun allowlist.

Ambiguous words (frame, poster, label, watermark, etc.) are deliberately
NOT hard-filtered in step 2 -- they're left for Grounding DINO in step 3
to resolve using the actual image, since a static list can't tell "a
picture frame on the wall" from "in this frame of the video."

Install:
    pip install transformers torch spacy pillow --break-system-packages
    python -m spacy download en_core_web_sm
"""

import re
import spacy
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

nlp = spacy.load("en_core_web_sm")

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()


# ------------------------------------------------------------------
# STEP 2 material: words/phrases that are NEVER real, visible content --
# safe to hard-filter with a plain lookup, no model call needed. This
# is the trimmed-down version of the earlier category sets, with every
# word we identified as context-dependent (frame, poster, label, tag,
# logo, banner, advertisement, watermark, collage, print, sign) removed
# and pushed into AMBIGUOUS_WORDS below instead.
# ------------------------------------------------------------------
HARD_FILTER_WORDS = {
    # hedging -- never a real object, always describes uncertainty
    "appears to be", "seems to be", "looks like", "it is unclear",
    "what appears to be", "what looks like", "seemingly", "appears",
    "seems", "likely", "possibly", "probably", "presumably", "perhaps",
    "maybe", "might be", "could be", "may be", "unclear", "uncertain",
    "suggesting", "suggests", "arguably", "conceivably", "apparently",

    # annotation/production meta -- never a real object
    "caption reads", "alt text", "labeled as", "tagged as",
    "described as", "titled", "captioned", "as shown", "as seen",
    "as depicted", "as pictured", "shown here", "seen here",
    "pictured here", "depicted here", "the following image",
    "above image", "below image", "image credit", "photo credit",
    "courtesy of", "source unknown", "stock photo", "stock image",
    "sample image", "example image", "generated image", "AI-generated",
    "rendered image", "caption"

    # true digital-UI chrome -- can't exist as content within a photo
    "sidebar", "toolbar", "menu", "url", "hyperlink", "text box",
    "dialogue box", "pop-up", "notification", "layout", "template",
    "interface", "webpage", "website", "taskbar", "scrollbar",
    "username", "handle",

    # image/medium self-reference -- refers to the artifact, not the scene
    "screenshot", "screen capture", "rendering", "render", "depiction",
    "photo shows", "image shows", "image depicts", "picture shows",
    "this image", "this photo", "this picture", "the photo", "the image",
    "the picture",

    # photography/quality descriptors -- describe the capture, not a thing
    "blurry", "blurred", "grainy", "grain", "pixelated", "pixelation",
    "out of focus", "in focus", "low resolution", "high resolution",
    "low quality", "high quality", "overexposed", "underexposed",
    "lens flare", "motion blur", "compression artifact",
    "black and white", "grayscale", "sepia toned", "monochrome",
    "vignette", "cropped image", "zoomed in", "zoomed out",

    # visual style descriptors -- describe rendering style, not an object
    "cartoon", "cartoonish", "animated", "anime style", "realistic",
    "photorealistic", "stylized", "digital art", "3D rendered",
    "computer-generated", "cinematic", "filtered", "filter applied", "hdr",

    # framing/composition language -- describes the shot, not an object
    "foreground", "background", "midground", "close-up", "wide shot",
    "overhead", "aerial view", "bird's-eye view", "cropped", "off-center",
    "in frame", "out of frame", "in view", "partially visible",
    "partially obscured",

    # approximation -- modifies a count, isn't itself an object
    "approximately", "roughly", "give or take", "estimated",
    "an estimated",
}

# Words that CAN go either way -- deliberately excluded from
# HARD_FILTER_WORDS. Left untouched in step 2; resolved by Grounding
# DINO in step 3 using the actual image.
AMBIGUOUS_WORDS = {
    "frame", "poster", "print", "label", "tag", "logo", "banner",
    "advertisement", "watermark", "collage", "sign", "backdrop",
}

_hard_pattern = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(HARD_FILTER_WORDS, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Protected abstract/semantic nouns -- never removed just because they
# don't get a Grounding DINO box. See earlier message for the full
# maximal version;
CONCEPTUAL_ALLOWLIST = {
    # emotions
    "love", "joy", "happiness", "sadness", "fear", "excitement",
    "anxiety", "calm", "peace", "nostalgia", "pride", "curiosity",
    "wonder", "awe", "melancholy", "frustration", "relief", "hope",
    "contentment", "gratitude", "affection", "loneliness", "comfort",

    # social / relational
    "family", "friendship", "community", "teamwork", "cooperation",
    "unity", "togetherness", "connection", "trust", "respect",
    "romance", "intimacy", "belonging", "companionship",

    # abstract activities
    "performance", "celebration", "tradition", "negotiation",
    "collaboration", "trade", "business", "commerce", "development",
    "progress", "achievement", "success", "struggle", "recovery",

    # values / qualities
    "freedom", "justice", "equality", "faith", "honor", "dignity",
    "wisdom", "courage", "determination", "energy", "atmosphere", "mood",

    # domains
    "healthcare", "education", "culture", "religion", "politics",
    "history", "heritage", "memory",
}


def extract_noun_chunks(caption: str) -> list[str]:
    """Noun chunks from the RAW caption -- do not pre-clean the text first."""
    doc = nlp(caption)
    return [chunk.text.strip() for chunk in doc.noun_chunks]


def hard_filter(chunks: list[str]) -> tuple[list[str], list[str]]:
    """
    STEP 2: drop any chunk that matches a HARD_FILTER_WORDS phrase,
    UNLESS that match is only because of an AMBIGUOUS_WORDS term inside
    it (those are deliberately left for step 3 to resolve).

    Returns (survivors, hard_removed).
    """
    survivors, removed = [], []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        is_hard_match = bool(_hard_pattern.search(chunk_lower))
        # If the ONLY reason it would match is an ambiguous word used as
        # a substring inside a hard-filter phrase, this check is skipped
        # in practice since AMBIGUOUS_WORDS and HARD_FILTER_WORDS don't
        # overlap by construction -- kept as an explicit safety check
        # in case the sets are edited later.
        is_ambiguous_only = any(w in chunk_lower.split() for w in AMBIGUOUS_WORDS) \
            and not any(w in chunk_lower for w in HARD_FILTER_WORDS if w not in AMBIGUOUS_WORDS)
        if is_hard_match and not is_ambiguous_only:
            removed.append(chunk)
        else:
            survivors.append(chunk)
    return survivors, removed


@torch.no_grad()
def ground_phrases(
    image: Image.Image,
    phrases: list[str],
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> set[str]:
    """STEP 3 core: query Grounding DINO with all surviving chunks for
    one image in a single batched forward pass."""
    text_prompt = ". ".join(phrases) + "."
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    grounded_raw = {label.lower().strip() for label in results["labels"]}
    grounded_phrases = set()
    for phrase in phrases:
        norm = phrase.lower().strip()
        if any(norm in g or g in norm for g in grounded_raw):
            grounded_phrases.add(phrase)
    return grounded_phrases


def filter_caption(image_path: str, caption: str) -> dict:
    """
    Full pipeline for one (image, caption) pair. Returns a dict with
    kept chunks and removed chunks broken out BY REASON, so you can
    audit each stage separately during the validation pass.
    """
    all_chunks = extract_noun_chunks(caption)
    if not all_chunks:
        return {"kept": [], "removed_wordlist": [], "removed_ungrounded": []}

    survivors, removed_wordlist = hard_filter(all_chunks)
    if not survivors:
        return {"kept": [], "removed_wordlist": removed_wordlist, "removed_ungrounded": []}

    image = Image.open(image_path).convert("RGB")
    grounded = ground_phrases(image, survivors)

    kept, removed_ungrounded = [], []
    for chunk in survivors:
        chunk_lower = chunk.lower().strip()
        is_protected = any(w in chunk_lower.split() for w in CONCEPTUAL_ALLOWLIST) \
            or chunk_lower in CONCEPTUAL_ALLOWLIST
        if chunk in grounded or is_protected:
            kept.append(chunk)
        else:
            removed_ungrounded.append(chunk)

    return {
        "kept": kept,
        "removed_wordlist": removed_wordlist,
        "removed_ungrounded": removed_ungrounded,
    }


def batch_filter(records: list[dict]) -> list[dict]:
    """Entry point for a validation-sample or full-dataset run.
    records: list of {"image_path": ..., "caption": ...}."""
    output = []
    for record in records:
        result = filter_caption(record["image_path"], record["caption"])
        output.append({**record, **result})
    return output


if __name__ == "__main__":
    sample_records = [
        {
            "image_path": "example.jpg",
            "caption": (
                "This image shows a blurry photo of a man standing near a "
                "market stall with roughly a dozen fruits displayed, "
                "radiating a sense of satisfaction and community trade. "
                "In the background, there is a watermark in the "
                "bottom-right corner."
            ),
        }
    ]

    for r in batch_filter(sample_records):
        print(f"Caption: {r['caption']}\n")
        print(f"  Kept:               {r['kept']}")
        print(f"  Removed (wordlist): {r['removed_wordlist']}")
        print(f"  Removed (ungrounded): {r['removed_ungrounded']}")
