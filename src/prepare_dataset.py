"""Clean and preprocess COCO 2017 val image-caption pairs for text-to-image experiments.

Steps
  1. Load the 5,000 val2017 images and their ~25,000 human captions (captions_val2017.json).
  2. Captions: normalise whitespace and punctuation, drop exact duplicates per image, drop captions that are too
     short (< 5 words) or too long for CLIP's 77-token context (> 40 words).
  3. Images: verify they open, convert to RGB (COCO has some greyscale images), drop very small images
     (short side < 256 px), then centre-crop to a square and resize to 512 x 512 (Stable Diffusion v1.5 resolution).
  4. Write data/processed/pairs.csv (one row per image: primary caption + all kept captions) and stats.json.

Usage:  python src/prepare_dataset.py [--n 5000] [--size 512]
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MIN_WORDS, MAX_WORDS, MIN_SIDE = 5, 40, 256


def clean_caption(text):
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace / newlines
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)          # no space before punctuation
    text = text.rstrip(" .")                                # drop trailing period(s)
    return text[:1].upper() + text[1:] if text else text


def center_crop_resize(img, size):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((size, size), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="max number of images to keep")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    ann = json.loads((RAW / "annotations" / "captions_val2017.json").read_text())
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    caps = defaultdict(list)
    for a in ann["annotations"]:
        caps[a["image_id"]].append(a["caption"])

    stats = Counter()
    stats["images_raw"], stats["captions_raw"] = len(files), len(ann["annotations"])
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id in sorted(files):
        if len(rows) >= args.n:
            break
        kept, seen = [], set()
        for c in caps[image_id]:
            c = clean_caption(c)
            n_words = len(c.split())
            if c.lower() in seen:
                stats["captions_duplicate"] += 1
            elif n_words < MIN_WORDS:
                stats["captions_too_short"] += 1
            elif n_words > MAX_WORDS:
                stats["captions_too_long"] += 1
            else:
                kept.append(c)
                seen.add(c.lower())
        if not kept:
            stats["images_no_caption"] += 1
            continue
        try:
            with Image.open(RAW / "val2017" / files[image_id]) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode != "RGB":
                    stats["images_converted_to_rgb"] += 1
                    im = im.convert("RGB")
                if min(im.size) < MIN_SIDE:
                    stats["images_too_small"] += 1
                    continue
                center_crop_resize(im, args.size).save(OUT / "images" / f"{image_id:012d}.jpg", quality=95)
        except Exception:
            stats["images_unreadable"] += 1
            continue
        primary = sorted(kept, key=len)[len(kept) // 2]     # median-length caption as the primary caption
        rows.append({"image_id": image_id, "file": f"images/{image_id:012d}.jpg", "caption": primary,
                     "n_captions": len(kept), "all_captions": json.dumps(kept)})
        stats["captions_kept"] += len(kept)

    with open(OUT / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lengths = [len(r["caption"].split()) for r in rows]
    stats["images_kept"] = len(rows)
    stats["primary_caption_words_mean"] = round(sum(lengths) / len(lengths), 2)
    (OUT / "stats.json").write_text(json.dumps(dict(stats), indent=2))
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
