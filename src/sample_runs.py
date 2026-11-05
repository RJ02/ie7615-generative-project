"""Milestone 1: five prompt -> image test runs with Stable Diffusion v1.5, scored with CLIP.

Three prompts are COCO captions from our cleaned set (so the generation can be compared with a real photo of the
same description) and two are new prompts outside COCO's style. Every run uses the same seed and settings.

Usage:  python src/sample_runs.py [--steps 25] [--guidance 7.5] [--seed 42]
Output: outputs/m1_samples/  (one PNG per prompt, grid.png, runs.json)
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "m1_samples"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
NEW_PROMPTS = ["A red vintage bicycle leaning against a brick wall in the rain",
               "A watercolor painting of a lighthouse at sunset"]


def clip_score(model, proc, image, text):
    x = proc(text=[text], images=image, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model(**x)
    t = torch.nn.functional.normalize(out.text_embeds, dim=-1)
    i = torch.nn.functional.normalize(out.image_embeds, dim=-1)
    return float(100 * (t * i).sum())                       # CLIPScore-style: 100 x cosine similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = pd.read_csv(ROOT / "data" / "processed" / "pairs.csv")
    coco = pairs.sample(3, random_state=7)
    runs = [{"prompt": r.caption, "source": "COCO caption", "coco_image": r.file, "image_id": int(r.image_id)}
            for r in coco.itertuples()] + [{"prompt": p, "source": "new prompt"} for p in NEW_PROMPTS]

    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", variant="fp16",
                                                   torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    OUT.mkdir(parents=True, exist_ok=True)
    for k, run in enumerate(runs, 1):
        g = torch.Generator("cpu").manual_seed(args.seed)
        t0 = time.time()
        img = pipe(run["prompt"], num_inference_steps=args.steps, guidance_scale=args.guidance,
                   height=512, width=512, generator=g).images[0]
        run.update(file=f"sample_{k}.png", seconds=round(time.time() - t0, 1),
                   clip_score=round(clip_score(clip, proc, img, run["prompt"]), 2), steps=args.steps,
                   guidance_scale=args.guidance, seed=args.seed, scheduler="DPMSolverMultistep", size=512)
        if "coco_image" in run:
            real = Image.open(ROOT / "data" / "processed" / run["coco_image"]).convert("RGB")
            run["clip_score_real_photo"] = round(clip_score(clip, proc, real, run["prompt"]), 2)
        img.save(OUT / run["file"])
        print(f"[{k}/5] {run['seconds']}s  CLIP {run['clip_score']}  {run['prompt']}")

    # contact sheet: generated image per prompt (with the real COCO photo above it where there is one)
    W = 300
    sheet = Image.new("RGB", (5 * W, 2 * W + 90), "white")
    d = ImageDraw.Draw(sheet)
    for k, run in enumerate(runs):
        if "coco_image" in run:
            sheet.paste(Image.open(ROOT / "data" / "processed" / run["coco_image"]).resize((W - 10, W - 10)), (k * W + 5, 5))
        else:
            d.text((k * W + 20, W // 2), "(no real photo:\nnew prompt)", fill=(120, 120, 120))
        sheet.paste(Image.open(OUT / run["file"]).resize((W - 10, W - 10)), (k * W + 5, W + 5))
        d.text((k * W + 8, 2 * W + 8), f"CLIP {run['clip_score']}", fill=(0, 0, 0))
    sheet.save(OUT / "grid.png")
    (OUT / "runs.json").write_text(json.dumps(runs, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
