import csv, os, time
import torch
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler,
                       EulerDiscreteScheduler, StableDiffusionPipeline)
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SCHEDULERS = {"dpm": DPMSolverMultistepScheduler, "ddim": DDIMScheduler, "euler": EulerDiscreteScheduler}

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def load_pipeline(device=None):
    device = device or get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe._base_sched_cfg = pipe.scheduler.config
    set_scheduler(pipe, "dpm")
    return pipe

def set_scheduler(pipe, name):
    pipe.scheduler = SCHEDULERS[name].from_config(pipe._base_sched_cfg)
    pipe._sched_name = name

@torch.no_grad()
def encode_prompt(pipe, text):
    ids = pipe.tokenizer(text, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(pipe.device)
    return pipe.text_encoder(ids)[0]          # (1, 77, 768)

def generate(pipe, prompt, scheduler="dpm", steps=25, w=7.5, seed=42, manual_embeds=True):
    if getattr(pipe, "_sched_name", None) != scheduler:
        set_scheduler(pipe, scheduler)
    gen = torch.Generator("cpu").manual_seed(seed)
    kw = dict(num_inference_steps=steps, guidance_scale=w, generator=gen, height=512, width=512)
    if manual_embeds:
        kw["prompt_embeds"] = encode_prompt(pipe, prompt)
        kw["negative_prompt_embeds"] = encode_prompt(pipe, "")
    else:
        kw["prompt"] = prompt
    if pipe.device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    img = pipe(**kw).images[0]
    if pipe.device.type == "cuda": torch.cuda.synchronize()
    return img, time.time() - t0

_clip = {}
def clip_score(image, text, device=None):
    device = device or get_device()
    if "m" not in _clip:
        _clip["m"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        _clip["p"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inp = _clip["p"](text=[text], images=image, return_tensors="pt", padding=True,
                     truncation=True, max_length=77).to(device)
    with torch.no_grad():
        out = _clip["m"](**inp)
        t = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        i = torch.nn.functional.normalize(out.image_embeds, dim=-1)
    return 100 * (t * i).sum(-1).item()

FIELDS = ["run","prompt_id","prompt","seed","scheduler","steps","w","seconds","clip_score","image_path"]

def run_batch(pipe, prompts, out_dir, log_csv, run_name, scheduler="dpm", steps=25, w=7.5, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    new = not os.path.exists(log_csv)
    images = []
    with open(log_csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new: wr.writeheader()
        for k, p in enumerate(prompts, 1):
            img, secs = generate(pipe, p, scheduler, steps, w, seed)
            path = os.path.join(out_dir, f"{run_name}_p{k:02d}.png")
            img.save(path)
            wr.writerow(dict(run=run_name, prompt_id=k, prompt=p, seed=seed, scheduler=scheduler,
                             steps=steps, w=w, seconds=round(secs,1),
                             clip_score=round(clip_score(img, p),2), image_path=path))
            f.flush(); images.append(img)
            print(f"[{run_name}] {k}/{len(prompts)} ({secs:.1f}s)")
    return images
