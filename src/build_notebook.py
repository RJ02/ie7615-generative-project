"""Builds notebooks/01_text_embedding_baseline.ipynb (run: python src/build_notebook.py)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Milestone 1: Text → embedding baseline

**IE 7615 Generative Project, Study Group 6**

This notebook builds the text side of our text-to-image pipeline on the cleaned COCO 2017 val pairs
(`data/processed/pairs.csv`, produced by `src/prepare_dataset.py`):

1. Caption statistics and CLIP tokenisation (77-token context).
2. **CLIP ViT-B/32** text and image embeddings for every pair, saved for later evaluation (CLIP score, retrieval).
3. Sanity check that the embeddings are aligned: matched vs. mismatched cosine similarity and text→image retrieval.
4. The **Stable Diffusion v1.5 conditioning**: how a prompt becomes the 77 × 768 CLIP ViT-L/14 embedding that the
   U-Net attends to, and the empty-prompt embedding used for classifier-free guidance."""),
code("""import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPTextModel

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA = ROOT / "data" / "processed"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
pairs = pd.read_csv(DATA / "pairs.csv")
stats = json.loads((DATA / "stats.json").read_text())
print(f"{len(pairs):,} images, {stats['captions_kept']:,} captions, device = {DEVICE}")
pairs.head()"""),
md("## 1. Captions and tokenisation"),
code("""clip_id = "openai/clip-vit-base-patch32"
tok = CLIPTokenizer.from_pretrained(clip_id)
all_caps = [c for cs in pairs.all_captions.map(json.loads) for c in cs]
n_tokens = np.array([len(tok(c).input_ids) for c in all_caps])
words = np.array([len(c.split()) for c in all_caps])
print(f"words per caption: mean {words.mean():.1f}, max {words.max()}")
print(f"CLIP tokens per caption: mean {n_tokens.mean():.1f}, max {n_tokens.max()}, "
      f"truncated at 77: {(n_tokens > 77).sum()}")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].hist(words, bins=range(4, 32), color="#3b6ea5"); ax[0].set_title("Words per caption")
ax[1].hist(n_tokens, bins=range(5, 40), color="#3b6ea5"); ax[1].set_title("CLIP tokens per caption (incl. start/end)")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.show()"""),
code("""example = pairs.caption.iloc[0]
ids = tok(example).input_ids
print(example)
print(ids)
print(tok.convert_ids_to_tokens(ids))"""),
md("## 2. CLIP ViT-B/32 text and image embeddings"),
code("""model = CLIPModel.from_pretrained(clip_id).to(DEVICE).eval()
proc = CLIPProcessor.from_pretrained(clip_id)

@torch.no_grad()
def embed_text(texts, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        t = proc(text=texts[i:i + bs], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        feats = model.text_projection(model.text_model(input_ids=t.input_ids, attention_mask=t.attention_mask).pooler_output)
        out.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def embed_images(paths, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(DATA / p).convert("RGB") for p in paths[i:i + bs]]
        x = proc(images=ims, return_tensors="pt").to(DEVICE)
        feats = model.visual_projection(model.vision_model(pixel_values=x.pixel_values).pooler_output)
        out.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
    return torch.cat(out).numpy()

T = embed_text(pairs.caption.tolist())
I = embed_images(pairs.file.tolist())
np.savez_compressed(DATA / "clip_vitb32_embeddings.npz", text=T, image=I, image_id=pairs.image_id.values)
print("text", T.shape, "image", I.shape)"""),
md("## 3. Are text and image embeddings aligned?"),
code("""matched = (T * I).sum(1)
rng = np.random.default_rng(0)
mismatched = (T * I[rng.permutation(len(I))]).sum(1)
print(f"cosine similarity  matched {matched.mean():.3f} ± {matched.std():.3f}   "
      f"random pairs {mismatched.mean():.3f} ± {mismatched.std():.3f}")
plt.figure(figsize=(6, 3.2))
plt.hist(mismatched, bins=60, alpha=.7, label="random caption", color="#9aa3ae")
plt.hist(matched, bins=60, alpha=.8, label="true caption", color="#3b6ea5")
plt.xlabel("CLIP cosine similarity"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()"""),
code("""def recall_at_k(T, I, ks=(1, 5, 10)):
    sims = T @ I.T                              # caption i should retrieve image i
    rank = (sims > sims[np.arange(len(T)), np.arange(len(T))][:, None]).sum(1)
    return {f"R@{k}": float((rank < k).mean()) for k in ks}

r1k = recall_at_k(T[:1000], I[:1000]); rall = recall_at_k(T, I)
print("text → image retrieval, 1,000 pairs:", {k: round(v, 3) for k, v in r1k.items()})
print(f"text → image retrieval, {len(T):,} pairs:", {k: round(v, 3) for k, v in rall.items()})
json.dump({"cos_matched": float(matched.mean()), "cos_random": float(mismatched.mean()),
           "retrieval_1k": r1k, "retrieval_all": rall}, open(DATA / "embedding_baseline.json", "w"), indent=2)"""),
code("""q = ["two dogs playing in the grass", "a plate of food with broccoli", "a train at a station", "a man riding a surfboard"]
qe = embed_text(q)
fig, axes = plt.subplots(len(q), 4, figsize=(9, 2.4 * len(q)))
for r, (text, e) in enumerate(zip(q, qe)):
    top = np.argsort(-(I @ e))[:4]
    for c, j in enumerate(top):
        axes[r, c].imshow(Image.open(DATA / pairs.file.iloc[j])); axes[r, c].axis("off")
        axes[r, c].set_title(f"{(I[j] @ e):.2f}", fontsize=8)
    axes[r, 0].text(-20, 112, text, ha="right", va="center", fontsize=9, wrap=True)
plt.suptitle("Free-text queries → nearest COCO images (CLIP cosine)", y=1.0); plt.tight_layout(); plt.show()"""),
md("## 4. Stable Diffusion v1.5 text conditioning (CLIP ViT-L/14)"),
code("""sd_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
sd_tok = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
sd_enc = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16).to(DEVICE)

@torch.no_grad()
def sd_condition(prompt):
    ids = sd_tok(prompt, padding="max_length", max_length=sd_tok.model_max_length, truncation=True, return_tensors="pt").input_ids
    return sd_enc(ids.to(DEVICE)).last_hidden_state          # (1, 77, 768): one vector per token, fed to U-Net cross-attention

cond, uncond = sd_condition(pairs.caption.iloc[0]), sd_condition("")
print("conditioning embedding:", tuple(cond.shape), "| unconditional (empty prompt, for CFG):", tuple(uncond.shape))
print("classifier-free guidance:  eps = eps(uncond) + w · (eps(cond) − eps(uncond)),  w = guidance scale")
diff = (cond - uncond).float().norm(dim=-1)[0].cpu()
plt.figure(figsize=(7, 2.6)); plt.bar(range(77), diff, color="#3b6ea5")
plt.title("How much each token position differs from the empty prompt"); plt.xlabel("token position"); plt.tight_layout(); plt.show()"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebooks" / "01_text_embedding_baseline.ipynb")
print("wrote notebooks/01_text_embedding_baseline.ipynb")
