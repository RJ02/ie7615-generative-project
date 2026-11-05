# From Words to Worlds: Text-to-Image Generation

**IE 7615 Deep Learning for AI · Generative Project** · Study Group 6: Sankalp Hegde · Reshma Jose · Jairaghavendra Sridhar

A text-to-image pipeline that conditions a pretrained latent diffusion model (Stable Diffusion v1.5) on Transformer
(CLIP) text embeddings, evaluated on COCO 2017 image–caption pairs.

📄 Milestone 1 proposal: [`report/milestone1_proposal.pdf`](report/milestone1_proposal.pdf)

## Milestone 1: setup and dataset (done)

| Task | Result |
|---|---|
| Dataset | COCO 2017 val: 5,000 images, 25,014 captions → **4,958 images, 24,788 captions** after cleaning ([`stats.json`](data/processed/stats.json)) |
| Preprocessing | Caption normalisation, de-duplication, 5–40 word filter (all within CLIP's 77 tokens); RGB conversion, short side ≥ 256 px, centre crop to 512 × 512 |
| Environment | PyTorch + Diffusers + Transformers (CLIP), Apple M4 GPU (MPS) or CUDA |
| Text → embedding baseline | [`notebooks/01_text_embedding_baseline.ipynb`](notebooks/01_text_embedding_baseline.ipynb): CLIP ViT-B/32 embeddings; matched cosine 0.305 vs. 0.154 random; caption→image R@1 51.5% (1k pairs), 29.5% (4,958 pairs); SD v1.5 conditioning (77 × 768) and the empty-prompt embedding for classifier-free guidance |
| 5 prompt → image runs | [`outputs/m1_samples/`](outputs/m1_samples/): SD v1.5, DPM-Solver, 25 steps, guidance 7.5, seed 42, with CLIP scores ([`runs.json`](outputs/m1_samples/runs.json)) |

![Five Milestone 1 samples](outputs/m1_samples/strip.jpg)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# COCO 2017 val images + captions (~1.1 GB)
mkdir -p data/raw && cd data/raw
curl -O http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip
cd ../..
```

Model weights (Stable Diffusion v1.5 fp16 and CLIP ViT-B/32, ~3 GB) download automatically from Hugging Face on first use.

## Run

```bash
python src/prepare_dataset.py            # -> data/processed/pairs.csv, images/ (512 x 512), stats.json
python src/build_notebook.py             # writes notebooks/01_text_embedding_baseline.ipynb; run it in Jupyter
python src/sample_runs.py                # -> outputs/m1_samples/ (5 images, grid, CLIP scores)
```

## Repository structure

```
├── src/
│   ├── prepare_dataset.py     # cleaning + preprocessing of COCO image-caption pairs
│   ├── build_notebook.py      # builds the text -> embedding baseline notebook
│   └── sample_runs.py         # 5 prompt -> image test runs with CLIP scores
├── notebooks/01_text_embedding_baseline.ipynb
├── data/processed/            # pairs.csv, stats.json, embedding_baseline.json (images not committed)
├── outputs/m1_samples/        # generated images + runs.json
└── report/                    # IEEE-format milestone reports (LaTeX + PDF)
```

## Roadmap

| Milestone | Focus |
|---|---|
| 1 ✅ | Dataset, environment, text-embedding baseline, first samples |
| 2 | Text encoder ↔ diffusion integration; tune classifier-free guidance and noise schedule |
| 3 | FID / Inception Score, CLIP score, parameter sensitivity (guidance, steps, sampler, encoder) |
| 4 | Final 6–8 page report, gallery, presentation |

Data: COCO 2017 (images under Flickr terms, annotations CC BY 4.0). No COCO images are stored in this repository.
