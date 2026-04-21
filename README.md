# EPDP — Efficient Pathology Diffusion Pipeline

Class-conditional, pixel-space diffusion framework for synthesizing
clinically and technically trustworthy histopathology image patches of
**breast** and **gastric** cancers, with reference-guided **CycleGAN**
stain normalization and **EfficientNetV2-L** downstream evaluation.

This repository accompanies the manuscript
*"Building a Clinically and Technically Trustworthy Synthetic
Histopathology Dataset for Breast and Gastric Cancer."*

---

## 1. Repository layout

```
EPDP-Pathology/
├── configs/                      # YAML hyperparameter sets (one per model)
│   ├── diffusion_breast.yaml
│   ├── diffusion_stomach.yaml
│   ├── cyclegan.yaml
│   └── classifier.yaml
├── src/
│   ├── diffusion/                # Pixel-space class-conditional DDPM
│   │   ├── unet.py               # Custom denoising U-Net (Res + memory-saving Attn)
│   │   ├── diffusion.py          # GaussianDiffusion: DDPM training + DDIM sampling
│   │   ├── embedding.py          # ConditionalEmbedding (class label → 256-d vector)
│   │   ├── beta_schedule.py      # linear / cosine β-schedules
│   │   └── scheduler.py          # GradualWarmupScheduler
│   ├── stain_norm/
│   │   └── cyclegan.py           # CycleGAN Generator / PatchGAN Discriminator
│   ├── classifier/
│   │   └── efficientnet_v2.py    # timm backbone + Linear head + SAM optimizer
│   └── evaluation/
│       ├── fid.py                # Fréchet Inception Distance
│       ├── lpips_metric.py       # 1 − LPIPS (AlexNet)
│       └── ssim_metric.py        # SSIM (skimage)
├── scripts/                      # Single-GPU training entry points
│   ├── train_diffusion.py        # Class-conditional DDPM (breast or stomach)
│   ├── train_cyclegan.py         # CycleGAN stain normalization (Stage 1)
│   └── train_classifier.py       # EfficientNetV2-L + SAM (EffNet-Real / Syn)
├── src/data/
│   └── datasets.py               # ClassFolderDataset, UnpairedFolderDataset
├── notebooks/                    # Reference Jupyter notebooks (kept for record)
│   ├── cyclegan_training.ipynb
│   ├── color_normalization_pipeline.ipynb
│   ├── efficientnetv2_classifier.ipynb
│   ├── evaluation_fid_lpips_ssim.ipynb
│   └── report_classification.ipynb
├── environment.yml               # conda env (Python 3.10 + CUDA 12.1)
├── requirements.txt              # pip-only fallback
└── docs/                         # extended notes (optional)
```

---

## 2. Environment

The reference environment is **Ubuntu 20.04 + CUDA 12.1**, single
**NVIDIA A100-SXM4-80GB**, `Python 3.10`, `PyTorch 2.3.1+cu121`.

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate epdp

# Or pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Dataset

> **No images and no model checkpoints are shipped with this repository.**
> The original NIA dataset is restricted-access medical data and cannot be
> redistributed. To run any training script you must arrange your own
> patches into the folder layouts below. See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)
> for the complete spec.

The manuscript dataset comprises 5,729 H&E-stained WSIs from 4 Korean
tertiary hospitals (Gachon, Catholic, Korea, Ajou). WSIs were
down-sampled from 40× to 20× and tiled to non-overlapping
**1024 × 1024** patches; only patches with < 25% diagnostically
irrelevant area were retained after a two-round expert review
(manuscript Table 1).

| Cancer  | Subtypes                                | Refined patches |
| ------- | --------------------------------------- | --------------: |
| Breast  | BRNT / BRLC / BRDC / BRIL / BRID        |          10,629 |
| Stomach | STNT / STIN / STDI / STMX               |           5,341 |

### 3.1 Required folder layouts

**Diffusion / classifier** (class-folder layout):

```text
<data_root>/
├── BRNT/        # one folder per class (folder names match class_list in the YAML)
│   ├── 0000.png
│   └── ...
├── BRLC/
├── BRDC/
├── BRIL/
└── BRID/
```

**CycleGAN** (two unpaired pools):

```text
<cyclegan_root>/
├── trainA/      # source domain (raw scans)
└── trainB/      # target / reference stain
```

| Item | Spec |
| --- | --- |
| Image format | PNG / JPEG / TIFF (anything `PIL.Image.open` can read) |
| Channels | RGB (single-channel auto-converted) |
| Patch size on disk | Any; loader resizes to `image_size` (default 1024) |
| Magnification | 40× WSI patches in the manuscript (informational) |

Synthetic outputs are released through the AI Hub platform
(<https://aihub.or.kr/>). The original WSIs remain restricted under each
institution's IRB.

---

## 4. Reproducibility — full implementation details

All numbers below come **directly from the training scripts and
configs** committed in this repo. The same parameters reproduce the
manuscript runs.

### 4.1 Class-conditional denoising U-Net (`src/diffusion/unet.py`)

| Component         | Setting                                                  |
| ----------------- | -------------------------------------------------------- |
| Input / output    | RGB, **1024 × 1024**, channels `in=3, out=3`             |
| Base width        | `mod_ch = 128`                                           |
| Channel mults     | `ch_mul = [1, 2, 4, 4, 4]`  (5 levels → 4 down/up-samples) |
| ResBlocks / level | `num_res_blocks = 2`                                     |
| Bottleneck width  | `128 × 4 = 512`                                          |
| Downsample        | `AvgPool2d(2,2)`  (`use_conv = False`)                   |
| Upsample          | Nearest interp. ×2 → `Conv 3×3`                          |
| Normalization     | `GroupNorm(32, ·)`                                       |
| Activation        | `SiLU` (Swish)                                           |
| Dropout           | `0.1` inside ResBlock                                    |
| Time embedding    | Sinusoidal(128) → `Linear 128→512 → SiLU → Linear 512→512` |
| Class embedding   | `Embedding(N+1, 256, padding_idx=0)` → `Linear 256→256 → SiLU → Linear 256→256` then `Linear 256→512 → SiLU → Linear 512→512` inside U-Net |
| Attention         | `AttnBlock` after every ResBlock — `GroupNorm(32) → Q/K/V Conv 1×1 (stride 16)` (memory-saving 1/16 spatial), bilinear up-sample back to original H×W |
| Skip connections  | Standard U-Net concatenation                             |
| Output head       | `GroupNorm(32) → SiLU → Conv 3×3 → 3-channel ε̂`         |

The **custom residual block** is the canonical
*Norm → Activation → Conv → time/class projection → Norm → Activation
→ Dropout → Conv* pattern, with a 1×1 shortcut when channel counts
change. It is inserted at every encoder/decoder level (twice per
level), as described in the manuscript.

### 4.2 DDPM / DDIM (`src/diffusion/diffusion.py`)

| Hyperparameter            | Value                       |
| ------------------------- | --------------------------- |
| Diffusion steps `T`       | `1000`                      |
| β-schedule                | linear, `1e-4 → 2e-2`       |
| Loss                      | `MSE(ε̂, ε)`                |
| Classifier-free guidance  | `w = 1.8`, dropout `0.1`    |
| Posterior variance interp | `v = 0.3` (between `β` and `β̃`) |
| Sampling                  | **DDIM**, `100 steps`, `η = 0`, `select = quadratic` |

### 4.3 Diffusion training loop (`scripts/train_diffusion.py`)

| Item              | Breast                  | Stomach                |
| ----------------- | ----------------------- | ---------------------- |
| Subtypes          | 5 (BRNT/BRLC/BRDC/BRIL/BRID) | 4 (STNT/STMX/STIN/STDI) |
| Optimizer         | `AdamW(lr, weight_decay=1e-6)` | same                  |
| Learning rate     | **5 × 10⁻⁵**            | **1 × 10⁻⁵**            |
| LR schedule       | `GradualWarmup(multiplier=1, warm_epoch=3)` → `ExponentialLR(γ=0.95)` | same |
| Batch size        | `1` (single 1024² patch / step) | same                |
| Epochs            | `1000`                  | same                   |
| Mixed precision   | `torch.cuda.amp.GradScaler` | same               |
| Augmentation      | `RandomHorizontalFlip(0.5)`, `RandomVerticalFlip(0.5)` | same |
| Normalization     | `mean = std = (0.5, 0.5, 0.5)` (range `[-1, 1]`) | same       |
| GPU               | NVIDIA A100-SXM4-80GB   | same                   |
| Memory footprint  | ≈ 43.4 GB (model, fwd/bwd, params combined) | same       |

DDIM samples are written every `sample_every_epochs` (configurable in
the YAML, default 25). Sampling is performed **without** any external
CycleGAN post-process — pass the diffusion samples through a separately
trained CycleGAN G_B at deployment time if stain normalization is
desired.

### 4.4 Reference-guided CycleGAN (`src/stain_norm/cyclegan.py`)

Two-stage training procedure (manuscript Fig. 2):

1. **Stage 1 — unpaired**: train CycleGAN on Dataset A
   (10 expert-selected references with histogram correlation > 0.9)
   ↔ Dataset B (random, equal size) → preliminary `G_BA`.
2. **Stage 2 — pseudo-paired**: apply `G_BA` to a masked Dataset B
   (white regions detected by RGB σ < 5) to obtain `B̂`. Re-train
   CycleGAN on the pseudo-paired (`B`, `B̂`) for the final stain
   normalizer.

| Item                  | Value                                          |
| --------------------- | ---------------------------------------------- |
| Patch size            | `1024 × 1024`                                  |
| Generator             | ResNet, `9 residual blocks`, ReflectionPad     |
| Discriminator         | 70×70 PatchGAN, base width 64                  |
| Optimizer             | Adam, `lr_G = lr_D = 2e-4`, `β = (0.5, 0.999)` |
| Cycle loss weight     | `λ_A = λ_B = 10`                               |
| Perceptual / style    | VGG16 (relu1_2, relu2_2, relu3_3, relu4_3), Gram-matrix MSE × `1e4` |
| Image pool            | size `10` (Shrivastava-style fake history)     |
| Epochs                | `100` (linear LR decay starts at epoch `50`)   |
| Batch size            | `1`                                            |
| Augmentation          | none (no flips during stain training)          |

### 4.5 EfficientNetV2-L classifier (`src/classifier/efficientnet_v2.py`)

Two classifiers are trained independently (**EffNet-Real**,
**EffNet-Syn**) with identical architecture / optimization; only the
training source differs. Both are evaluated on the same held-out real
patches.

| Item              | Value                                          |
| ----------------- | ---------------------------------------------- |
| Backbone (timm)   | `tf_efficientnetv2_l` (pretrained, 1280-d)     |
| Head              | single `Linear(1280, n_classes)`               |
| Input size        | `512 × 512` (resized from 1024 patches)        |
| Optimizer         | **SAM** (ρ = 0.05) wrapping **SGD** (mom 0.9)  |
| Learning rate     | `2 × 10⁻⁴`                                     |
| Batch size        | `1`                                            |
| Epochs            | `1000` (best-val checkpoint)                   |
| Loss              | `cross_entropy(softmax(logits), one_hot)`      |
| Augmentation      | `H/V flip @ 0.5`                               |
| Normalization     | `mean = std = (0.5, 0.5, 0.5)`                 |
| Train/val split   | `80 / 20`, `random_state = 42`                 |

> **Note on the backbone.** The original notebook in this repo
> (`notebooks/efficientnetv2_classifier.ipynb`) calls
> `tf_efficientnetv2_xl`, while the manuscript reports
> EfficientNetV2-**L**. The manuscript's reported numbers are with
> `tf_efficientnetv2_l`; the XL notebook is kept for transparency.

### 4.6 Quantitative evaluation (`src/evaluation/`)

| Metric    | Definition                                            | Implementation                          |
| --------- | ----------------------------------------------------- | --------------------------------------- |
| FID       | distributional distance over InceptionV3 features (299×299) | `pytorch-fid` (Heusel 2017)        |
| 1 − LPIPS | perceptual similarity (`AlexNet` backbone, [-1, 1])   | `lpips==0.1.x` (Zhang 2018)             |
| SSIM      | structural similarity, win 11, RGB channel-axis       | `skimage.metrics.structural_similarity` |

All metrics are reported as **mean ± std** of pairwise scores.
Real-vs-real establishes the baseline; real-vs-synthetic is the
quantity of interest. Per-subtype gaps and the relative-difference
formula are defined in the manuscript (Eq. 4).

### 4.7 Visual Turing test (VTT)

Two board-certified pathologists (≥ 10 yrs each) recruited via NIA
classified 50 real + 50 synthetic 1024² patches per organ in
randomized order. Accuracy in `[40 %, 60 %]` indicates synthetic
images are perceptually indistinguishable. Reported VTT accuracies in
this study lie in the `50 %–56 %` range (manuscript Table 7).

---

## 5. Quick start

All three training entry points are single-GPU, YAML-driven, and write
**only** inside `--output-dir` (no hard-coded paths). Set up your
dataset following [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) first.

```bash
# 1. Train the breast class-conditional diffusion model
python scripts/train_diffusion.py \
    --config    configs/diffusion_breast.yaml \
    --data-root /path/to/breast_patches \
    --output-dir runs/diffusion_breast \
    --device    cuda:0

# 2. Train the stomach class-conditional diffusion model
python scripts/train_diffusion.py \
    --config    configs/diffusion_stomach.yaml \
    --data-root /path/to/stomach_patches \
    --output-dir runs/diffusion_stomach \
    --device    cuda:0

# 3. Train CycleGAN stain normalization (Stage 1: unpaired A↔B)
python scripts/train_cyclegan.py \
    --config    configs/cyclegan.yaml \
    --data-root /path/to/cyclegan_root \
    --output-dir runs/cyclegan

# 4. Train the EfficientNetV2-L subtype classifier with SAM
#    --data-root → real patches  for EffNet-Real
#    --data-root → synthetic patches generated by step 1 for EffNet-Syn
python scripts/train_classifier.py \
    --config    configs/classifier.yaml \
    --organ     breast \
    --data-root /path/to/breast_patches \
    --output-dir runs/classifier_real_breast
```

Outputs of every script land at::

```text
<output_dir>/
├── checkpoints/ckpt_<epoch>.pt
├── samples/                 # diffusion / CycleGAN preview images
├── log.csv                  # epoch, loss, lr (and val acc for classifier)
└── config.snapshot.yaml     # the YAML used for this run
```

Resume any run by passing `--resume runs/<name>/checkpoints/ckpt_<epoch>.pt`.

The original Jupyter notebooks under `notebooks/` are kept verbatim for
record-keeping but are **not** the recommended entry point — use the
scripts above.

---

## 6. Citation

If you use this code or the released synthetic dataset, please cite
the manuscript (full reference will be added on publication) and
acknowledge the supporting grants:

- National Institute of Health (NIH) — project R25TA00471930-00
- Gachon University research fund 2024 — GCU-202410530001

---

## 7. Provenance

This repository is a curated and refactored snapshot of the EPDP work
originally developed in
[`Leeyoungsup/24_NIA_histopathology`](https://github.com/Leeyoungsup/24_NIA_histopathology).
Only the modules used by the manuscript (diffusion, CycleGAN stain
normalization, EfficientNetV2 classifier, FID/LPIPS/SSIM evaluation)
were carried over; experimental side branches (StyleGAN2-ADA,
RealESRGAN, HoVer-Net, MIL attention, captioning, etc.) were dropped
to keep the codebase reproducible and reviewable.
