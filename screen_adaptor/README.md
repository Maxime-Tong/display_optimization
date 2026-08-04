# Screen Adaptor V2

Compact screen colour optimisation pipeline: predict per‑pixel ellipsoid radii,
apply a closed‑form colour transform, and export 3D LUTs for real‑time inference.

## Install

```bash
pip install -e .[dev]
```

## Quick start

```bash
# Single‑dataset training
python -m src.screen_adaptor.pipeline pretrain --data-dir D:\workspace\master\3DGS\Vulkan\screen_adaptor\datasets\ --output-dir outputs --device cuda --config D:\workspace\master\3DGS\Vulkan\screen_adaptor_v2\configs\pretrain_config.json  

# Evaluate (single LUT)
python -m src.screen_adaptor.eval --lut outputs/base_lut.pt --input-dir D:\workspace\master\3DGS\Vulkan\screen_adaptor\datasets\cf --output-dir results
```

## Loss function

The total loss is:

```
L = L_power + λ_p · L_perceptual + λ_s · L_ssim + λ_r · ‖r‖₂²
```

| Component | Description |
|---|---|
| `L_power` | Weighted power of optimized output — drives power consumption toward target ratio **α** (0–1). |
| `L_perceptual` | LPIPS (VGG) perceptual loss. |
| `L_ssim` | `1 − SSIM` structural similarity loss. |
| `‖r‖₂²` | L2 regularisation on predicted radii — keeps adjustments small and training stable. |

### Key hyper‑parameters

| Key | Default | Meaning |
|---|---|---|
| `alpha_power` | 0.7 | Target power‑consumption ratio (0–1). Lower = more aggressive energy saving. |
| `lambda_perceptual` | 0.5 | Weight of LPIPS perceptual loss. |
| `lambda_ssim` | 0.5 | Weight of SSIM loss. |

## JSON configuration

All training hyper‑parameters can be set via a JSON file.  Only keys you want to
override need to be present; missing keys fall back to defaults.

Example (`configs/train_config.json`):

```json
{
    "steps": 2000,
    "lr": 5e-4,
    "alpha_power": 0.6,
    "lambda_perceptual": 0.5,
    "lambda_ssim": 0.5,
}
```

## Pipeline (multi‑dataset clustering)

For multiple datasets, first cluster by DKL colour features, then train one LUT
per cluster:

```bash
# Phase 1: extract features + cluster → manifest
python -m src.screen_adaptor.pipeline derive \
  --data-dir datasets \
  --manifest outputs/scene_manifest.json \
  --clusters 4

# Phase 2: train per‑cluster LUTs (with JSON config)
python -m src.screen_adaptor.pipeline train `
  --manifest outputs/scene_manifest.json `
  --lut-dir outputs/luts `
  --config configs/train_config.json `
  --device cuda

# Or do both at once:
python -m src.screen_adaptor.pipeline full \
  --data-dir datasets \
  --output-dir outputs \
  --config configs/train_config.json \
  --clusters 8 \
  --device cuda

# Or with pretrain + finetune:
python -m src.screen_adaptor.pipeline full-pt \
  --data-dir datasets \
  --output-dir outputs \
  --pretrain-config configs/pretrain_config.json \
  --finetune-config configs/finetune_config.json \
  --clusters 8 \
  --device cuda
```

## Inference / evaluation

```bash
# Single LUT
python -m src.screen_adaptor.eval \
  --lut outputs/base_lut.pt \
  --input-dir path/to/images \
  --output-dir results

# Scene‑aware LUT switching (from manifest)
python -m src.screen_adaptor.eval \
  --input-dir datasets \
  --output-dir results \
  --scene-manifest outputs/scene_manifest.json \
  --eval-mode

# With foveated rendering & temporal smoothing
python -m src.screen_adaptor.eval \
  --lut outputs/base_lut.pt \
  --input-dir path/to/images \
  --output-dir results \
  --foveated 0.5 --temporal 0.3
```

## File structure

```
screen_adaptor_v2/
├── configs/
│   ├── train_config.json           # train hyper‑parameter config
│   ├── pretrain_config.json        # pretrain phase config
│   └── finetune_config.json        # finetune phase config
├── models/
│   └── vgg16-397923af.pth          # LPIPS VGG weights
├── src/screen_adaptor/
│   ├── __init__.py                 # public API re‑exports
│   ├── pipeline.py                 # training loop + loss + clustering + CLI
│   ├── eval.py                     # inference / evaluation with metrics
│   ├── model.py                    # EllipsoidRadiusNet + LUTColorTransformer
│   ├── color_ops.py                # closed‑form transform, power helpers
│   ├── scene_matcher.py            # DKL feature extraction + k‑means + manifest
│   ├── utils.py                    # image loading, foveated phi map
│   └── colorspace.py               # color space matrices (sRGB↔XYZ↔LMS↔DKL)
├── tests/
│   ├── test.py
│   └── visualize.py
├── outputs/                        # generated LUTs and manifests
├── run.ps1                         # quick‑start PowerShell script
├── eval.ps1                        # batch evaluation script
└── README.md
```

## CLI reference

### `pipeline.py` (training)

| Command | Description |
|---|---|
| `derive` | Extract DKL features → k‑means cluster → save manifest |
| `pretrain` | Train a base model on sampled images from all scenes |
| `train` | Train per‑cluster LUTs from existing manifest |
| `full` | `derive` + `train` (from scratch) |
| `full-pt` | `pretrain` + `derive` + finetune `train` |

### `eval.py` (evaluation)

| Flag | Description |
|---|---|
| `--lut` | Path to single LUT (.pt) |
| `--input-dir` | Directory of images to evaluate |
| `--output-dir` | Where to save optimized images |
| `--scene-manifest` | JSON manifest for scene‑aware LUT switching |
| `--eval-mode` | Only compute metrics, don't save images |
| `--json-output` | JSON file for evaluation results |
| `--foveated` | Foveated modulation strength [0, 1] |
| `--temporal` | Temporal smoothing strength [0, 1] |
| `--power-weights` | 3 floats for R, G, B OLED power coefficients |