# Display Optimization for VR

A collection of display optimization techniques for Virtual Reality headsets, focused on **reducing OLED display power consumption while remaining perceptually lossless**. The project combines color-perception-aware algorithms, FPGA-accelerated compression, and screen-level power reduction pipelines.

## Overview

This repository contains four complementary modules plus three end-to-end benchmark scripts:

| Module | Description |
|--------|-------------|
| [`screen_adaptor/`](screen_adaptor/) | Compact screen color optimisation pipeline: predicts per-pixel ellipsoid radii, applies a closed-form colour transform, and exports 3D LUTs for real-time inference |
| [`hvs_vr_encoding/`](hvs_vr_encoding/) | FPGA demo of color discrimination-guided framebuffer compression (ASPLOS 2024) — leverages eccentricity-dependent human color discrimination to improve Base-Delta compression |
| [`vr-power-saver/`](vr-power-saver/) | Color-perception-guided display power reduction using an RBF network trained on HVS data (eccentricity-aware color shift with foveated transition masks) |
| [`gradual_adaptation/`](gradual_adaptation/) | Gradual Chromatic Adaptation (GCA) — slowly shifts the illuminant white point along a daylight trajectory so the color change is imperceptible, saving OLED power without eye tracking |

The three root-level benchmark scripts apply these methods to game screenshot datasets and report power savings plus perceptual quality metrics (PSNR, SSIM, Metameric loss):

- `benchmark_vr_power_saver.py` — applies the vr-power-saver RBF pipeline
- `benchmark_hvs_vr_encoding.py` — applies a modified HVS-VR-encoding objective (maximally reduces each pixel's changeable delta within its JND ellipsoid)
- `benchmark_gradual_adaptation.py` — simulates the 2-minute gradual ramp (GCA + GUD) with per-image time-stamped evaluation

## Directory Layout

```
display_project/
├── screen_adaptor/                 # LUT-based screen colour optimizer
│   ├── src/screen_adaptor/         #   pipeline / eval / model / color_ops / scene_matcher
│   ├── configs/                    #   training JSON configs
│   ├── odak/                       #   odak library dependency (perceptual losses)
│   └── tests/                      #   test & visualization scripts
├── hvs_vr_encoding/                # ASPLOS 2024 FPGA compression demo
│   ├── host/                       #   CPU/GPU modules: projection, lens correction,
│   │                               #   color optimizer, base_delta, video processing
│   ├── fpga/                       #   HLS implementations + bitstreams + IP repo
│   └── scripts/                    #   pipeline scripts (CPU / GPU / GPU+FPGA)
├── vr-power-saver/                 # Color-perception-guided power reduction
│   ├── color_model/                #   BaseColorModel (RBF network)
│   ├── util/                       #   colorspace, torch RBF, VR tools
│   └── power_saver_demo.py         #   demo entry point
├── gradual_adaptation/             # Gradual chromatic adaptation (GCA)
│   ├── color_adaptation.py         #   per-frame optimizer
│   ├── trajectory.py               #   daylight & linear trajectories
│   ├── power_model.py              #   OLED power model
│   └── image_optimizer.py          #   full frame pipeline (GCA + GUD)
├── benchmark_*.py                  # end-to-end benchmark scripts
├── result/                         # benchmark outputs (CSV / JSON / Markdown)
└── gradual_adaptation_benchmark_results/
```

## Metrics

All benchmarks measure **screen display power** and quality metrics on the same datasets:

- **Saving%** = `1 − power(optimized) / power(original)`, where `power(img) = Σ (R·0.229 + G·0.243 + B·0.526)` (OLED RGB power coefficients)
- **PSNR** and **SSIM** — pixel-level fidelity
- **MetaM** — odak `MetamericLoss` (HVS metameric / perceptual loss)

## Quick Start

Each submodule has its own README with detailed usage:

```bash
# Screen adaptor (train / evaluate LUTs)
cd screen_adaptor
pip install -e .[dev]
python -m src.screen_adaptor.pipeline pretrain --data-dir datasets/ --output-dir outputs --config configs/pretrain_config.json

# VR power saver demo
cd vr-power-saver
python power_saver_demo.py

# Gradual chromatic adaptation
cd gradual_adaptation
python -c "from gradual_adaptation import GradualChromaticOptimizer; ..."
```

### Running the benchmarks

```bash
# vr-power-saver on all datasets (10 images per dataset by default)
python benchmark_vr_power_saver.py --max-images 10

# hvs_vr_encoding modified objective benchmark
python benchmark_hvs_vr_encoding.py --max-images 10

# gradual adaptation (GCA + GUD) ramp benchmark
python benchmark_gradual_adaptation.py --max-images 10
```

Results are written to `result/` (or the script's `--output-dir`).

## Notes

- Large data files (datasets, model weights, pre-trained checkpoints) and regenerable artifacts (bitstreams, outputs, logs) are **not** committed — see `.gitignore`.
- The FPGA `.bit`/`.hwh` bitstreams in `hvs_vr_encoding/fpga/` must be regenerated via Vivado (see the module README); pre-generated files are available online via the links in that README.