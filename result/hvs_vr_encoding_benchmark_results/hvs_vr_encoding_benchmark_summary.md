# hvs_vr_encoding (Modified Objective) Benchmark Results

- Method: hvs_vr_encoding color-discrimination-guided framebuffer compression with MODIFIED optimization objective (per-pixel max delta reduction)
- H-FOV: 110°, Max Ecc: 35°, Ecc No-Compress: 10°, Tile Size: 4
- Benchmark metric: **screen (OLED) display power** saving with weights (R,G,B) = (0.229, 0.243, 0.526)
- MetaM computed with max side 768 px (odak MetamericLoss)
- Images per dataset: up to 10

| Dataset | Images | Power Saving (%) | PSNR (dB) | SSIM | MetaM | BD orig (%) | BD opt (%) |
|---------|--------|:----------------:|:---------:|:-----:|:------:|:-----------:|:----------:|
| cf | 10 | 4.75 | 30.62 | 0.9662 | 0.000007 | 35.43 | 35.39 |
| delta_force | 10 | 4.98 | 33.09 | 0.9871 | 0.000006 | 42.55 | 42.95 |
| dfm300 | 10 | 4.80 | 30.64 | 0.9886 | 0.000008 | 42.66 | 43.00 |
| genshin_impact | 10 | 7.26 | 26.13 | 0.9540 | 0.000010 | 33.76 | 33.89 |
| jkchess | 10 | 4.49 | 30.99 | 0.9567 | 0.000008 | 49.44 | 50.05 |
| miHoYo | 10 | 4.05 | 29.52 | 0.9657 | 0.000007 | 57.89 | 57.86 |
| nrc | 10 | 9.72 | 22.59 | 0.9146 | 0.000014 | 48.45 | 47.50 |
| sgame0 | 10 | 4.25 | 31.81 | 0.9769 | 0.000006 | 42.47 | 42.92 |
