# hvs_vr_encoding (Modified Objective) Benchmark Results

- Method: hvs_vr_encoding color-discrimination-guided framebuffer compression with MODIFIED optimization objective (per-pixel max delta reduction)
- H-FOV: 110°, Max Ecc: 35°, Ecc No-Compress: 10°, Tile Size: 4
- Benchmark metric: **screen (OLED) display power** saving with weights (R,G,B) = (0.229, 0.243, 0.526)
- MetaM computed with max side 768 px (odak MetamericLoss)
- Images per dataset: up to 10

| Dataset | Images | Power Saving (%) | PSNR (dB) | SSIM | MetaM | BD orig (%) | BD opt (%) |
|---------|--------|:----------------:|:---------:|:-----:|:------:|:-----------:|:----------:|
| cf | 10 | 5.33 | 29.49 | 0.9593 | 0.000008 | 35.65 | 35.65 |
| delta_force | 10 | 4.69 | 32.99 | 0.9875 | 0.000005 | 40.62 | 41.01 |
| dfm300 | 10 | 4.53 | 30.41 | 0.9868 | 0.000007 | 43.42 | 43.79 |
| genshin_impact | 10 | 8.74 | 24.17 | 0.9315 | 0.000012 | 32.67 | 33.29 |
| jkchess | 10 | 4.77 | 29.47 | 0.9492 | 0.000010 | 45.01 | 45.60 |
| miHoYo | 10 | 4.72 | 28.39 | 0.9600 | 0.000009 | 56.68 | 56.67 |
| nrc | 10 | 8.21 | 23.55 | 0.9326 | 0.000014 | 49.25 | 48.69 |
| sgame0 | 10 | 4.74 | 31.75 | 0.9818 | 0.000007 | 42.60 | 43.00 |
