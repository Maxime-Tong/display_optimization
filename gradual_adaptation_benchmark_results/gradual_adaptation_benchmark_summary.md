# Gradual Chromatic Adaptation (GCA) + GUD Benchmark Results

- Method: slow white-point shift along a yellow-green u'v' trajectory from D65 (Bradford CAT), perceptually invisible via human chromatic adaptation, + GUD uniform dimming
- Trajectory: **1.47**, velocity: **0.000419 u'v'/s** (adaptation-bounded, perceived cast <= 5 JND), time: **120.0 s** (t_max = 120.0 s)
- GUD: enabled=True, target=0.85
- Benchmark metric: **screen (OLED) display power** saving with weights (R,G,B) = (0.229, 0.243, 0.526)
- MetaM computed with max side 768 px (odak MetamericLoss), per image on its own time-stamp state
- Images per dataset: up to 10
- Temporal frame-skipped power: 10 images sampled across [0, 120.0 s]; image k rendered at t=(k+0.5)/N*120.0 s, per-image saving ramps 0 -> max, dataset mean = ramp average

| Dataset | Images | Power Saving (%) | PSNR (dB) | SSIM | MetaM |
|---------|--------|:----------------:|:---------:|:-----:|:------:|
| cf | 10 | 15.90 | 25.84 | 0.9676 | 0.000024 |
| delta_force | 10 | 15.86 | 27.68 | 0.9707 | 0.000024 |
| dfm300 | 10 | 15.42 | 24.51 | 0.9709 | 0.000027 |
| genshin_impact | 10 | 16.61 | 24.61 | 0.9460 | 0.000033 |
| jkchess | 10 | 16.40 | 25.80 | 0.9775 | 0.000031 |
| miHoYo | 10 | 15.48 | 23.74 | 0.9759 | 0.000035 |
| nrc | 10 | 15.55 | 23.29 | 0.9654 | 0.000034 |
| sgame0 | 10 | 15.85 | 27.21 | 0.9680 | 0.000024 |
