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
| cf | 10 | 15.85 | 25.89 | 0.9680 | 0.000030 |
| delta_force | 10 | 15.89 | 28.03 | 0.9707 | 0.000026 |
| dfm300 | 10 | 15.49 | 25.03 | 0.9709 | 0.000029 |
| genshin_impact | 10 | 16.54 | 25.01 | 0.9486 | 0.000035 |
| jkchess | 10 | 16.66 | 26.91 | 0.9769 | 0.000033 |
| miHoYo | 10 | 15.49 | 23.68 | 0.9766 | 0.000041 |
| nrc | 10 | 16.18 | 24.05 | 0.9520 | 0.000037 |
| sgame0 | 10 | 16.00 | 27.68 | 0.9686 | 0.000026 |
