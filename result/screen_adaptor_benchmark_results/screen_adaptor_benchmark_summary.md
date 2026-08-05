# screen_adaptor Benchmark Results

- Method: EllipsoidRadiusNet-derived 3D LUT color transform
- LUT mode: **cluster** (8 LUTs)
- Foveated: 1.0, Temporal: 0.0
- Scene matching enabled (dynamic LUT switching via DKL features)
- Benchmark metric: **screen (OLED) display power** saving with weights (R,G,B) = (0.229, 0.243, 0.526)
- MetaM computed with max side 768 px (odak MetamericLoss)
- Images per dataset: up to 10

| Dataset | Images | Power Saving (%) | PSNR (dB) | SSIM | MetaM |
|---------|--------|:----------------:|:---------:|:-----:|:------:|
| cf | 10 | 10.25 | 28.08 | 0.9849 | 0.000014 |
| delta_force | 10 | 10.85 | 29.55 | 0.9875 | 0.000013 |
| dfm300 | 10 | 10.10 | 27.16 | 0.9868 | 0.000015 |
| genshin_impact | 10 | 10.83 | 26.79 | 0.9820 | 0.000017 |
| jkchess | 10 | 9.67 | 30.41 | 0.9901 | 0.000012 |
| miHoYo | 10 | 8.87 | 26.33 | 0.9949 | 0.000018 |
| nrc | 10 | 10.29 | 25.86 | 0.9828 | 0.000019 |
| sgame0 | 10 | 10.81 | 29.36 | 0.9857 | 0.000013 |
