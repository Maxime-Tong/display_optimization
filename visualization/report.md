# LUT Specialization Analysis Report

Generated on 2026-08-16 11:21

## 1. Overview

- Base LUT: `D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\outputs\base_lut.pt`
- Finetuned LUTs: `D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\outputs\luts` (8 clusters discovered)
- Grid resolution: 16 x 16 x 16 x 3
- Delta-E metric: cie2000
- Specialization score: distance_from_base / (adjustment_std + ε), ε = 1e-06

## 2. Distance from base LUT

| scene | euclidean | manhattan | cosine |
|---|---|---|---|
| cluster_5 | 1.1067 | 108.2323 | 0.0001 |
| cluster_2 | 1.0952 | 108.1310 | 0.0001 |
| cluster_3 | 1.0738 | 107.3143 | 0.0000 |
| cluster_7 | 1.0665 | 102.8890 | 0.0001 |
| cluster_1 | 1.0530 | 107.5968 | 0.0000 |
| cluster_6 | 1.0235 | 103.7142 | 0.0000 |
| cluster_0 | 0.9405 | 91.3317 | 0.0000 |
| cluster_4 | 0.8006 | 64.4419 | 0.0000 |

If all finetuned LUTs sit at nearly the same distance from the base and the pairwise distances between them are small, the finetuning produced a generic (shared) adjustment rather than scene-specific ones.

## 3. Per-scene colour difference (ΔE vs base)

| scene | mean | std | p50 | p95 | p99 | max | nonzero_frac |
|---|---|---|---|---|---|---|---|
| cluster_5 | 0.7331 | 0.1917 | 0.7088 | 1.0793 | 1.2464 | 1.8149 | 0.9998 |
| cluster_2 | 0.7241 | 0.1688 | 0.7050 | 1.0241 | 1.1578 | 1.7734 | 0.9998 |
| cluster_7 | 0.7192 | 0.1587 | 0.7084 | 0.9878 | 1.1086 | 1.7008 | 0.9998 |
| cluster_3 | 0.7089 | 0.1435 | 0.7001 | 0.9488 | 1.0604 | 1.6794 | 0.9998 |
| cluster_0 | 0.6925 | 0.1485 | 0.6888 | 0.9111 | 1.0757 | 1.6709 | 0.9998 |
| cluster_1 | 0.6923 | 0.1216 | 0.6949 | 0.8848 | 0.9710 | 1.6023 | 0.9998 |
| cluster_6 | 0.6793 | 0.1225 | 0.6817 | 0.8725 | 0.9544 | 1.5949 | 0.9998 |
| cluster_4 | 0.5816 | 0.2697 | 0.5257 | 1.0725 | 1.2856 | 1.5919 | 0.9998 |

Mean ΔE below ~1 is barely perceptible; 1-3 is a noticeable but small difference; above ~5 the finetuned LUT visibly changes scene colours.

## 4. Specialization scores (ranked)

| scene | distance_from_base_euclidean | mean_abs_magnitude | adjustment_std | max_magnitude | mean_abs_delta_r | mean_abs_delta_g | mean_abs_delta_b | specialization_score |
|---|---|---|---|---|---|---|---|---|
| cluster_6 | 1.0235 | 0.0084 | 0.0023 | 0.0204 | 0.0057 | 0.0074 | 0.0122 | 441.7858 |
| cluster_0 | 0.9405 | 0.0074 | 0.0021 | 0.0186 | 0.0028 | 0.0087 | 0.0108 | 439.7140 |
| cluster_1 | 1.0530 | 0.0088 | 0.0024 | 0.0214 | 0.0063 | 0.0076 | 0.0124 | 435.8225 |
| cluster_3 | 1.0738 | 0.0087 | 0.0027 | 0.0222 | 0.0057 | 0.0073 | 0.0132 | 401.9756 |
| cluster_2 | 1.0952 | 0.0088 | 0.0028 | 0.0225 | 0.0058 | 0.0068 | 0.0138 | 396.9492 |
| cluster_5 | 1.1067 | 0.0088 | 0.0029 | 0.0228 | 0.0058 | 0.0065 | 0.0142 | 386.1394 |
| cluster_7 | 1.0665 | 0.0084 | 0.0028 | 0.0223 | 0.0043 | 0.0073 | 0.0136 | 374.7577 |
| cluster_4 | 0.8006 | 0.0052 | 0.0027 | 0.0157 | 0.0015 | 0.0026 | 0.0116 | 295.8273 |

- Coefficient of variation of scores: 0.113 (> 0.3 suggests real differences between scenes).
- Most specialized: cluster_6 (score 441.786).
- Least specialized: cluster_4 (score 295.827).

## 5. Adjustments vs scene image statistics

Pearson / Spearman correlations between per-cluster LUT adjustments and scene luminance / colour statistics (n = 8 clusters - treat p-values with caution).

| adjustment_stat | scene_stat | pearson_r | pearson_p | spearman_rho | spearman_p |
|---|---|---|---|---|---|
| max_magnitude | g_std | 0.9407 | 0.0005 | 0.9048 | 0.0020 |
| mean_abs_delta_b | luminance_std | 0.9384 | 0.0006 | 0.9048 | 0.0020 |
| max_magnitude | hist_entropy_r | 0.9358 | 0.0006 | 0.7857 | 0.0208 |
| mean_abs_delta_b | g_std | 0.9294 | 0.0008 | 0.8810 | 0.0039 |
| adjustment_std | b_mean | 0.9220 | 0.0011 | 0.7857 | 0.0208 |
| max_magnitude | luminance_std | 0.9169 | 0.0013 | 0.9286 | 0.0009 |
| mean_abs_delta_r | hist_entropy_g | 0.9163 | 0.0014 | 0.8095 | 0.0149 |
| adjustment_std | hist_entropy_b | 0.9120 | 0.0016 | 0.8333 | 0.0102 |
| max_magnitude | g_mean | 0.8995 | 0.0024 | 0.8095 | 0.0149 |
| max_magnitude | luminance_mean | 0.8855 | 0.0034 | 0.8095 | 0.0149 |

## 6. Caveats

- LUTs are 16^3 PyTorch checkpoints (the original prompt assumed 17^3 .npy; the loader supports both formats).
- Only 9 LUTs exist, so t-SNE on LUT vectors is unstable; the PCA companion and per-image scene-feature embedding should be preferred for interpretation.
- Correlations use n = 8 clusters - indicative only, p-values are unreliable at this sample size.

## 7. Conclusion

Adjustments are small and uniform across scenes: finetuning produced a generic, near-shared LUT rather than scene-specific ones.
