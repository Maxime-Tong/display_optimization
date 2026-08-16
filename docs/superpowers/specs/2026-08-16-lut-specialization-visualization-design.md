# LUT Specialization Visualization Suite — Design

Date: 2026-08-16
Status: Approved (approach A)

## Goal

Determine whether the scene-finetuned LUTs produced by `screen_adaptor` are
genuinely specialized to their scenes or statistically indistinguishable from
the base (generic) LUT. Deliver a self-contained analysis/visualization suite in
`visualization/` that produces figures, CSVs, and a written summary.

## Verified data layout (differs from the original prompt)

- LUTs are PyTorch checkpoints (`{"lut": tensor}`), shape `[16, 16, 16, 3]`,
  `float32` in `[0, 1]` — **not** 17³ `.npy` as the prompt assumed.
- Base LUT: `screen_adaptor/outputs/base_lut.pt`.
- Finetuned LUTs: `screen_adaptor/outputs/luts/cluster_{0..7}_lut.pt`
  (8 clusters).
- "Scene" in this framework == DKL-feature cluster. Each cluster has one
  finetuned LUT and a set of assigned images (8 datasets, e.g. `cf`,
  `genshin_impact`, `sgame0`).
- Scene metadata: `screen_adaptor/outputs/scene_manifest.json`
  (prototypes, 9-dim normalized DKL features, feature mean/std) and
  `scene_manifest.assignments.json` (image path → cluster index).
- Runtime: `D:\miniconda3\envs\3dgs\python.exe` has numpy/matplotlib/seaborn/
  scipy/torch/PIL. scikit-learn and colour-science must be installed (user
  approved).

## Approach (chosen: A)

Full suite as specified in the user prompt, run against the latest `outputs/`
run. Loader is resolution-agnostic so 17³ `.npy` files also work if config is
pointed at them. t-SNE is the primary embedding; a PCA companion and a
per-image scene-feature embedding are added because 9 LUT points alone is
statistically thin. UMAP is optional (attempted import, skipped with a note if
unavailable).

## Components

```
visualization/
  config.yaml          # paths, metric/plot parameters, image sampling
  utils.py             # LUT loading, distance metrics, CIEDE2000, image stats
  visualize_lut.py     # orchestration: metrics -> ΔE -> specialization -> report
  run_analysis.sh      # env-aware execution wrapper
  requirements.txt     # pinned dependency list
  figures/             # output PNGs (created at runtime)
  data/                # output CSVs (created at runtime)
  report.md            # generated summary
```

### utils.py

- `load_lut(path)` — accepts `.pt` (dict or raw tensor), `.npy`, `.npz`;
  returns `[N, N, N, 3]` float32.
- `flatten_lut(lut)` — `[N³, 3]`.
- `pairwise_distances(luts, metrics)` — Euclidean / Manhattan / cosine via
  scikit-learn on flattened LUT vectors; returns symmetric matrices.
- `compute_delta_e(base, finetuned)` — per-grid-point CIEDE2000 (colour-science,
  sRGB → Lab, D65) between base and finetuned output colors at matching grid
  coordinates.
- `adjustment_stats(lut, base)` — per-point adjustment magnitudes, mean abs
  delta per channel, std of per-point magnitudes.
- `scene_image_stats(cluster_paths, ...)` — per-cluster mean luminance and std,
  mean RGB histogram (configurable bins), and DKL features reusing
  `screen_adaptor.src.screen_adaptor.scene_matcher` helpers.
- `correlation_table(...)` — Pearson r and Spearman rho (with p-values) between
  per-cluster adjustment stats and scene image stats.

### visualize_lut.py

1. Load config; create `figures/` and `data/`; locate LUTs (auto-discover
   `cluster_{i}_lut.pt` from `lut_dir`).
2. **Distance metrics**: pairwise distance heatmap (3 metrics), grouped bar
   chart of distance-from-base, t-SNE embedding of LUTs + PCA companion,
   per-image scene-feature t-SNE colored by cluster.
3. **ΔE**: per-scene 3D scatter (input RGB position, color = ΔE), 2D mid-plane
   slice heatmaps (R/G/B at 0.5), overlaid ΔE histograms; summary stats per
   scene (mean, p95, max, std).
4. **Specialization**: score = `||F_i − B||₂ / (std_p(||F_i(p) − B(p)||₂) + ε)`;
   ranking table, bar chart, scatter of adjustment magnitude vs. scene
   luminance, correlation table.
5. **Report**: `report.md` with tables, rankings, interpretation, and caveats
   (16³ data, n=8 correlation, 9-point t-SNE).

## Outputs

`figures/`: `distance_heatmap.png`, `distance_from_base_bar.png`,
`lut_embedding_tsne.png`, `lut_embedding_pca.png`, `scene_feature_tsne.png`,
`delta_e_3d_scene_{i}.png`, `delta_e_slices_scene_{i}.png`,
`delta_e_histograms.png`, `specialization_scores.png`,
`correlation_luminance.png`, `correlation_matrix.png`.

`data/`: `distances.csv`, `delta_e_per_point.csv`, `delta_e_stats.csv`,
`specialization_scores.csv`, `scene_image_stats.csv`, `correlations.csv`.

`report.md`: generated summary.

## Error handling

- Missing input files → explicit `FileNotFoundError` with expected path.
- Base/finetuned grid mismatch → raise with shapes.
- Missing scikit-learn/colour-science → actionable install message.
- UMAP import failure → skip UMAP figure, note in report (no crash).
- `run_analysis.sh` → `set -euo pipefail`; resolves script directory; honors
  `PYTHON` env var, defaults to the 3dgs env.

## Verification

- Base-vs-base distance == 0 (sanity check).
- All CSVs/figures produced; report generated.
- Spot-check ΔE stats and specialization ranking against raw numpy values.
- Script is rerunnable and idempotent (overwrites outputs).

## Out of scope

- Modifying `screen_adaptor` training/inference code.
- Comparative analysis across older runs (`luts_1E-4`, `outputs-ori`) — config
  paths make this possible later, but approach C was not chosen.

## Dependencies

numpy, scipy, matplotlib, seaborn, scikit-learn, colour-science, torch (for
`.pt` loading), Pillow (image stats). Installed into the `3dgs` conda env.
