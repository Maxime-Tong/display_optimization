#!/usr/bin/env python
"""LUT specialization analysis and visualization suite.

Pipeline
--------
1. Distance metrics   -- pairwise Euclidean / Manhattan / cosine distances
                         between the base LUT and every finetuned LUT, shown
                         as heatmaps, a distance-from-base bar chart, and
                         t-SNE / PCA embeddings.
2. Colour difference  -- per-grid-point CIEDE2000 vs the base LUT, shown as
                         3D scatter plots, 2D mid-plane slice heatmaps and
                         histograms.
3. Specialization     -- score = distance_from_base / (adjustment_std + eps),
                         plus correlations between LUT adjustments and the
                         per-cluster scene image statistics.
4. Report             -- ``report.md`` highlighting the most/least specialized
                         scenes.

Usage
-----
    python visualize_lut.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils


def load_config(config_path: str | Path) -> dict:
    """Load config.yaml and resolve relative paths against its directory."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.resolve().parent
    for key in (
        "base_lut",
        "lut_dir",
        "manifest",
        "assignments",
        "figures",
        "data",
        "report",
    ):
        raw = config.get("paths", {}).get(key)
        if raw:
            path = Path(raw)
            config["paths"][key] = str(
                path.resolve() if path.is_absolute() else (base_dir / path).resolve()
            )
    return config


def plot_distance_metrics(
    names: list[str],
    matrices: dict[str, np.ndarray],
    out_dir: Path,
    dpi: int,
) -> None:
    """Heatmap of every pairwise metric + bar chart of distance from base."""
    # --- Pairwise distance heatmaps (1 row, one panel per metric) -----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (metric, matrix) in zip(axes, matrices.items()):
        sns.heatmap(
            matrix,
            xticklabels=names,
            yticklabels=names,
            annot=True,
            fmt=".3g",
            cmap="viridis",
            ax=ax,
            cbar_kws={"label": metric},
        )
        ax.set_title(f"{metric} distance")
    fig.suptitle("Pairwise LUT distances (base + finetuned clusters)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    utils.save_fig(fig, out_dir / "distance_heatmap.png", dpi=dpi)

    # --- Distance from base (bar chart, grouped by metric) -----------------
    fig, ax = plt.subplots(figsize=(11, 6))
    cluster_names = names[1:]
    x = np.arange(len(cluster_names))
    width = 0.27
    for offset, (metric, matrix) in enumerate(matrices.items()):
        values = matrix[0, 1:]  # row 0 == base
        ax.bar(x + (offset - 1) * width, values, width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_names, rotation=45, ha="right")
    ax.set_ylabel("distance from base LUT")
    ax.set_title("Distance of each finetuned LUT from the base LUT")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    utils.save_fig(fig, out_dir / "distance_from_base_bar.png", dpi=dpi)


def plot_lut_embeddings(
    vectors: np.ndarray,
    names: list[str],
    out_dir: Path,
    config: dict,
) -> None:
    """t-SNE and PCA embeddings; each LUT is one point in colour space."""
    embedding_cfg = config["embedding"]
    random_state = int(embedding_cfg.get("random_state", 42))
    perplexity = embedding_cfg.get("perplexity", "auto")
    if perplexity == "auto":
        perplexity = min(30.0, (vectors.shape[0] - 1) / 3.0)

    palette = plt.get_cmap("tab10")
    point_colors = ["black"] + [palette(i) for i in range(len(names) - 1)]

    def _annotate(ax, embedding, title):
        for i, name in enumerate(names):
            marker = "*" if name == "base" else "o"
            size = 260 if name == "base" else 90
            ax.scatter(
                embedding[i, 0],
                embedding[i, 1],
                s=size,
                c=[point_colors[i]],
                marker=marker,
                edgecolors="white",
                linewidths=0.8,
                zorder=5 if name == "base" else 3,
            )
            ax.annotate(
                name,
                (embedding[i, 0], embedding[i, 1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    method = str(embedding_cfg.get("method", "auto")).lower()

    # --- t-SNE (primary embedding) -----------------------------------------
    if method in ("auto", "tsne", "umap"):
        from sklearn.manifold import TSNE

        tsne_embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
        ).fit_transform(vectors)
        fig, ax = plt.subplots(figsize=(9, 7))
        _annotate(
            ax,
            tsne_embedding,
            f"t-SNE embedding of LUTs (perplexity={perplexity:.1f})",
        )
        fig.tight_layout()
        utils.save_fig(fig, out_dir / "lut_embedding_tsne.png", dpi=config["plot"]["dpi"])

    # --- UMAP (only when explicitly requested and installed) ---------------
    if method == "umap":
        try:
            import umap  # type: ignore

            umap_embedding = umap.UMAP(random_state=random_state).fit_transform(vectors)
            fig, ax = plt.subplots(figsize=(9, 7))
            _annotate(ax, umap_embedding, "UMAP embedding of LUTs")
            fig.tight_layout()
            utils.save_fig(
                fig, out_dir / "lut_embedding_umap.png", dpi=config["plot"]["dpi"]
            )
        except Exception:
            print("[warn] umap-learn not installed - skipping UMAP figure")

    # --- PCA companion (more stable for tiny sample sizes) -----------------
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=random_state).fit(vectors)
    pca_embedding = pca.transform(vectors)
    explained = pca.explained_variance_ratio_.sum()
    fig, ax = plt.subplots(figsize=(9, 7))
    _annotate(
        ax,
        pca_embedding,
        f"PCA embedding of LUTs (explained variance = {explained:.1%})",
    )
    fig.tight_layout()
    utils.save_fig(fig, out_dir / "lut_embedding_pca.png", dpi=config["plot"]["dpi"])


def plot_scene_feature_embedding(
    cluster_paths: dict[int, list[str]],
    prototypes: list[dict],
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    out_dir: Path,
    config: dict,
) -> None:
    """t-SNE of per-image scene features, colored by cluster assignment.

    This answers the "are the clusters really distinct" side of the question:
    if the finetuned LUTs are specialized, the underlying scene clusters should
    form visible groups in feature space.
    """
    image_cfg = config["image_stats"]
    max_per_cluster = int(image_cfg.get("max_images_per_cluster", 60))
    max_size = int(image_cfg.get("max_size", 128))
    random_state = int(config["embedding"].get("random_state", 42))
    rng = np.random.default_rng(random_state)

    features, labels = [], []
    use_dkl = utils.HAS_DKL and feat_mean.size == 9
    for cluster, paths in sorted(cluster_paths.items()):
        sample = sorted(paths)
        if len(sample) > max_per_cluster:
            sample = list(rng.choice(sample, size=max_per_cluster, replace=False))
        for path in sample:
            rgb = utils.load_image_rgb(path, max_size=max_size)
            feature = utils.dkl_fast_feature(rgb) if use_dkl else utils.fallback_feature(rgb)
            if feature is None:
                continue
            features.append(feature)
            labels.append(cluster)

    if len(features) < 20:
        print("[warn] too few scene images for feature embedding - skipping")
        return

    features = np.stack(features).astype(np.float32)
    labels = np.asarray(labels, dtype=int)

    # Normalize with the manifest statistics so prototype centers live in the
    # same space as the sampled image features.
    if use_dkl:
        normalized = (features - feat_mean) / np.where(feat_std < 1e-8, 1.0, feat_std)
        proto_features = np.stack(
            [proto["feature"] for proto in prototypes], axis=0
        ).astype(np.float32)
        combined = np.vstack([normalized, proto_features])
        is_prototype = np.zeros(combined.shape[0], dtype=bool)
        is_prototype[len(normalized) :] = True
    else:
        combined = features
        is_prototype = np.zeros(features.shape[0], dtype=bool)

    from sklearn.manifold import TSNE

    perplexity = min(30.0, (combined.shape[0] - 1) / 3.0)
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
    ).fit_transform(combined)

    fig, ax = plt.subplots(figsize=(11, 8))
    palette = plt.get_cmap("tab10")
    n_clusters = len(set(labels))
    # Sample points occupy the first rows; prototype rows are appended after.
    sample_embedding = embedding[: len(features)]
    for cluster in range(n_clusters):
        mask = labels == cluster
        ax.scatter(
            sample_embedding[mask, 0],
            sample_embedding[mask, 1],
            s=12,
            alpha=0.6,
            color=palette(cluster),
            label=f"cluster_{cluster} (n={int(mask.sum())})",
        )
    for i in np.where(is_prototype)[0]:
        cluster = int(i - len(features))
        ax.scatter(
            embedding[i, 0],
            embedding[i, 1],
            marker="*",
            s=320,
            color="black",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        ax.annotate(
            f"proto_{cluster}",
            (embedding[i, 0], embedding[i, 1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("t-SNE of sampled scene images (color = cluster, * = prototype)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    utils.save_fig(fig, out_dir / "scene_feature_tsne.png", dpi=config["plot"]["dpi"])


def plot_delta_e(
    base: np.ndarray,
    cluster_luts: dict[str, np.ndarray],
    out_dir: Path,
    data_dir: Path,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    """Per-scene Delta E analysis: 3D scatter, slices, histogram + CSVs."""
    n = base.shape[0]
    metric = config["analysis"]["delta_e_metric"]
    indices, values = utils.lut_grid(n)
    slice_value = float(config["analysis"].get("slice_value", 0.5))
    slice_index = int(round(slice_value * (n - 1)))
    max_points = int(config["plot"].get("max_points_3d", 4096))
    rng = np.random.default_rng(42)

    all_delta = {}
    stats_rows = []
    per_point_frames = []

    for i, (name, lut) in enumerate(cluster_luts.items()):
        delta = utils.compute_delta_e(base, lut, metric=metric)
        all_delta[name] = delta

        stats_rows.append(
            {
                "scene": name,
                "mean": float(delta.mean()),
                "std": float(delta.std()),
                "p50": float(np.percentile(delta, 50)),
                "p95": float(np.percentile(delta, 95)),
                "p99": float(np.percentile(delta, 99)),
                "max": float(delta.max()),
                "nonzero_frac": float((delta > 0.01).mean()),
            }
        )
        per_point_frames.append(
            pd.DataFrame(
                {
                    "scene": name,
                    "idx": np.arange(delta.size),
                    "r_idx": indices[:, 0],
                    "g_idx": indices[:, 1],
                    "b_idx": indices[:, 2],
                    "r": values[:, 0],
                    "g": values[:, 1],
                    "b": values[:, 2],
                    "delta_e": delta,
                }
            )
        )

        # --- 3D scatter: input color position, colored by Delta E ----------
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        if delta.size > max_points:
            selected = rng.choice(delta.size, size=max_points, replace=False)
        else:
            selected = np.arange(delta.size)
        scatter = ax.scatter(
            values[selected, 0],
            values[selected, 1],
            values[selected, 2],
            c=delta[selected],
            cmap="viridis",
            s=8,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.7, label=f"ΔE {metric}")
        ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
        ax.set_title(f"Per-point ΔE vs base LUT - {name}")
        utils.save_fig(fig, out_dir / f"delta_e_3d_scene_{i}.png", dpi=config["plot"]["dpi"])

        # --- 2D mid-plane slices: R/G/B fixed at `slice_value` -------------
        # With meshgrid(..., indexing="ij") and flattened order R (outer),
        # G (middle), B (inner), fixing one channel leaves the other two in a
        # known order: rows are the slower-varying channel, columns the faster.
        slice_axes = {"R": ("B", "G"), "G": ("B", "R"), "B": ("G", "R")}
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for channel, ax in enumerate(axes):
            mask = indices[:, channel] == slice_index
            slice_grid = delta[mask].reshape(n, n)
            im = ax.imshow(
                slice_grid,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap="viridis",
            )
            channel_name = "RGB"[channel]
            x_label, y_label = slice_axes[channel_name]
            ax.set_title(f"{name} - ΔE at {channel_name}={slice_value:.2f}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            fig.colorbar(im, ax=ax, label=f"ΔE {metric}")
        fig.tight_layout()
        utils.save_fig(fig, out_dir / f"delta_e_slices_scene_{i}.png", dpi=config["plot"]["dpi"])

    stats_df = pd.DataFrame(stats_rows)
    per_point_df = pd.concat(per_point_frames, ignore_index=True)
    stats_df.to_csv(data_dir / "delta_e_stats.csv", index=False)
    per_point_df.to_csv(data_dir / "delta_e_per_point.csv", index=False)

    # --- Overlaid Delta E histograms for every scene -----------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, delta in all_delta.items():
        ax.hist(
            delta,
            bins=40,
            alpha=0.55,
            label=f"{name} (μ={delta.mean():.2f}, p95={np.percentile(delta, 95):.2f})",
        )
    ax.set_xlabel(f"ΔE {metric} vs base LUT")
    ax.set_ylabel("grid points")
    ax.set_yscale("log")
    ax.set_title("Per-scene ΔE histograms vs the base LUT")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    utils.save_fig(fig, out_dir / "delta_e_histograms.png", dpi=config["plot"]["dpi"])

    return stats_df, per_point_df, list(all_delta.values())


def plot_specialization(
    names: list[str],
    matrices: dict[str, np.ndarray],
    cluster_luts: dict[str, np.ndarray],
    base: np.ndarray,
    out_dir: Path,
    data_dir: Path,
    config: dict,
) -> pd.DataFrame:
    """Specialization score = distance_from_base / (adjustment_std + eps)."""
    epsilon = float(config["analysis"].get("epsilon", 1e-6))
    rows = []
    for i, name in enumerate(names[1:]):
        distance = float(matrices["euclidean"][0, i + 1])  # row 0 == base
        adjustment = utils.adjustment_stats(base, cluster_luts[name])
        score = distance / (adjustment["adjustment_std"] + epsilon)
        rows.append(
            {
                "scene": name,
                "distance_from_base_euclidean": distance,
                "mean_abs_magnitude": adjustment["mean_abs_magnitude"],
                "adjustment_std": adjustment["adjustment_std"],
                "max_magnitude": adjustment["max_magnitude"],
                "mean_abs_delta_r": adjustment["mean_abs_delta_r"],
                "mean_abs_delta_g": adjustment["mean_abs_delta_g"],
                "mean_abs_delta_b": adjustment["mean_abs_delta_b"],
                "specialization_score": score,
            }
        )

    score_df = pd.DataFrame(rows)
    score_df = score_df.sort_values("specialization_score", ascending=False).reset_index(drop=True)
    score_df.to_csv(data_dir / "specialization_scores.csv", index=False)

    # --- Bar chart of scores, ranked, plus a distance-vs-std scatter -------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    order = score_df["scene"].tolist()
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(order)))
    bars = ax.bar(order, score_df["specialization_score"], color=colors)
    ax.bar_label(bars, fmt="%.3g", fontsize=8)
    ax.set_ylabel("specialization score")
    ax.set_title("distance_from_base / (adjustment_std + ε)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    for _, row in score_df.iterrows():
        ax.scatter(
            row["adjustment_std"],
            row["distance_from_base_euclidean"],
            s=90,
            label=row["scene"],
        )
        ax.annotate(
            row["scene"],
            (row["adjustment_std"], row["distance_from_base_euclidean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("std of per-point adjustment magnitude")
    ax.set_ylabel("Euclidean distance from base")
    ax.set_title("Adjustment size vs spread (low spread + high distance = specialized)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    utils.save_fig(fig, out_dir / "specialization_scores.png", dpi=config["plot"]["dpi"])

    return score_df


def plot_correlations(
    score_df: pd.DataFrame,
    scene_df: pd.DataFrame,
    out_dir: Path,
    data_dir: Path,
    config: dict,
) -> pd.DataFrame:
    """Correlate per-scene LUT adjustments with scene image statistics."""
    adjustment_cols = [
        "mean_abs_magnitude",
        "adjustment_std",
        "max_magnitude",
        "mean_abs_delta_r",
        "mean_abs_delta_g",
        "mean_abs_delta_b",
    ]
    scene_cols = [
        "luminance_mean",
        "luminance_std",
        "r_mean",
        "g_mean",
        "b_mean",
        "r_std",
        "g_std",
        "b_std",
        "saturation_mean",
        "hist_entropy_r",
        "hist_entropy_g",
        "hist_entropy_b",
    ]

    # Both frames are sorted by cluster index for a row-aligned join.
    adjustment = score_df.sort_values("scene").reset_index(drop=True)
    scene = scene_df.sort_values("cluster").reset_index(drop=True)
    correlation_df = utils.build_correlation_table(
        adjustment, scene, adjustment_cols, scene_cols
    )
    correlation_df.to_csv(data_dir / "correlations.csv", index=False)

    # --- Scatter: adjustment magnitude vs luminance / saturation -----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    pairs = [
        ("luminance_mean", "mean_abs_magnitude", "mean scene luminance"),
        ("saturation_mean", "mean_abs_magnitude", "mean scene saturation"),
    ]
    for ax, (scene_col, adj_col, label) in zip(axes, pairs):
        x = scene[scene_col].to_numpy(float)
        y = adjustment[adj_col].to_numpy(float)
        ax.scatter(x, y, s=90)
        for i, scene_name in enumerate(adjustment["scene"]):
            ax.annotate(
                scene_name, (x[i], y[i]), xytext=(5, 5), textcoords="offset points", fontsize=8
            )
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, slope * xs + intercept, "--", color="gray", alpha=0.8)
        result = utils.pearson_spearman(x, y)
        ax.set_xlabel(label)
        ax.set_ylabel("mean |adjustment|")
        ax.set_title(f"r = {result['pearson_r']:.2f}  (p = {result['pearson_p']:.3f})")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    utils.save_fig(fig, out_dir / "correlation_luminance.png", dpi=config["plot"]["dpi"])

    # --- Correlation matrix heatmap ----------------------------------------
    pivot = correlation_df.pivot(
        index="adjustment_stat", columns="scene_stat", values="pearson_r"
    )
    pivot = pivot.reindex(index=adjustment_cols, columns=scene_cols)
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Correlation between LUT adjustments and scene image statistics")
    fig.tight_layout()
    utils.save_fig(fig, out_dir / "correlation_matrix.png", dpi=config["plot"]["dpi"])

    return correlation_df


def build_report(
    config: dict,
    names: list[str],
    matrices: dict[str, np.ndarray],
    grid_n: int,
    delta_stats: pd.DataFrame,
    score_df: pd.DataFrame,
    correlation_df: pd.DataFrame | None,
    scene_df: pd.DataFrame | None,
    notes: list[str],
) -> str:
    """Assemble report.md from every computed table."""
    base_name = names[0]
    cluster_names = names[1:]

    # Distance-from-base table (row 0 of each matrix).
    distance_rows = []
    for i, name in enumerate(cluster_names):
        distance_rows.append(
            {
                "scene": name,
                **{m: float(matrices[m][0, i + 1]) for m in matrices},
            }
        )
    distance_df = pd.DataFrame(distance_rows).sort_values(
        "euclidean", ascending=False
    ).reset_index(drop=True)

    lines = [
        "# LUT Specialization Analysis Report",
        "",
        f"Generated on {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "",
        "## 1. Overview",
        "",
        f"- Base LUT: `{config['paths']['base_lut']}`",
        f"- Finetuned LUTs: `{config['paths']['lut_dir']}` "
        f"({len(cluster_names)} clusters discovered)",
        f"- Grid resolution: {grid_n} x {grid_n} x {grid_n} x 3",
        f"- Delta-E metric: {config['analysis']['delta_e_metric']}",
        f"- Specialization score: distance_from_base / (adjustment_std + ε), "
        f"ε = {float(config['analysis']['epsilon']):g}",
        "",
        "## 2. Distance from base LUT",
        "",
        utils.df_to_markdown(distance_df),
        "",
        "If all finetuned LUTs sit at nearly the same distance from the base and "
        "the pairwise distances between them are small, the finetuning produced "
        "a generic (shared) adjustment rather than scene-specific ones.",
        "",
    ]

    lines += [
        "## 3. Per-scene colour difference (ΔE vs base)",
        "",
        utils.df_to_markdown(delta_stats.sort_values("mean", ascending=False)),
        "",
        "Mean ΔE below ~1 is barely perceptible; 1-3 is a noticeable but small "
        "difference; above ~5 the finetuned LUT visibly changes scene colours.",
        "",
    ]

    spread = 0.0
    lines += [
        "## 4. Specialization scores (ranked)",
        "",
        utils.df_to_markdown(score_df),
        "",
    ]
    if len(score_df) > 1:
        scores = score_df["specialization_score"].to_numpy(float)
        spread = float(np.std(scores) / (np.mean(scores) + 1e-12))
        lines += [
            f"- Coefficient of variation of scores: {spread:.3f} "
            "(> 0.3 suggests real differences between scenes).",
            f"- Most specialized: {score_df.iloc[0]['scene']} "
            f"(score {score_df.iloc[0]['specialization_score']:.3f}).",
            f"- Least specialized: {score_df.iloc[-1]['scene']} "
            f"(score {score_df.iloc[-1]['specialization_score']:.3f}).",
            "",
        ]

    if correlation_df is not None and len(correlation_df):
        top = correlation_df.head(10).copy()
        for col in ("pearson_r", "pearson_p", "spearman_rho", "spearman_p"):
            top[col] = top[col].round(4)
        lines += [
            "## 5. Adjustments vs scene image statistics",
            "",
            "Pearson / Spearman correlations between per-cluster LUT "
            "adjustments and scene luminance / colour statistics (n = "
            f"{len(score_df)} clusters - treat p-values with caution).",
            "",
            utils.df_to_markdown(top),
            "",
        ]

    lines += ["## 6. Caveats", ""]
    lines += [f"- {note}" for note in notes]
    lines += [
        "",
        "## 7. Conclusion",
        "",
    ]
    # Quantitative heuristic for the summary verdict.
    mean_de = float(delta_stats["mean"].mean())
    max_de = float(delta_stats["mean"].max())
    min_de = float(delta_stats["mean"].min())
    max_dist = float(distance_df["euclidean"].max())
    if max_dist < 1e-3:
        verdict = (
            "The finetuned LUTs are essentially identical to the base LUT "
            "(Euclidean distance < 1e-3)."
        )
    elif mean_de < 1.0 and (max_de - min_de) < 1.0:
        verdict = (
            "Adjustments are small and uniform across scenes: finetuning "
            "produced a generic, near-shared LUT rather than scene-specific ones."
        )
    elif spread > 0.3 and max_de > 2.0:
        verdict = (
            "Adjustments vary strongly between scenes with perceptible ΔE: "
            "the finetuned LUTs appear genuinely scene-specialized."
        )
    else:
        verdict = (
            "Mixed evidence: some scenes show stronger, more perceptible "
            "adjustments while others stay close to the base LUT. "
            "See the tables above for per-scene detail."
        )
    lines.append(verdict)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--skip-image-stats",
        action="store_true",
        help="skip per-cluster scene image statistics and correlations",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    figure_dir = Path(paths["figures"])
    data_dir = Path(paths["data"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    random_state = int(config["embedding"].get("random_state", 42))
    np.random.seed(random_state)

    # ------------------------------------------------------------------ 1. LUTs
    print("[1/5] Loading LUTs ...")
    base = utils.load_lut(paths["base_lut"])
    lut_paths = utils.discover_lut_paths(paths["lut_dir"], paths.get("lut_pattern", ""))
    cluster_luts = {}
    for index, path in enumerate(lut_paths):
        cluster_luts[f"cluster_{index}"] = utils.load_lut(path)

    names = ["base"] + list(cluster_luts.keys())
    for name, lut in cluster_luts.items():
        if lut.shape != base.shape:
            raise ValueError(
                f"Grid mismatch: base {base.shape} vs {name} {lut.shape}"
            )
    grid_n = base.shape[0]
    print(f"  base + {len(cluster_luts)} finetuned LUTs, grid {grid_n}^3")

    # Sanity check: distance of the base LUT to itself must be zero.
    matrices = utils.pairwise_distances([base] + list(cluster_luts.values()))
    for metric, matrix in matrices.items():
        if not np.allclose(matrix[0, 0], 0.0, atol=1e-9):
            print(f"[warn] base-to-base {metric} distance != 0: {matrix[0, 0]:.3g}")

    # ------------------------------------------------------------- 2. Metrics
    print("[2/5] Distance metrics + embeddings ...")
    plot_distance_metrics(names, matrices, figure_dir, config["plot"]["dpi"])

    # Each LUT becomes one sample: base first, then the clusters in order.
    vectors = np.vstack(
        [base.reshape(1, -1)]
        + [lut.reshape(1, -1) for lut in cluster_luts.values()]
    )
    plot_lut_embeddings(vectors, names, figure_dir, config)

    distances_rows = []
    for metric, matrix in matrices.items():
        for i, name_a in enumerate(names):
            for j, name_b in enumerate(names):
                distances_rows.append(
                    {"metric": metric, "lut_a": name_a, "lut_b": name_b, "distance": float(matrix[i, j])}
                )
    pd.DataFrame(distances_rows).to_csv(data_dir / "distances.csv", index=False)

    # ----------------------------------------------------------------- 3. ΔE
    print("[3/5] Colour difference (ΔE) ...")
    delta_stats, _, _ = plot_delta_e(
        base, cluster_luts, figure_dir, data_dir, config
    )

    # ----------------------------------------------------- 4. Specialization
    print("[4/5] Specialization scores + correlations ...")
    score_df = plot_specialization(
        names, matrices, cluster_luts, base, figure_dir, data_dir, config
    )

    correlation_df = None
    scene_df = None
    notes = []
    if not args.skip_image_stats and config["image_stats"].get("enabled", True):
        manifest = Path(paths["manifest"])
        assignments_file = Path(paths["assignments"])
        if manifest.exists() and assignments_file.exists():
            prototypes, feat_mean, feat_std = utils.load_manifest(manifest)
            assignments = utils.load_assignments(assignments_file)
            cluster_paths = utils.cluster_image_paths(assignments, len(cluster_luts))
            print(f"  scene images found: {sum(len(v) for v in cluster_paths.values())}")

            image_cfg = config["image_stats"]
            scene_df = utils.compute_scene_stats(
                cluster_paths,
                max_images=int(image_cfg.get("max_images_per_cluster", 60)),
                max_size=int(image_cfg.get("max_size", 128)),
                bins=int(image_cfg.get("histogram_bins", 32)),
                seed=int(image_cfg.get("sample_seed", 0)),
            )
            scene_df.to_csv(data_dir / "scene_image_stats.csv", index=False)
            correlation_df = plot_correlations(
                score_df, scene_df, figure_dir, data_dir, config
            )
            plot_scene_feature_embedding(
                cluster_paths, prototypes, feat_mean, feat_std, figure_dir, config
            )
        else:
            notes.append(
                "Manifest/assignments not found - scene image statistics and "
                "correlations were skipped."
            )

    notes.append(
        f"LUTs are {grid_n}^3 PyTorch checkpoints (the original prompt assumed "
        "17^3 .npy; the loader supports both formats)."
    )
    notes.append(
        "Only 9 LUTs exist, so t-SNE on LUT vectors is unstable; the PCA "
        "companion and per-image scene-feature embedding should be preferred "
        "for interpretation."
    )
    notes.append(
        "Correlations use n = 8 clusters - indicative only, p-values are "
        "unreliable at this sample size."
    )

    # ------------------------------------------------------------------ 5. Report
    print("[5/5] Writing report ...")
    report = build_report(
        config,
        names,
        matrices,
        grid_n,
        delta_stats,
        score_df,
        correlation_df,
        scene_df,
        notes,
    )
    Path(paths["report"]).write_text(report, encoding="utf-8")
    print(f"Done. Figures: {figure_dir} | Data: {data_dir} | Report: {paths['report']}")


if __name__ == "__main__":
    main()
