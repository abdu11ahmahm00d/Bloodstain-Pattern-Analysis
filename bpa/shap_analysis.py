"""
shap_analysis.py — CatBoost SHAP analysis for BPA.
Loads saved CatBoost model, computes SHAP values via TreeExplainer,
generates summary plots and feature ranking.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BPA_DIR = Path(__file__).parent.parent
FEATURES_PATH = BPA_DIR / "data" / "features.csv"
MODEL_PATH = BPA_DIR / "models" / "bpa_classifier_final.joblib"
SHAP_DIR = BPA_DIR / "shap_results"

FEATURE_COLS = [
    "axis_ratio_mean",
    "axis_ratio_std",
    "axis_ratio_max",
    "area_mm2_mean",
    "area_mm2_std",
    "area_mm2_max",
    "solidity_mean",
    "solidity_std",
    "solidity_max",
    "eccentricity_mean",
    "eccentricity_std",
    "eccentricity_max",
    "circularity_mean",
    "circularity_std",
    "circularity_max",
    "stain_count",
    "total_stained_area_mm2",
    "stain_density_per_cm2",
    "spatial_spread_x_std",
    "spatial_spread_y_std",
    "spatial_spread_total",
    "nn_dist_mean_mm",
    "nn_dist_std_mm",
    "convex_hull_area_mm2",
    "mean_orientation_deg",
    "circular_variance",
    "convergence_x",
    "convergence_y",
    "convergence_dispersion",
    "has_convergence",
]

CLASS_NAMES = ["gunshot", "impact", "low_velocity"]


def main():
    SHAP_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("BPA — CatBoost SHAP Analysis")
    print("=" * 60)

    # --- Step 1: Load and prepare data ---
    print("\n[1/4] Loading features...")
    df = pd.read_csv(FEATURES_PATH)
    before = len(df)
    df = df[df["stain_count"] > 0]
    n_filtered = before - len(df)

    le = LabelEncoder()
    y = le.fit_transform(df["mechanism_class"])

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    X_clean = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

    print(f"  {len(df)} samples ({n_filtered} filtered for stain_count=0)")
    print(f"  {len(available)} features")
    print(f"  Classes: {dict(zip(le.classes_, np.bincount(y)))}")

    # --- Step 2: Load model ---
    print("\n[2/4] Loading CatBoost model...")
    pipe = joblib.load(MODEL_PATH)
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    print(
        f"  CatBoost: {clf.tree_count_} trees, depth={clf.get_params().get('depth', '?')}"
    )

    X_scaled = scaler.transform(X_clean)

    # --- Step 3: SHAP values ---
    print("\n[3/4] Computing SHAP values...")
    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        n_classes = len(shap_values)
        shap_list = shap_values
        print(f"  SHAP list: {n_classes} classes, each {shap_values[0].shape}")
    elif shap_values.ndim == 3:
        n_classes = shap_values.shape[2]
        shap_list = [shap_values[:, :, i] for i in range(n_classes)]
        print(f"  SHAP array: {shap_values.shape}")
    else:
        n_classes = 1
        shap_list = [shap_values]
        print(f"  SHAP array: {shap_values.shape}")

    # --- Step 4: Plots and output ---
    print("\n[4/4] Generating outputs...")

    # Combined global SHAP
    shap_global = np.mean([np.abs(sv) for sv in shap_list], axis=0)
    mean_shap = np.mean(shap_global, axis=0)

    # Feature ranking CSV
    ranking = sorted(
        zip(available, mean_shap.tolist()), key=lambda x: x[1], reverse=True
    )
    ranking_df = pd.DataFrame(ranking, columns=["feature", "mean_abs_shap"])
    ranking_df.to_csv(SHAP_DIR / "feature_ranking.csv", index=False)

    print("\n  Top 15 Features by Mean |SHAP|:")
    print(f"  {'Rank':>4s}  {'Feature':30s} {'Mean |SHAP|':>10s}")
    print(f"  {'-' * 46}")
    for i, (feat, score) in enumerate(ranking[:15]):
        print(f"  {i + 1:4d}  {feat:30s} {score:10.4f}")

    # Convergence summary
    conv_names = ["convergence_x", "convergence_y", "convergence_dispersion"]
    conv_sum = sum(mean_shap[available.index(f)] for f in conv_names if f in available)
    all_sum = sum(mean_shap)
    conv_pct = (conv_sum / all_sum) * 100
    print(
        f"\n  Convergence features combined: {conv_sum:.4f} ({conv_pct:.1f}% of total)"
    )

    # Global summary beeswarm
    print("\n  Plot: global_summary.png")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_global, X_scaled, feature_names=available, show=False, max_display=20
    )
    plt.title("SHAP Summary Beeswarm (CatBoost)")
    plt.tight_layout()
    plt.savefig(str(SHAP_DIR / "global_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP feature importance bar chart
    print("  Plot: shap_feature_importance.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_features = [r[0] for r in ranking[:15]][::-1]
    plot_values = [r[1] for r in ranking[:15]][::-1]
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(plot_features)))
    ax.barh(plot_features, plot_values, color=colors)
    ax.set_xlabel("Mean |SHAP| (impact on model output)")
    ax.set_title("SHAP Feature Importance (CatBoost)")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.savefig(
        str(SHAP_DIR / "shap_feature_importance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Per-class summary plots
    if n_classes > 1:
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"  Plot: summary_{class_name}.png")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_list[i],
                X_scaled,
                feature_names=available,
                show=False,
                max_display=15,
            )
            plt.title(f"SHAP Values for Class: {class_name}")
            plt.tight_layout()
            plt.savefig(
                str(SHAP_DIR / f"summary_{class_name}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    # Dependence plots for top 3 features
    top3 = [ranking[i][0] for i in range(3)]
    print("\n  Dependence plots for top 3 features...")
    for feat_name in top3:
        feat_idx = available.index(feat_name)
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feat_idx, shap_global, X_scaled, feature_names=available, show=False
        )
        plt.title(f"SHAP Dependence: {feat_name}")
        plt.tight_layout()
        plt.savefig(
            str(SHAP_DIR / f"dependence_{feat_name}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(f"  Plot: dependence_{feat_name}.png")

    # Force plot for a representative sample per class
    print("\n  Force plots (representative samples)...")
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y == i
        if class_mask.sum() > 0:
            idx = np.where(class_mask)[0][0]
            shap_values_class = shap_list[i] if n_classes > 1 else shap_list[0]
            plt.figure(figsize=(20, 3))
            shap.force_plot(
                explainer.expected_value[i]
                if n_classes > 1
                else explainer.expected_value,
                shap_values_class[idx, :],
                X_scaled[idx, :],
                feature_names=available,
                matplotlib=True,
                show=False,
            )
            plt.title(f"Force Plot: {class_name}")
            plt.tight_layout()
            plt.savefig(
                str(SHAP_DIR / f"force_{class_name}.png"), dpi=150, bbox_inches="tight"
            )
            plt.close()
            print(f"  Plot: force_{class_name}.png")

    # Save arrays
    np.save(str(SHAP_DIR / "shap_values.npy"), np.array(shap_list, dtype=object))
    np.save(str(SHAP_DIR / "X_scaled.npy"), X_scaled)

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print(f"  All outputs saved to: {SHAP_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
