"""
golden_path.py — Lean golden path for BPA.
Trains final CatBoost model, computes live CV, runs SHAP.
"""

import sys, time, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from catboost import CatBoostClassifier

# Paths
BPA_DIR = Path(__file__).parent
FEATURES_PATH = BPA_DIR / "data" / "features.csv"
MODEL_DIR = BPA_DIR / "models"
RESULTS_DIR = BPA_DIR / "results"

# Feature columns (from train_classifier.py)
FEATURE_COLS = [
    "axis_ratio_mean", "axis_ratio_std", "axis_ratio_max",
    "area_mm2_mean", "area_mm2_std", "area_mm2_max",
    "solidity_mean", "solidity_std", "solidity_max",
    "eccentricity_mean", "eccentricity_std", "eccentricity_max",
    "circularity_mean", "circularity_std", "circularity_max",
    "stain_count", "total_stained_area_mm2", "stain_density_per_cm2",
    "spatial_spread_x_std", "spatial_spread_y_std", "spatial_spread_total",
    "nn_dist_mean_mm", "nn_dist_std_mm", "convex_hull_area_mm2",
    "mean_orientation_deg", "circular_variance",
    "convergence_x", "convergence_y", "convergence_dispersion",
    "has_convergence",
]

print("=" * 60)
print("BPA GOLDEN PATH")
print("=" * 60)

# --- Step 1: Load and prepare data ---
print("\n[1/4] Loading features...")
df = pd.read_csv(FEATURES_PATH)

# Filter stain_count=0
before = len(df)
df = df[df["stain_count"] > 0]
n_filtered = before - len(df)

le = LabelEncoder()
y = le.fit_transform(df["mechanism_class"])

available_features = [c for c in FEATURE_COLS if c in df.columns]
X = df[available_features].copy()

nan_count = int(X.isna().sum().sum())
print(f"  {len(X)} samples ({n_filtered} filtered for stain_count=0)")
print(f"  {len(available_features)} features, {nan_count} NaN values (CatBoost native handling)")
print(f"  Classes: {dict(zip(le.classes_, np.bincount(y)))}")

# For sklearn CV (StandardScaler doesn't handle NaN), impute
X_clean = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

# --- Step 2: 5-fold Stratified CV ---
print("\n[2/4] 5-fold stratified CV (manual folds)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = np.zeros_like(y)
cv_fold_metrics = []

pipe_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CatBoostClassifier(
        iterations=400, depth=4, learning_rate=0.05,
        l2_leaf_reg=5, auto_class_weights="SqrtBalanced",
        random_state=42, verbose=0,
    ))
])

fold_times = []
fold_metrics_list = []
for fold, (train_idx, test_idx) in enumerate(skf.split(X_clean, y), 1):
    X_tr, X_te = X_clean[train_idx], X_clean[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    t0 = time.time()
    pipe_cv.fit(X_tr, y_tr)
    y_pred_cv[test_idx] = pipe_cv.predict(X_te).ravel()
    t1 = time.time()
    fold_acc = accuracy_score(y_te, y_pred_cv[test_idx])
    fold_f1 = f1_score(y_te, y_pred_cv[test_idx], average="macro")
    fold_times.append(t1 - t0)
    fold_metrics_list.append(fold_acc)
    print(f"  Fold {fold}: {t1-t0:.1f}s, acc={fold_acc:.4f}, f1_macro={fold_f1:.4f}")

cv_accuracy = accuracy_score(y, y_pred_cv)
cv_f1_macro = f1_score(y, y_pred_cv, average="macro", zero_division=0)
cv_f1_weighted = f1_score(y, y_pred_cv, average="weighted", zero_division=0)
cv_report = classification_report(y, y_pred_cv, target_names=le.classes_, zero_division=0, output_dict=True)

per_class_f1 = {}
for cls_name in le.classes_:
    per_class_f1[cls_name] = round(cv_report[cls_name]["f1-score"], 4)

print(f"\n  CV Results: acc={cv_accuracy:.4f}, f1_macro={cv_f1_macro:.4f}, f1_weighted={cv_f1_weighted:.4f}")
print(f"  Per-class F1: {per_class_f1}")
print(f"  Avg fold time: {np.mean(fold_times):.1f}s")

# --- Step 3: Train final model on all data ---
print("\n[3/4] Training final model on all data...")
pipe_final = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CatBoostClassifier(
        iterations=400, depth=4, learning_rate=0.05,
        l2_leaf_reg=5, auto_class_weights="SqrtBalanced",
        random_state=42, verbose=50,
    ))
])
pipe_final.fit(X_clean, y)
y_pred_final = pipe_final.predict(X_clean).ravel()

# Save model
MODEL_DIR.mkdir(exist_ok=True)
model_path = MODEL_DIR / "bpa_classifier_final.joblib"
joblib.dump(pipe_final, model_path)
print(f"  Model saved: {model_path}")

# Feature importance
clf = pipe_final.named_steps["clf"]
importances = clf.feature_importances_
feat_imp = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)

print("\n  Top 10 Feature Importances:")
for name, imp in feat_imp[:10]:
    print(f"    {name:30s}: {imp:.2f}")

# Save metadata
metadata = {
    "model_type": "CatBoost",
    "hyperparameters": {
        "iterations": 400, "depth": 4, "learning_rate": 0.05,
        "l2_leaf_reg": 5, "auto_class_weights": "SqrtBalanced",
    },
    "features": available_features,
    "n_features": len(available_features),
    "classes": le.classes_.tolist(),
    "class_mapping": {i: c for i, c in enumerate(le.classes_)},
    "train_samples": len(X),
    "filtered_stain_count_0": n_filtered,
    "class_distribution": dict(zip(le.classes_, np.bincount(y).tolist())),
    "cv_results": {
        "accuracy": round(float(cv_accuracy), 4),
        "f1_macro": round(float(cv_f1_macro), 4),
        "f1_weighted": round(float(cv_f1_weighted), 4),
        "per_class_f1": per_class_f1,
        "n_folds": 5,
        "fold_metrics": {
            "accuracy": [round(float(a), 4) for a in fold_metrics_list],
        },
    },
    "dataset_composition": df["source_dataset"].value_counts().to_dict(),
}

meta_path = MODEL_DIR / "model_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Metadata saved: {meta_path}")

# Save feature importances
imp_path = MODEL_DIR / "feature_importances.json"
with open(imp_path, "w") as f:
    json.dump([{"feature": n, "importance": float(i)} for n, i in feat_imp], f, indent=2)
print(f"  Feature importances saved: {imp_path}")

# --- Step 4: SHAP Analysis ---
print("\n[4/4] Running SHAP analysis...")
try:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE

    SHAP_DIR = BPA_DIR / "shap_results"
    SHAP_DIR.mkdir(exist_ok=True)

    # SMOTE + XGBoost for SHAP
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_clean, y)
    print(f"  SMOTE: {len(X_clean)} -> {len(X_bal)} samples")

    scaler_shap = StandardScaler()
    X_scaled = scaler_shap.fit_transform(X_bal)
    X_orig_scaled = scaler_shap.transform(X_clean)

    clf_shap = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    clf_shap.fit(X_scaled, y_bal)

    explainer = shap.TreeExplainer(clf_shap)
    shap_values = explainer.shap_values(X_orig_scaled)

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        n_classes = shap_values.shape[2]
        shap_values_list = [shap_values[:, :, i] for i in range(n_classes)]
    elif isinstance(shap_values, list):
        n_classes = len(shap_values)
        shap_values_list = shap_values
    else:
        n_classes = 1
        shap_values_list = [shap_values]

    # Global summary
    print("  Generating global summary...")
    if n_classes > 1:
        shap_combined = np.mean([np.abs(sv) for sv in shap_values_list], axis=0)
    else:
        shap_combined = shap_values_list[0]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_combined, X_orig_scaled, feature_names=available_features, show=False, max_display=20)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(str(SHAP_DIR / "global_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Per-class summary
    if n_classes > 1:
        for i, class_name in enumerate(le.classes_):
            print(f"  Summary for class: {class_name}")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_list[i], X_orig_scaled, feature_names=available_features, show=False, max_display=15)
            plt.title(f"SHAP Values for Class: {class_name}")
            plt.tight_layout()
            plt.savefig(str(SHAP_DIR / f"summary_{class_name}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # Feature ranking CSV
    mean_shap = np.mean(np.abs(shap_combined), axis=0)
    feature_ranking = sorted(zip(available_features, mean_shap.tolist()), key=lambda x: x[1], reverse=True)
    ranking_df = pd.DataFrame(feature_ranking, columns=["feature", "mean_abs_shap"])
    ranking_df.to_csv(str(SHAP_DIR / "feature_ranking.csv"), index=False)

    print("\n  Top 15 Features by Mean |SHAP|:")
    for i, (feat, score) in enumerate(feature_ranking[:15]):
        print(f"    {i+1:2d}. {feat:30s} {score:.4f}")

    np.save(str(SHAP_DIR / "shap_values.npy"), np.array(shap_values_list))
    np.save(str(SHAP_DIR / "X_scaled.npy"), X_scaled)
    print(f"  SHAP values saved to {SHAP_DIR}/")

except ImportError as e:
    print(f"  SHAP skipped (missing package: {e})")

print("\n" + "=" * 60)
print("GOLDEN PATH COMPLETE")
print(f"  Model: {model_path}")
print(f"  CV Accuracy: {cv_accuracy:.4f}")
print(f"  CV F1 Macro: {cv_f1_macro:.4f}")
print("=" * 60)
