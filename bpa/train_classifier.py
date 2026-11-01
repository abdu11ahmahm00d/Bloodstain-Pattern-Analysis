"""
train_classifier.py — XGBoost + baseline classifiers for multi-class BPA classification.

Experiments:
  1. Within-source multi-class (LOO-CV or 5-fold)
  2. Cross-source generalization
  3. Binary ablation (gunshot vs impact)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, cross_val_predict,
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMB = True
except ImportError:
    HAS_IMB = False


# ─── Feature columns (must match feature_extraction.py) ───────────────────

FEATURE_COLS = [
    # Group 1: Stain Morphology (aggregated)
    "axis_ratio_mean", "axis_ratio_std", "axis_ratio_max",
    "area_mm2_mean", "area_mm2_std", "area_mm2_max",
    "solidity_mean", "solidity_std", "solidity_max",
    "eccentricity_mean", "eccentricity_std", "eccentricity_max",
    "circularity_mean", "circularity_std", "circularity_max",
    # Group 2: Pattern Distribution
    "stain_count", "total_stained_area_mm2", "stain_density_per_cm2",
    "spatial_spread_x_std", "spatial_spread_y_std", "spatial_spread_total",
    "nn_dist_mean_mm", "nn_dist_std_mm", "convex_hull_area_mm2",
    # Group 3: Directionality
    "mean_orientation_deg", "circular_variance",
    "convergence_x", "convergence_y", "convergence_dispersion",
    # Group 4: Context (image_width/image_height excluded — camera resolution confounds)
    "has_convergence",
]


def load_features(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """Load features CSV and prepare X, y."""
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing target
    df = df.dropna(subset=["mechanism_class"])
    
    # Filter out stain_count=0 images (44 such images — no morphological signal)
    before = len(df)
    df = df[df["stain_count"] > 0]
    n_filtered = before - len(df)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["mechanism_class"])
    
    # Select feature columns (only those that exist)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()
    
    # NaN handling: convergence features (convergence_x, convergence_y,
    # convergence_dispersion) are NaN for images with <2 stains. CatBoost
    # handles NaN natively — no imputation needed. For sklearn models used
    # in experiments, we fill NaN with column median.
    # The has_convergence boolean flag allows the model to learn that
    # missing convergence values are themselves informative.
    
    print(f"Loaded {len(X)} samples ({n_filtered} filtered for stain_count=0), {len(available_features)} features")
    print(f"Classes: {dict(zip(le.classes_, np.bincount(y)))}")
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"  NaN values remaining: {nan_count} (handled natively by CatBoost)")
    
    return df, X.values, y, le


def get_models() -> Dict[str, Pipeline]:
    """Get all classifier pipelines."""
    models = {}
    
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
            ))
        ])
    
    if HAS_LGB:
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ))
        ])
    
    models["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        ))
    ])
    
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
        ))
    ])
    
    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            random_state=42,
        ))
    ])
    
    return models


def get_weighted_models() -> Dict[str, Pipeline]:
    """Get classifier pipelines with class_weight='balanced'."""
    models = {}
    
    if HAS_XGB:
        models["XGBoost_weighted"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
                scale_pos_weight=1,  # handled by class_weight below
            ))
        ])
    
    models["RandomForest_weighted"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ))
    ])
    
    models["LogisticRegression_weighted"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ))
    ])
    
    models["SVM_weighted"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            random_state=42,
        ))
    ])
    
    return models


def get_smote_models() -> Dict[str, Pipeline]:
    """Get classifier pipelines with SMOTE oversampling."""
    if not HAS_IMB:
        print("WARNING: imbalanced-learn not installed. Run: pip install imbalanced-learn")
        return {}
    
    models = {}
    
    if HAS_XGB:
        models["XGBoost_SMOTE"] = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
            ))
        ])
    
    models["RandomForest_SMOTE"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        ))
    ])
    
    models["LogisticRegression_SMOTE"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
        ))
    ])
    
    models["SVM_SMOTE"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            random_state=42,
        ))
    ])
    
    return models


def get_subsampled_models() -> Dict[str, Pipeline]:
    """Get classifier pipelines with majority class undersampling."""
    if not HAS_IMB:
        print("WARNING: imbalanced-learn not installed. Run: pip install imbalanced-learn")
        return {}
    
    models = {}
    
    # Target: balance all classes to match the largest minority class (impact=30)
    # This preserves more data than matching the smallest class (gunshot=16)
    # Note: In CV folds, minority classes have fewer samples, so we use the
    # min of target and available samples per class
    sampling_strategy = "auto"  # let imblearn handle it
    
    if HAS_XGB:
        models["XGBoost_subsample"] = ImbPipeline([
            ("scaler", StandardScaler()),
            ("under", RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)),
            ("clf", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
            ))
        ])
    
    models["RandomForest_subsample"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("under", RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        ))
    ])
    
    models["LogisticRegression_subsample"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("under", RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
        ))
    ])
    
    models["SVM_subsample"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("under", RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            random_state=42,
        ))
    ])
    
    return models


def experiment_1_within_source(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    n_folds: int = 5,
) -> Dict[str, dict]:
    """
    Experiment 1: Within-source multi-class classification.
    5-fold stratified CV on full dataset.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Within-Source Multi-Class Classification")
    print("="*60)
    
    models = get_models()
    results = {}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Cross-val predictions for full confusion matrix
        y_pred = cross_val_predict(model, X, y, cv=skf)
        
        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average="macro")
        f1_weighted = f1_score(y, y_pred, average="weighted")
        precision = precision_score(y, y_pred, average="macro", zero_division=0)
        recall = recall_score(y, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y, y_pred)
        
        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision,
            "recall_macro": recall,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y, y_pred, target_names=le.classes_, zero_division=0
            ),
        }
        
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  F1 (macro):  {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"\n{classification_report(y, y_pred, target_names=le.classes_, zero_division=0)}")
    
    return results


def experiment_2_cross_source(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
) -> Dict[str, dict]:
    """
    Experiment 2: Cross-source generalization.
    Train on Attinger, test on Kannan (and reverse).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Cross-Source Generalization")
    print("="*60)
    
    attinger_mask = df["source_dataset"].str.contains("attinger")
    kannan_mask = df["source_dataset"] == "kannan"
    
    results = {}
    models = get_models()
    
    for train_source, test_source, train_mask, test_mask in [
        ("Attinger", "Kannan", attinger_mask, kannan_mask),
        ("Kannan", "Attinger", kannan_mask, attinger_mask),
    ]:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"\nSkipping {train_source}->{test_source}: insufficient data")
            continue
        
        # Check if training set has multiple classes
        unique_train = np.unique(y_train)
        if len(unique_train) < 2:
            print(f"\nSkipping {train_source}->{test_source}: training set has only 1 class")
            continue
        
        print(f"\n--- Train: {train_source} ({len(X_train)}) -> Test: {test_source} ({len(X_test)}) ---")
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            key = f"{name}_{train_source}->{test_source}"
            results[key] = {
                "accuracy": acc,
                "f1_macro": f1,
                "confusion_matrix": cm.tolist(),
                "train_source": train_source,
                "test_source": test_source,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            
            print(f"  {name}: acc={acc:.4f}, f1={f1:.4f}")
    
    return results


def experiment_3_binary_ablation(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
) -> Dict[str, dict]:
    """
    Experiment 3: Binary ablation — gunshot vs impact only.
    For comparison with Li & Shen (2025) and Liu et al. (2020).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Binary Ablation (Gunshot vs Impact)")
    print("="*60)
    
    # Find gunshot and impact class indices
    class_names = le.classes_
    binary_classes = []
    for cls in ["gunshot", "impact"]:
        if cls in class_names:
            binary_classes.append(np.where(class_names == cls)[0][0])
    
    if len(binary_classes) < 2:
        print("ERROR: Need both 'gunshot' and 'impact' classes")
        return {}
    
    # Filter to binary
    mask = np.isin(y, binary_classes)
    X_bin, y_bin = X[mask], y[mask]
    
    # Remap labels to 0/1
    le_bin = LabelEncoder()
    y_bin = le_bin.fit_transform(y_bin)
    
    print(f"Binary dataset: {len(X_bin)} samples")
    print(f"Classes: {dict(zip(le_bin.classes_, np.bincount(y_bin)))}")
    
    models = get_models()
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        y_pred = cross_val_predict(model, X_bin, y_bin, cv=skf)
        
        acc = accuracy_score(y_bin, y_pred)
        f1 = f1_score(y_bin, y_pred, average="binary")
        cm = confusion_matrix(y_bin, y_pred)
        
        results[name] = {
            "accuracy": acc,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        }
        
        print(f"  {name}: acc={acc:.4f}, f1={f1:.4f}")
    
    return results


def experiment_4_class_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    n_folds: int = 5,
) -> Dict[str, dict]:
    """
    Experiment 4: Class imbalance mitigation strategies.
    Tests: class weights, SMOTE, majority undersampling.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Class Imbalance Mitigation")
    print("="*60)
    
    results = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Strategy 1: Class weights
    print("\n--- Strategy 1: Class Weights ---")
    weighted_models = get_weighted_models()
    for name, model in weighted_models.items():
        y_pred = cross_val_predict(model, X, y, cv=skf)
        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average="macro")
        f1_weighted = f1_score(y, y_pred, average="weighted")
        cm = confusion_matrix(y, y_pred)
        
        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm.tolist(),
            "strategy": "class_weights",
        }
        print(f"  {name}: acc={acc:.4f}, f1_macro={f1_macro:.4f}")
    
    # Strategy 2: SMOTE
    if HAS_IMB:
        print("\n--- Strategy 2: SMOTE ---")
        smote_models = get_smote_models()
        for name, model in smote_models.items():
            try:
                y_pred = cross_val_predict(model, X, y, cv=skf)
                acc = accuracy_score(y, y_pred)
                f1_macro = f1_score(y, y_pred, average="macro")
                f1_weighted = f1_score(y, y_pred, average="weighted")
                cm = confusion_matrix(y, y_pred)
                
                results[name] = {
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "confusion_matrix": cm.tolist(),
                    "strategy": "SMOTE",
                }
                print(f"  {name}: acc={acc:.4f}, f1_macro={f1_macro:.4f}")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")
    
    # Strategy 3: Majority undersampling
    if HAS_IMB:
        print("\n--- Strategy 3: Majority Undersampling ---")
        subsample_models = get_subsampled_models()
        for name, model in subsample_models.items():
            try:
                y_pred = cross_val_predict(model, X, y, cv=skf)
                acc = accuracy_score(y, y_pred)
                f1_macro = f1_score(y, y_pred, average="macro")
                f1_weighted = f1_score(y, y_pred, average="weighted")
                cm = confusion_matrix(y, y_pred)
                
                results[name] = {
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "confusion_matrix": cm.tolist(),
                    "strategy": "undersampling",
                }
                print(f"  {name}: acc={acc:.4f}, f1_macro={f1_macro:.4f}")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")
    
    return results


def run_all_experiments(
    features_csv: str,
    output_dir: str = "results",
):
    """Run all experiments and save results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df, X, y, le = load_features(features_csv)
    
    # Save label mapping
    with open(Path(output_dir) / "label_mapping.json", "w") as f:
        json.dump({i: c for i, c in enumerate(le.classes_)}, f, indent=2)
    
    # NaN imputation for sklearn-based experiments (CatBoost handles NaN natively)
    X_clean = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    
    all_results = {}
    
    # Experiment 1
    exp1 = experiment_1_within_source(X_clean, y, le)
    all_results["experiment_1_within_source"] = exp1
    with open(Path(output_dir) / "exp1_within_source.json", "w") as f:
        json.dump(exp1, f, indent=2, default=str)
    
    # Experiment 2
    exp2 = experiment_2_cross_source(df, X_clean, y, le)
    all_results["experiment_2_cross_source"] = exp2
    with open(Path(output_dir) / "exp2_cross_source.json", "w") as f:
        json.dump(exp2, f, indent=2, default=str)
    
    # Experiment 3
    exp3 = experiment_3_binary_ablation(X_clean, y, le)
    all_results["experiment_3_binary_ablation"] = exp3
    with open(Path(output_dir) / "exp3_binary_ablation.json", "w") as f:
        json.dump(exp3, f, indent=2, default=str)
    
    # Experiment 4
    exp4 = experiment_4_class_imbalance(X_clean, y, le)
    all_results["experiment_4_class_imbalance"] = exp4
    with open(Path(output_dir) / "exp4_class_imbalance.json", "w") as f:
        json.dump(exp4, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to {output_dir}/")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    import sys
    
    features_csv = sys.argv[1] if len(sys.argv) > 1 else "features.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    
    run_all_experiments(features_csv, output_dir)
