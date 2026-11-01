"""
save_final_model.py — Train and save the best BPA classifier.

Champion: CatBoost + SqrtBalanced + l2=5
CV results computed live — no hardcoded values.
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report
from catboost import CatBoostClassifier

from .train_classifier import load_features, FEATURE_COLS


def train_and_save():
    """Train final model on all data and save."""
    print("Loading features...")
    df, X, y, le = load_features("features.csv")
    
    print(f"\nTraining final CatBoost model...")
    print(f"  iterations=400, depth=4, lr=0.05, l2_leaf_reg=5")
    print(f"  auto_class_weights='SqrtBalanced'")
    
    # Build pipeline — CatBoost handles NaN natively, no imputation needed
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CatBoostClassifier(
            iterations=400,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=5,
            auto_class_weights="SqrtBalanced",
            random_state=42,
            verbose=50,
        ))
    ])
    
    # Compute CV results on full dataset
    print("\nComputing 5-fold stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(pipe, X, y, cv=skf)
    cv_accuracy = accuracy_score(y, y_pred_cv)
    cv_f1_macro = f1_score(y, y_pred_cv, average="macro", zero_division=0)
    cv_f1_weighted = f1_score(y, y_pred_cv, average="weighted", zero_division=0)
    cv_report = classification_report(y, y_pred_cv, target_names=le.classes_, zero_division=0, output_dict=True)
    
    # Per-class F1
    per_class_f1 = {}
    for cls_name in le.classes_:
        cls_idx = list(le.classes_).index(cls_name)
        per_class_f1[cls_name] = round(cv_report[cls_name]["f1-score"], 4)
    
    # Train on full dataset
    print("\nRetraining on full dataset...")
    pipe.fit(X, y)
    
    # Save model
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "bpa_classifier_final.joblib"
    joblib.dump(pipe, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metadata with computed CV results
    metadata = {
        "model_type": "CatBoost",
        "hyperparameters": {
            "iterations": 400,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5,
            "auto_class_weights": "SqrtBalanced",
        },
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "classes": le.classes_.tolist(),
        "class_mapping": {i: c for i, c in enumerate(le.classes_)},
        "train_samples": len(X),
        "class_distribution": dict(zip(le.classes_, np.bincount(y).tolist())),
        "cv_results": {
            "accuracy": round(float(cv_accuracy), 4),
            "f1_macro": round(float(cv_f1_macro), 4),
            "f1_weighted": round(float(cv_f1_weighted), 4),
            "per_class_f1": per_class_f1,
        },
    }
    
    meta_path = output_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")
    
    # Feature importance
    clf = pipe.named_steps["clf"]
    importances = clf.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Feature Importances:")
    for name, imp in feat_imp[:10]:
        print(f"  {name:30s}: {imp:.2f}")
    
    # Save feature importances
    imp_path = output_dir / "feature_importances.json"
    with open(imp_path, "w") as f:
        json.dump([{"feature": n, "importance": float(i)} for n, i in feat_imp], f, indent=2)
    print(f"\nFeature importances saved to: {imp_path}")
    
    return pipe, le


if __name__ == "__main__":
    train_and_save()
