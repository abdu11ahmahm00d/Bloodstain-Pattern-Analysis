"""
tune_model.py — Hyperparameter tuning + advanced imbalance strategies for BPA classification.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMB = True
except ImportError:
    HAS_IMB = False

from .train_classifier import FEATURE_COLS, load_features


def try_smote_variants(X, y, n_folds=5):
    """Test different SMOTE variants with XGBoost."""
    print("\n" + "="*60)
    print("SMOTE VARIANTS COMPARISON (XGBoost)")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    xgb_params = dict(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42,
    )
    
    variants = {
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=42),
        "SVMSMOTE": SVMSMOTE(random_state=42),
        "SMOTETomek": SMOTETomek(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42),
    }
    
    results = {}
    for name, sampler in variants.items():
        try:
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("sampler", sampler),
                ("clf", xgb.XGBClassifier(**xgb_params)),
            ])
            y_pred = cross_val_predict(pipe, X, y, cv=skf)
            acc = accuracy_score(y, y_pred)
            f1_macro = f1_score(y, y_pred, average="macro")
            f1_weighted = f1_score(y, y_pred, average="weighted")
            cm = confusion_matrix(y, y_pred)
            
            results[name] = {
                "accuracy": acc, "f1_macro": f1_macro,
                "f1_weighted": f1_weighted, "confusion_matrix": cm.tolist(),
            }
            print(f"  {name:20s}: acc={acc:.4f}, f1_macro={f1_macro:.4f}")
        except Exception as e:
            print(f"  {name:20s}: FAILED - {e}")
    
    return results


def try_different_models(X, y, n_folds=5):
    """Test different model architectures with SMOTE."""
    print("\n" + "="*60)
    print("MODEL COMPARISON (with SMOTE)")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42)
    
    models = {}
    
    # XGBoost variants
    for depth, lr, n_est in [(4, 0.05, 300), (5, 0.1, 200), (6, 0.1, 300), (3, 0.1, 200)]:
        name = f"XGB_d{depth}_lr{lr}_n{n_est}"
        models[name] = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42,
        )
    
    # LightGBM
    if HAS_LGB:
        models["LGB_d5_n200"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        models["LGB_d6_n300"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
    
    # RandomForest with more trees
    models["RF_n500_d15"] = RandomForestClassifier(
        n_estimators=500, max_depth=15, random_state=42,
    )
    models["RF_n300_d10_balanced"] = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42,
    )
    
    results = {}
    for name, clf in models.items():
        try:
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", smote),
                ("clf", clf),
            ])
            y_pred = cross_val_predict(pipe, X, y, cv=skf)
            acc = accuracy_score(y, y_pred)
            f1_macro = f1_score(y, y_pred, average="macro")
            f1_weighted = f1_score(y, y_pred, average="weighted")
            cm = confusion_matrix(y, y_pred)
            
            results[name] = {
                "accuracy": acc, "f1_macro": f1_macro,
                "f1_weighted": f1_weighted, "confusion_matrix": cm.tolist(),
            }
            print(f"  {name:30s}: acc={acc:.4f}, f1_macro={f1_macro:.4f}")
        except Exception as e:
            print(f"  {name:30s}: FAILED - {e}")
    
    return results


def try_ensemble(X, y, n_folds=5):
    """Test ensemble methods."""
    print("\n" + "="*60)
    print("ENSEMBLE METHODS")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42)
    scaler = StandardScaler()
    
    results = {}
    
    # 1. Soft voting ensemble
    print("\n--- Soft Voting ---")
    estimators = [
        ("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42,
        )),
    ]
    if HAS_LGB:
        estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )))
    
    voting = VotingClassifier(estimators=estimators, voting="soft")
    pipe = ImbPipeline([
        ("scaler", scaler), ("smote", smote), ("clf", voting),
    ])
    y_pred = cross_val_predict(pipe, X, y, cv=skf)
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred)
    results["Voting_soft"] = {
        "accuracy": acc, "f1_macro": f1_macro, "confusion_matrix": cm.tolist(),
    }
    print(f"  Soft Voting: acc={acc:.4f}, f1_macro={f1_macro:.4f}")
    
    # 2. Stacking ensemble
    print("\n--- Stacking ---")
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
    )
    pipe = ImbPipeline([
        ("scaler", scaler), ("smote", smote), ("clf", stack),
    ])
    y_pred = cross_val_predict(pipe, X, y, cv=skf)
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred)
    results["Stacking"] = {
        "accuracy": acc, "f1_macro": f1_macro, "confusion_matrix": cm.tolist(),
    }
    print(f"  Stacking: acc={acc:.4f}, f1_macro={f1_macro:.4f}")
    
    return results


def main():
    features_csv = "features.csv"
    output_dir = "results/tuning"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df, X, y, le = load_features(features_csv)
    
    all_results = {}
    
    # 1. SMOTE variants
    smote_results = try_smote_variants(X, y)
    all_results["smote_variants"] = smote_results
    
    # 2. Model architectures
    model_results = try_different_models(X, y)
    all_results["model_comparison"] = model_results
    
    # 3. Ensembles
    ensemble_results = try_ensemble(X, y)
    all_results["ensembles"] = ensemble_results
    
    # Find best overall
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS")
    print("="*60)
    
    all_flat = {}
    for category, res_dict in all_results.items():
        for name, metrics in res_dict.items():
            all_flat[f"{category}/{name}"] = metrics
    
    ranked = sorted(all_flat.items(), key=lambda x: x[1]["f1_macro"], reverse=True)
    print("\nTop 5 by F1 (macro):")
    for i, (name, metrics) in enumerate(ranked[:5]):
        print(f"  {i+1}. {name:45s} acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f}")
    
    # Save
    with open(Path(output_dir) / "tuning_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/")
    return all_results


if __name__ == "__main__":
    main()
