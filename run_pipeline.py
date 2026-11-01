"""
run_pipeline.py — Main orchestrator for BPA multi-class classification pipeline.

Usage:
  python run_pipeline.py --step all          # Run full pipeline
  python run_pipeline.py --step manifest     # Build manifest only
  python run_pipeline.py --step features     # Extract features only
  python run_pipeline.py --step train        # Train classifiers only
  python run_pipeline.py --step shap         # Run SHAP analysis only
  python run_pipeline.py --step test         # Quick test on 5 images
"""

import argparse
import sys
from pathlib import Path


# Paths: datasets live under BPA2, code and outputs under BPA
BPA2_DIR = Path("C:/Users/YourName/Documents/BPA_Research/BPA2")
BPA_DIR = Path(__file__).parent
KANNAN_DIR = BPA2_DIR / "kannan_drip"
MANIFEST_PATH = BPA_DIR / "data" / "unified_manifest.csv"
FEATURES_PATH = BPA_DIR / "data" / "features.csv"
RESULTS_DIR = BPA_DIR / "results"
SHAP_DIR = BPA_DIR / "shap_results"
DEBUG_DIR = BPA_DIR / "debug"


def step_manifest():
    """Step 1: Build unified manifest."""
    from bpa.unified_manifest import build_unified_manifest
    
    print("="*60)
    print("STEP 1: Building Unified Manifest")
    print("="*60)
    
    df = build_unified_manifest(
        base_dir=str(BPA2_DIR),
        kannan_dir=str(KANNAN_DIR),
        output_path=str(MANIFEST_PATH),
    )
    return df


def step_features():
    """Step 2: Extract features from all images."""
    import pandas as pd
    from bpa.feature_extraction import extract_features_from_image
    
    print("\n" + "="*60)
    print("STEP 2: Feature Extraction")
    print("="*60)
    
    if not MANIFEST_PATH.exists():
        print("ERROR: Manifest not found. Run --step manifest first.")
        sys.exit(1)
    
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Processing {len(manifest)} images...")
    
    records = []
    failed = []
    
    for i, row in manifest.iterrows():
        img_path = row["image_path"]
        dpi = row.get("dpi", 600.0)
        substrate = row.get("substrate", "unknown")
        mechanism = row.get("mechanism_class", "unknown")
        source = row.get("source_dataset", "unknown")
        
        print(f"  [{i+1}/{len(manifest)}] {Path(img_path).name}...", end=" ")
        
        try:
            feat = extract_features_from_image(
                img_path, dpi=dpi,
                substrate=substrate,
                mechanism_class=mechanism,
                source_dataset=source,
                debug_dir=str(DEBUG_DIR) if i < 10 else None,  # debug first 10
            )
            records.append(feat)
            print(f"OK ({feat.get('stain_count', 0)} stains)")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((img_path, str(e)))
    
    df = pd.DataFrame(records)
    df.to_csv(FEATURES_PATH, index=False)
    
    print(f"\nFeatures extracted: {len(df)} images")
    print(f"Failed: {len(failed)} images")
    print(f"Saved to: {FEATURES_PATH}")
    
    if failed:
        with open(BPA2_DIR / "failed_extractions.txt", "w") as f:
            for path, err in failed:
                f.write(f"{path}\t{err}\n")
    
    return df


def step_train():
    """Step 3: Run all classification experiments."""
    from bpa.train_classifier import run_all_experiments
    
    print("\n" + "="*60)
    print("STEP 3: Classification Experiments")
    print("="*60)
    
    if not FEATURES_PATH.exists():
        print("ERROR: Features not found. Run --step features first.")
        sys.exit(1)
    
    results = run_all_experiments(
        features_csv=str(FEATURES_PATH),
        output_dir=str(RESULTS_DIR),
    )
    return results


def step_shap():
    """Step 4: SHAP analysis."""
    from bpa.shap_analysis import run_shap_analysis
    
    print("\n" + "="*60)
    print("STEP 4: SHAP Analysis")
    print("="*60)
    
    if not FEATURES_PATH.exists():
        print("ERROR: Features not found. Run --step features first.")
        sys.exit(1)
    
    run_shap_analysis(
        features_csv=str(FEATURES_PATH),
        output_dir=str(SHAP_DIR),
    )


def step_test():
    """Quick test: process 5 images to verify pipeline works."""
    from bpa.feature_extraction import extract_features_from_image
    
    print("="*60)
    print("QUICK TEST: Processing 5 sample images")
    print("="*60)
    
    # Find 5 sample images from local datasets
    import glob
    samples = []
    
    impact_imgs = glob.glob(f"{BPA2_DIR}/attinger/*/*.jpg")
    if impact_imgs:
        samples.append(("attinger/impact", impact_imgs[0], 600.0))
    
    lowvel_imgs = glob.glob(f"{BPA2_DIR}/attinger2/*/*.jpg")
    if lowvel_imgs:
        samples.append(("attinger2/low_velocity", lowvel_imgs[0], 600.0))
    
    gunshot_imgs = glob.glob(f"{BPA2_DIR}/ferrarini/**/*.jpg", recursive=True)
    if gunshot_imgs:
        samples.append(("ferrarini/gunshot", gunshot_imgs[0], 600.0))
    
    if not samples:
        print("ERROR: No sample images found!")
        sys.exit(1)
    
    for label, img_path, dpi in samples:
        print(f"\n--- {label}: {Path(img_path).name} ---")
        try:
            feat = extract_features_from_image(
                img_path, dpi=dpi,
                mechanism_class=label.split("/")[1],
                debug_dir=str(DEBUG_DIR),
            )
            print(f"  Stains detected: {feat.get('stain_count', 0)}")
            print(f"  Total area: {feat.get('total_stained_area_mm2', 0):.1f} mm²")
            print(f"  Axis ratio (mean): {feat.get('axis_ratio_mean', 0):.2f}")
            print(f"  Eccentricity (mean): {feat.get('eccentricity_mean', 0):.3f}")
            print(f"  Circularity (mean): {feat.get('circularity_mean', 0):.3f}")
            print(f"  Spatial spread: {feat.get('spatial_spread_total', 0):.1f}")
            print(f"  Debug images saved to {DEBUG_DIR}/")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTest complete. Check {DEBUG_DIR}/ for debug images.")


def main():
    parser = argparse.ArgumentParser(description="BPA pipeline")
    parser.add_argument("--step", required=True,
                       choices=["all", "manifest", "features", "train", "shap", "test"],
                       help="Which step to run")
    args = parser.parse_args()
    
    if args.step == "test":
        step_test()
    elif args.step == "manifest":
        step_manifest()
    elif args.step == "features":
        step_features()
    elif args.step == "train":
        step_train()
    elif args.step == "shap":
        step_shap()
    elif args.step == "all":
        step_manifest()
        step_features()
        step_train()
        step_shap()
        print("\n" + "="*60)
        print("FULL PIPELINE COMPLETE")
        print("="*60)


if __name__ == "__main__":
    main()
