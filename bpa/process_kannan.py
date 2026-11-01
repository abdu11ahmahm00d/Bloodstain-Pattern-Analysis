"""
process_kannan.py — Batch feature extraction for Kannan drip dataset.

Reads unified_manifest.csv, extracts features for Kannan images,
and appends them to features.csv.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from .feature_extraction import extract_features_from_image


def process_kannan_images(
    manifest_path: str,
    features_path: str,
    debug_dir: str = None,
):
    """
    Process Kannan images from manifest and append to features.csv.
    """
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    kannan = manifest[manifest["source_dataset"] == "kannan"].copy()
    print(f"Found {len(kannan)} Kannan images in manifest")
    
    # Load existing features
    existing = pd.read_csv(features_path)
    print(f"Existing features: {len(existing)} rows")
    
    # Check which Kannan images already have features
    existing_paths = set(existing["image_path"].str.replace("\\", "/").str.lower())
    kannan["norm_path"] = kannan["image_path"].str.replace("\\", "/").str.lower()
    new_images = kannan[~kannan["norm_path"].isin(existing_paths)]
    print(f"New images to process: {len(new_images)}")
    
    if len(new_images) == 0:
        print("Nothing to process!")
        return existing
    
    # Process each image
    records = []
    for i, (_, row) in enumerate(new_images.iterrows()):
        img_path = row["image_path"]
        print(f"  [{i+1}/{len(new_images)}] {Path(img_path).name}...", end=" ", flush=True)
        
        try:
            feat = extract_features_from_image(
                img_path,
                dpi=row.get("dpi", 6000.0),
                substrate=row.get("substrate", "unknown"),
                mechanism_class=row["mechanism_class"],
                source_dataset=row["source_dataset"],
                debug_dir=debug_dir,
            )
            records.append(feat)
            print(f"OK ({feat.get('stain_count', 0)} stains)")
        except Exception as e:
            print(f"FAILED: {e}")
    
    # Create DataFrame and append
    new_df = pd.DataFrame(records)
    combined = pd.concat([existing, new_df], ignore_index=True)
    
    # Save
    combined.to_csv(features_path, index=False)
    print(f"\nSaved: {len(combined)} total rows -> {features_path}")
    print(f"  Previously: {len(existing)}")
    print(f"  Added: {len(new_df)}")
    
    # Class distribution
    print(f"\nClass distribution:")
    print(combined["mechanism_class"].value_counts().to_string())
    
    return combined


if __name__ == "__main__":
    # Auto-detect paths based on OS
    if sys.platform == "win32":
        manifest = "C:\\Users\\YourName\\Documents\\BPA_Research\\BPA2\\unified_manifest.csv"
        features = "C:\\Users\\YourName\\Documents\\BPA_Research\\BPA2\\features.csv"
        debug = "C:\\Users\\YourName\\Documents\\BPA_Research\\BPA2\\debug"
    else:
        manifest = "/mnt/c/Users/YourName/Documents/BPA_Research/BPA2/unified_manifest.csv"
        features = "/mnt/c/Users/YourName/Documents/BPA_Research/BPA2/features.csv"
        debug = "/mnt/c/Users/YourName/Documents/BPA_Research/BPA2/debug"
    
    process_kannan_images(manifest, features, debug)
