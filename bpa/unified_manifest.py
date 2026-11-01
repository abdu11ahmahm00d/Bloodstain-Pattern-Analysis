"""
unified_manifest.py — Build a unified manifest combining all BPA datasets.

Datasets:
  - attinger/        → impact spatter (30 images, HP_19–HP_63)
  - attinger2/       → low-velocity (31 images, HP_0–HP_18 + C1–C12)
  - ferrarini/       → gunshot backspatter (16 images)
  - datasets_new/attinger_impact/  → 49 impact + 12 sliding-cylinder low-velocity
  - datasets_new/attinger_gunshot/ → 67 gunshot
  - kannan_drip/     → 457 low-velocity drip patterns

Creates a single CSV with all images, their labels, and metadata.
"""

import os
import json
import re
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict


def parse_impact_txt(txt_path: str) -> dict:
    """Parse attinger (impact spatter) metadata .txt file."""
    meta = {}
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Extract key fields
    patterns = {
        'dpi': r'Image Scale:\s*(\d+)\s*dpi',
        'pixels_per_cm': r'(\d+\.?\d*)\s*pixel/cm',
        'room_temp': r'Room Temp\s*=\s*([\d.]+)',
        'room_humidity': r'Room Humidity\s*=\s*([\d.]+)',
        'hematocrit': r'Hematocrit\s*=\s*(\d+)',
        'blood_volume_ml': r'Blood Volume\s*=\s*([\d.]+)',
        'blood_supply': r'Blood supply\s*=\s*(\w+)',
        'dowel_height_station': r'Dowel height station:\s*(\d+)',
        'dowel_angle_deg': r'Dowel Angle:\s*(\d+)',
        'bungee_count': r'# of Rubber Bungees:\s*(\d+)',
        'moment_arm_cm': r'Moment arm:\s*([\d.]+)',
        'impact_velocity_mps': r'Impact Velocity\s*=\s*([\d.]+)',
        'target_distance_cm': r'Target distance\s*=\s*([\d.]+)',
        'x_origin_cm': r'x_origin\s*=\s*([\d.-]+)',
        'y_origin_cm': r'y_origin\s*=\s*([\d.-]+)',
        'z_origin_cm': r'z_origin\s*=\s*([\d.-]+)',
        'x_target_cm': r'x_target\s*=\s*([\d.-]+)',
        'y_target_cm': r'y_target\s*=\s*([\d.-]+)',
        'z_target_cm': r'z_target\s*=\s*([\d.-]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            val = match.group(1)
            try:
                meta[key] = float(val)
            except ValueError:
                meta[key] = val
    
    return meta


def parse_low_velocity_txt(txt_path: str) -> dict:
    """Parse attinger2 (low-velocity) metadata .txt file."""
    meta = {}
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    patterns = {
        'dpi': r'Image Scale:\s*(\d+)\s*dpi',
        'pixels_per_cm': r'(\d+\.?\d*)\s*pixel/cm',
        'cylinder_height_cm': r'Cylinder Height\s*=\s*([\d.]+)',
        'sliding_angle_deg': r'Sliding Angle\s*=\s*(\d+)',
        'drop_volume_ul': r'Drop Volume\s*=\s*([\d.]+)',
        'x_origin_cm': r'x_origin\s*=\s*([\d.-]+)',
        'y_origin_cm': r'y_origin\s*=\s*([\d.-]+)',
        'z_origin_cm': r'z_origin\s*=\s*([\d.-]+)',
        'x_target_cm': r'x_target\s*=\s*([\d.-]+)',
        'y_target_cm': r'y_target\s*=\s*([\d.-]+)',
        'z_target_cm': r'z_target\s*=\s*([\d.-]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            val = match.group(1)
            try:
                meta[key] = float(val)
            except ValueError:
                meta[key] = val
    
    return meta


def parse_gunshot_txt(txt_path: str) -> dict:
    """Parse ferrarini (gunshot backspatter) metadata .txt file."""
    meta = {}
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    patterns = {
        'dpi': r'Image Scale:\s*(\d+)\s*dpi',
        'pixels_per_cm': r'(\d+\.?\d*)\s*pixel/cm',
        'target_distance_cm': r'Target distance\s*=\s*([\d.]+)',
        'caliber': r'Caliber:\s*(\S+)',
        'ammunition': r'Ammunition:\s*(.+)',
        'blood_type': r'Blood Type:\s*(.+)',
        'hematocrit': r'Hematocrit\s*=\s*(\d+)',
        'x_origin_cm': r'x_origin\s*=\s*([\d.-]+)',
        'y_origin_cm': r'y_origin\s*=\s*([\d.-]+)',
        'z_origin_cm': r'z_origin\s*=\s*([\d.-]+)',
        'x_target_cm': r'x_target\s*=\s*([\d.-]+)',
        'y_target_cm': r'y_target\s*=\s*([\d.-]+)',
        'z_target_cm': r'z_target\s*=\s*([\d.-]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            val = match.group(1).strip()
            try:
                meta[key] = float(val)
            except ValueError:
                meta[key] = val
    
    return meta


def build_attinger_manifest(base_dir: str) -> List[Dict]:
    """Build manifest entries for all Attinger-lab datasets (attinger, attinger2, ferrarini, datasets_new)."""
    entries = []
    base = Path(base_dir)
    
    def add_entry(img_path, mechanism_class, source_dataset, meta, sample_id, extra=None):
        entry = {
            "image_path": str(img_path),
            "mechanism_class": mechanism_class,
            "source_dataset": source_dataset,
            "blood_type": "swine",
            "substrate": "poster_board",
            "dpi": meta.get("dpi", 600.0),
            "pixels_per_cm": meta.get("pixels_per_cm", 236.2),
            "sample_id": sample_id,
        }
        if extra:
            entry.update(extra)
        entries.append(entry)
    
    # --- 1. attinger/ — Impact spatter (HP_XX), original 30 images ---
    attinger_dir = base / "attinger"
    if attinger_dir.exists():
        for subdir in sorted(attinger_dir.iterdir()):
            if not subdir.is_dir():
                continue
            img_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            txt_files = list(subdir.glob("*.txt"))
            if not img_files:
                continue
            meta = parse_impact_txt(str(txt_files[0])) if txt_files else {}
            add_entry(img_files[0], "impact", "attinger_impact", meta, subdir.name, {
                "dowel_angle_deg": meta.get("dowel_angle_deg"),
                "dowel_height_station": meta.get("dowel_height_station"),
                "moment_arm_cm": meta.get("moment_arm_cm"),
            })
    
    # --- 2. attinger2/ — Low-velocity (HP_0–HP_18 + C1–C12), original 31 images ---
    attinger2_dir = base / "attinger2"
    if attinger2_dir.exists():
        for subdir in sorted(attinger2_dir.iterdir()):
            if not subdir.is_dir():
                continue
            img_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            txt_files = list(subdir.glob("*.txt"))
            if not img_files:
                continue
            meta = parse_low_velocity_txt(str(txt_files[0])) if txt_files else {}
            add_entry(img_files[0], "low_velocity", "attinger_lowvelocity", meta, subdir.name, {
                "cylinder_height_cm": meta.get("cylinder_height_cm"),
                "sliding_angle_deg": meta.get("sliding_angle_deg"),
            })
    
    # --- 3. ferrarini/ — Gunshot backspatter, original 16 images ---
    ferrarini_dir = base / "ferrarini"
    if ferrarini_dir.exists():
        for subdir in sorted(ferrarini_dir.rglob("*")):
            if not subdir.is_dir():
                continue
            img_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            txt_files = list(subdir.glob("*.txt"))
            if not img_files:
                continue
            meta = parse_gunshot_txt(str(txt_files[0])) if txt_files else {}
            weapon_type = "unknown"
            if "experiments_with_g" in str(subdir):
                weapon_type = "gun"
            elif "experiments_with_r" in str(subdir):
                weapon_type = "rifle"
            add_entry(img_files[0], "gunshot", "ferrarini", meta, subdir.name, {
                "weapon_type": weapon_type,
                "target_distance_cm": meta.get("target_distance_cm"),
                "caliber": meta.get("caliber"),
            })
    
    # --- 4. datasets_new/attinger_impact/ — 49 impact (HP series) + 12 sliding cylinder (C series) ---
    new_impact_dir = base / "datasets_new" / "attinger_impact"
    if new_impact_dir.exists():
        for subdir in sorted(new_impact_dir.iterdir()):
            if not subdir.is_dir():
                continue
            img_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            txt_files = list(subdir.glob("*.txt"))
            if not img_files:
                continue
            meta = parse_impact_txt(str(txt_files[0])) if txt_files else {}
            sample_name = subdir.name
            if sample_name.startswith("C"):
                add_entry(img_files[0], "low_velocity", "attinger_lowvelocity", meta, sample_name, {
                    "cylinder_height_cm": meta.get("cylinder_height_cm"),
                    "sliding_angle_deg": meta.get("sliding_angle_deg"),
                })
            else:
                add_entry(img_files[0], "impact", "attinger_impact", meta, sample_name, {
                    "dowel_angle_deg": meta.get("dowel_angle_deg"),
                    "dowel_height_station": meta.get("dowel_height_station"),
                    "moment_arm_cm": meta.get("moment_arm_cm"),
                })
    
    # --- 5. datasets_new/attinger_gunshot/ — 67 gunshot images ---
    new_gunshot_dir = base / "datasets_new" / "attinger_gunshot"
    if new_gunshot_dir.exists():
        for subdir in sorted(new_gunshot_dir.iterdir()):
            if not subdir.is_dir():
                continue
            img_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            txt_files = list(subdir.glob("*.txt"))
            if not img_files:
                continue
            meta = parse_gunshot_txt(str(txt_files[0])) if txt_files else {}
            add_entry(img_files[0], "gunshot", "attinger_gunshot", meta, subdir.name, {
                "target_distance_cm": meta.get("target_distance_cm"),
            })
    
    return entries


def build_kannan_manifest(kannan_dir: str) -> List[Dict]:
    """
    Build manifest entries for Kannan et al. drip patterns dataset.
    
    Expected structure: kannan_dir/substrate_height/sample_id/img.jpg
    E.g.: kannan_drip/grass25cm_essential/1.1/IMG_9754.JPG
          kannan_drip/paper25cm/1-1/1-1edit.jpg
    """
    entries = []
    base = Path(kannan_dir)
    
    if not base.exists():
        print(f"WARNING: Kannan directory not found: {kannan_dir}")
        return entries
    
    # Parse folder names: substrate + height
    substrate_map = {
        "paper": "paper",
        "grass": "grass",
        "tile": "ceramic_tile",
        "snow": "snow",
        "ceramic": "ceramic_tile",
    }
    
    height_pattern = re.compile(r'(\d+)cm')
    
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Parse substrate and height from folder name
        folder_name = subdir.name.lower()
        substrate = "unknown"
        height_cm = None
        
        for key, val in substrate_map.items():
            if key in folder_name:
                substrate = val
                break
        
        height_match = height_pattern.search(folder_name)
        if height_match:
            height_cm = float(height_match.group(1))
        
        # Find all images in this folder (recursive for nested sample dirs)
        for img_path in sorted(subdir.rglob("*")):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
                continue
            entries.append({
                "image_path": str(img_path),
                "mechanism_class": "low_velocity",
                "source_dataset": "kannan",
                "blood_type": "bovine",
                "substrate": substrate,
                "dpi": 6000,  # Kannan dataset is 6000 dpi
                "pixels_per_cm": 6000 / 2.54,  # ~2362.2 px/cm
                "sample_id": img_path.parent.name,
                "drip_height_cm": height_cm,
            })
    
    return entries
def build_unified_manifest(
    base_dir: str = ".",
    kannan_dir: str = None,
    output_path: str = "unified_manifest.csv",
) -> pd.DataFrame:
    """
    Build the complete unified manifest combining all datasets.
    """
    all_entries = []
    
    # Attinger-lab datasets (attinger, attinger2, ferrarini, datasets_new)
    print("Building Attinger manifest...")
    attinger_entries = build_attinger_manifest(base_dir)
    all_entries.extend(attinger_entries)
    for ds in sorted(set(e['source_dataset'] for e in attinger_entries)):
        print(f"  {ds}: {sum(1 for e in attinger_entries if e['source_dataset'] == ds)} images")
    
    # Kannan dataset
    if kannan_dir:
        print("Building Kannan manifest...")
        kannan_entries = build_kannan_manifest(kannan_dir)
        all_entries.extend(kannan_entries)
        print(f"  kannan: {len(kannan_entries)} images")
    
    df = pd.DataFrame(all_entries)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nUnified manifest: {len(df)} images -> {output_path}")
    print(f"\nClass distribution:")
    print(df['mechanism_class'].value_counts().to_string())
    print(f"\nSource distribution:")
    print(df['source_dataset'].value_counts().to_string())
    
    return df


if __name__ == "__main__":
    import sys
    
    # Auto-detect paths based on OS
    if sys.platform == "win32":
        base_dir = "C:\\Users\\YourName\\Documents\\BPA_Research\\BPA2"
        kannan_dir = "C:\\Users\\YourName\\Documents\\BPA_Research\\BPA2\\kannan_drip"
        output = "unified_manifest.csv"
    else:
        base_dir = "/mnt/c/Users/YourName/Documents/BPA_Research/BPA2"
        kannan_dir = "/mnt/c/Users/YourName/Documents/BPA_Research/BPA2/kannan_drip"
        output = "unified_manifest.csv"
    
    df = build_unified_manifest(base_dir, kannan_dir, output)
