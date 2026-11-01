"""
feature_extraction.py — Extract ~35 morphological and spatial features from bloodstain images.

Feature Groups:
  Group 1: Stain Morphology (per stain, aggregated as mean/std/max)
  Group 2: Pattern Distribution (per image)
  Group 3: Directionality (per image)
  Group 4: Context (substrate, DPI, etc.)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from .stain_detection import StainInfo, detect_stains


def circular_mean(angles_deg: np.ndarray) -> float:
    """Circular mean of angles in degrees."""
    angles_rad = np.deg2rad(angles_deg)
    sin_sum = np.mean(np.sin(angles_rad))
    cos_sum = np.mean(np.cos(angles_rad))
    return np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360


def circular_variance(angles_deg: np.ndarray) -> float:
    """Circular variance (0 = all same direction, 1 = uniform)."""
    angles_rad = np.deg2rad(angles_deg)
    R = np.sqrt(np.mean(np.sin(angles_rad))**2 + np.mean(np.cos(angles_rad))**2)
    return 1.0 - R


def estimate_convergence_point(stains: List[StainInfo]) -> tuple:
    """
    Estimate convergence point by extending major axis lines backward.
    Returns (cx, cy, dispersion) where dispersion is std of intersection points.
    """
    if len(stains) < 2:
        return (np.nan, np.nan, np.nan)
    
    intersections_x = []
    intersections_y = []
    
    for i in range(len(stains)):
        for j in range(i + 1, len(stains)):
            s1 = stains[i]
            s2 = stains[j]
            
            # Direction vectors from orientation
            angle1 = np.deg2rad(s1.orientation_deg)
            angle2 = np.deg2rad(s2.orientation_deg)
            
            # Line 1: point (x1,y1) with direction (dx1,dy1)
            x1, y1 = s1.centroid_x, s1.centroid_y
            dx1, dy1 = np.cos(angle1), np.sin(angle1)
            
            # Line 2: point (x2,y2) with direction (dx2,dy2)
            x2, y2 = s2.centroid_x, s2.centroid_y
            dx2, dy2 = np.cos(angle2), np.sin(angle2)
            
            # Solve for intersection: P1 + t1*D1 = P2 + t2*D2
            denom = dx1 * dy2 - dy1 * dx2
            if abs(denom) < 1e-10:
                continue  # parallel lines
            
            t1 = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
            
            # Intersection point (extending backward = negative t)
            ix = x1 + t1 * dx1
            iy = y1 + t1 * dy1
            
            intersections_x.append(ix)
            intersections_y.append(iy)
    
    if len(intersections_x) < 2:
        return (np.nan, np.nan, np.nan)
    
    ix_arr = np.array(intersections_x)
    iy_arr = np.array(intersections_y)
    
    # Filter outliers (points far from median)
    mx, my = np.median(ix_arr), np.median(iy_arr)
    dists = np.sqrt((ix_arr - mx)**2 + (iy_arr - my)**2)
    mask = dists < np.percentile(dists, 90)  # keep within 90th percentile
    
    if mask.sum() < 2:
        return (mx, my, np.nan)
    
    return (
        np.mean(ix_arr[mask]),
        np.mean(iy_arr[mask]),
        np.std(dists[mask])
    )


def compute_nearest_neighbor_distances(centroids: np.ndarray) -> tuple:
    """Compute mean and std of nearest-neighbor distances."""
    if len(centroids) < 2:
        return (np.nan, np.nan)
    
    from scipy.spatial.distance import cdist
    dists = cdist(centroids, centroids)
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    return (np.mean(nn_dists), np.std(nn_dists))


def extract_features(
    stains: List[StainInfo],
    metadata: dict,
    substrate: str = "unknown",
    mechanism_class: str = "unknown",
    source_dataset: str = "unknown",
) -> Dict[str, Any]:
    """
    Extract all features from detected stains for one image.
    
    Returns dict with ~35 features.
    """
    features = {
        # Identifiers
        "image_path": metadata["image_path"],
        "mechanism_class": mechanism_class,
        "source_dataset": source_dataset,
        "substrate": substrate,
        "dpi": metadata.get("dpi", 600.0),
    }
    
    if len(stains) == 0:
        # No stains detected — fill with zeros/nan
        for key in _get_feature_names():
            features[key] = np.nan
        features["stain_count"] = 0
        return features
    
    # --- Group 1: Stain Morphology (aggregated) ---
    axis_ratios = np.array([s.axis_ratio for s in stains])
    areas = np.array([s.area_mm2 for s in stains])
    solidities = np.array([s.solidity for s in stains])
    eccentricities = np.array([s.eccentricity for s in stains])
    orientations = np.array([s.orientation_deg for s in stains])
    circularities = np.array([s.circularity for s in stains])
    
    for name, vals in [
        ("axis_ratio", axis_ratios),
        ("area_mm2", areas),
        ("solidity", solidities),
        ("eccentricity", eccentricities),
        ("circularity", circularities),
    ]:
        features[f"{name}_mean"] = np.mean(vals)
        features[f"{name}_std"] = np.std(vals) if len(vals) > 1 else 0.0
        features[f"{name}_max"] = np.max(vals)
    
    # --- Group 2: Pattern Distribution ---
    centroids = np.array([[s.centroid_x, s.centroid_y] for s in stains])
    
    features["stain_count"] = len(stains)
    features["total_stained_area_mm2"] = np.sum(areas)
    
    # Stain density (stains per cm²)
    img_area_cm2 = (metadata["image_width"] * metadata["image_height"]) / (metadata.get("pixels_per_cm", 236.2) ** 2)
    features["stain_density_per_cm2"] = len(stains) / img_area_cm2 if img_area_cm2 > 0 else 0
    
    # Spatial spread
    features["spatial_spread_x_std"] = np.std(centroids[:, 0])
    features["spatial_spread_y_std"] = np.std(centroids[:, 1])
    features["spatial_spread_total"] = np.sqrt(np.var(centroids[:, 0]) + np.var(centroids[:, 1]))
    
    # Nearest-neighbor distances (in mm)
    pixels_per_cm = metadata.get("pixels_per_cm", 236.2)
    nn_mean, nn_std = compute_nearest_neighbor_distances(centroids)
    features["nn_dist_mean_mm"] = nn_mean / (pixels_per_cm / 10) if not np.isnan(nn_mean) else np.nan
    features["nn_dist_std_mm"] = nn_std / (pixels_per_cm / 10) if not np.isnan(nn_std) else np.nan
    
    # Convex hull area
    if len(centroids) >= 3:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(centroids)
            features["convex_hull_area_mm2"] = hull.volume / (pixels_per_cm / 10) ** 2  # volume = area in 2D
        except Exception:
            features["convex_hull_area_mm2"] = np.nan
    else:
        features["convex_hull_area_mm2"] = np.nan
    
    # --- Group 3: Directionality ---
    features["mean_orientation_deg"] = circular_mean(orientations)
    features["circular_variance"] = circular_variance(orientations)
    
    cx, cy, disp = estimate_convergence_point(stains)
    features["convergence_x"] = cx
    features["convergence_y"] = cy
    features["convergence_dispersion"] = disp / (pixels_per_cm / 10) if not np.isnan(disp) else np.nan
    features["has_convergence"] = 1.0 if len(stains) >= 2 else 0.0  # explicit flag for NaN rows
    
    # --- Group 4: Context ---
    # image_width / image_height excluded — camera resolution confounds classification
    
    return features


def _get_feature_names() -> List[str]:
    """Return list of all feature column names (excluding identifiers)."""
    names = []
    # Group 1: morphology aggregated
    for suffix in ["axis_ratio", "area_mm2", "solidity", "eccentricity", "circularity"]:
        for agg in ["mean", "std", "max"]:
            names.append(f"{suffix}_{agg}")
    # Group 2: distribution
    names += [
        "stain_count", "total_stained_area_mm2", "stain_density_per_cm2",
        "spatial_spread_x_std", "spatial_spread_y_std", "spatial_spread_total",
        "nn_dist_mean_mm", "nn_dist_std_mm", "convex_hull_area_mm2",
    ]
    # Group 3: directionality
    names += [
        "mean_orientation_deg", "circular_variance",
        "convergence_x", "convergence_y", "convergence_dispersion",
    ]
    # Group 4: context (image_width/image_height excluded — camera resolution confounds)
    names += ["has_convergence"]
    return names


def extract_features_from_image(
    image_path: str,
    dpi: float = 600.0,
    substrate: str = "unknown",
    mechanism_class: str = "unknown",
    source_dataset: str = "unknown",
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: detect stains → extract features for one image.
    """
    stains, metadata = detect_stains(image_path, dpi=dpi, debug_dir=debug_dir)
    return extract_features(stains, metadata, substrate, mechanism_class, source_dataset)


def process_dataset(
    image_paths: List[str],
    dpi: float = 600.0,
    substrate: str = "unknown",
    mechanism_class: str = "unknown",
    source_dataset: str = "unknown",
    debug_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process a list of images and return a DataFrame of features.
    """
    records = []
    for i, path in enumerate(image_paths):
        if verbose:
            print(f"  [{i+1}/{len(image_paths)}] {Path(path).name}...", end=" ")
        try:
            feat = extract_features_from_image(
                path, dpi=dpi, substrate=substrate,
                mechanism_class=mechanism_class,
                source_dataset=source_dataset,
                debug_dir=debug_dir,
            )
            records.append(feat)
            if verbose:
                print(f"OK ({feat.get('stain_count', 0)} stains)")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <image_path> [dpi]")
        sys.exit(1)
    
    img_path = sys.argv[1]
    dpi = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    
    feat = extract_features_from_image(img_path, dpi=dpi, debug_dir="debug_output")
    for k, v in sorted(feat.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
