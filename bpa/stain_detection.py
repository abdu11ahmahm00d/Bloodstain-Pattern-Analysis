"""
stain_detection.py — Robust bloodstain detection using L*a*b* color space + Otsu thresholding.

Replaces HSV approach which fails on dried blood (brown vs red).
L*a*b* a* channel encodes red-green axis and is robust to lighting/drying.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class StainInfo:
    """Per-stain measurements from fitted ellipse."""
    centroid_x: float          # px
    centroid_y: float          # px
    area_px: float             # pixels
    area_mm2: float            # mm² (using DPI)
    major_axis: float          # px
    minor_axis: float          # px
    axis_ratio: float          # major / minor
    eccentricity: float        # ellipse eccentricity
    orientation_deg: float     # angle of major axis (degrees)
    solidity: float            # area / convex hull area
    circularity: float         # 4π·area / perimeter²
    contour: np.ndarray        # raw contour (not stored in CSV)


def detect_stains(
    image_path: str,
    dpi: float = 600.0,
    min_area_px: int = 50,
    max_area_ratio: float = 0.1,  # ignore stains > 10% of image area
    morph_kernel_size: int = 5,
    debug_dir: Optional[str] = None,
    max_dimension: int = 4000,  # resize if larger dimension exceeds this
) -> Tuple[List[StainInfo], dict]:
    """
    Detect bloodstains in an image using L*a*b* + Otsu thresholding.
    
    Args:
        image_path: path to JPG/PNG image
        dpi: dots per inch for mm² conversion
        min_area_px: minimum contour area in pixels (filters noise)
        max_area_ratio: ignore contours larger than this fraction of image area
        morph_kernel_size: kernel size for morphological operations
        debug_dir: if set, save debug images showing detected stains
        max_dimension: resize image if width or height exceeds this (saves memory)
        
    Returns:
        stains: list of StainInfo objects
        metadata: dict with image-level stats
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    h, w = img.shape[:2]
    
    # Resize large images to save memory/time
    scale = 1.0
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
        # Adjust min_area_px for scaled image
        min_area_px = max(int(min_area_px * scale * scale), 10)
    
    pixels_per_cm = dpi / 2.54
    image_area = h * w
    
    # --- Step 1: Convert to L*a*b* and extract a* channel ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # a* channel: positive = red, negative = green
    # Bloodstains (fresh or dried) are red/brown → high a* values
    # Background (white poster board) → low a* values (neutral gray)
    
    # --- Step 2: Otsu thresholding on a* channel ---
    _, binary = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- Step 3: Morphological cleaning ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    # Close small gaps within stains
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # --- Step 4: Find connected components / contours ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stains: List[StainInfo] = []
    for contour in contours:
        area_px = cv2.contourArea(contour)
        
        # Filter: too small
        if area_px < min_area_px:
            continue
        
        # Filter: too large (probably background artifact)
        if area_px / image_area > max_area_ratio:
            continue
        
        # Filter: need at least 5 points for ellipse fitting
        if len(contour) < 5:
            continue
        
        # --- Step 5: Fit ellipse ---
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (width, height), angle = ellipse
        
        # Ensure major axis >= minor axis
        major = max(width, height)
        minor = min(width, height)
        
        # Avoid division by zero
        if minor < 1.0:
            minor = 1.0
        
        axis_ratio = major / minor
        
        # Eccentricity: sqrt(1 - (b/a)^2)
        eccentricity = np.sqrt(1.0 - (minor / major) ** 2)
        
        # Convert area to mm²
        area_mm2 = area_px / (pixels_per_cm ** 2) * 100  # cm² to mm²
        
        # Perimeter for circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area_px / (perimeter ** 2)) if perimeter > 0 else 0
        
        # Solidity: area / convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0
        
        stain = StainInfo(
            centroid_x=cx,
            centroid_y=cy,
            area_px=area_px,
            area_mm2=area_mm2,
            major_axis=major,
            minor_axis=minor,
            axis_ratio=axis_ratio,
            eccentricity=eccentricity,
            orientation_deg=angle,
            solidity=solidity,
            circularity=circularity,
            contour=contour,
        )
        stains.append(stain)
    
    # --- Image-level metadata ---
    metadata = {
        "image_path": str(image_path),
        "image_width": w,
        "image_height": h,
        "dpi": dpi,
        "pixels_per_cm": pixels_per_cm,
        "stain_count": len(stains),
        "total_stained_area_px": sum(s.area_px for s in stains),
        "total_stained_area_mm2": sum(s.area_mm2 for s in stains),
        "binary_mask_path": None,
    }
    
    # --- Debug output ---
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        
        # Draw detected stains on original image
        debug_img = img.copy()
        for i, stain in enumerate(stains):
            # Draw ellipse
            cv2.ellipse(debug_img, 
                       ((stain.centroid_x, stain.centroid_y), 
                        (stain.major_axis, stain.minor_axis), 
                        stain.orientation_deg),
                       (0, 255, 0), 2)
            # Label
            cv2.putText(debug_img, str(i), 
                       (int(stain.centroid_x), int(stain.centroid_y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        stem = Path(image_path).stem
        cv2.imwrite(str(debug_path / f"{stem}_debug.jpg"), debug_img)
        cv2.imwrite(str(debug_path / f"{stem}_binary.jpg"), binary)
        
        metadata["binary_mask_path"] = str(debug_path / f"{stem}_binary.jpg")
    
    return stains, metadata


def detect_stains_adaptive(
    image_path: str,
    dpi: float = 600.0,
    min_area_px: int = 50,
    debug_dir: Optional[str] = None,
) -> Tuple[List[StainInfo], dict]:
    """
    Alternative detector using adaptive thresholding.
    Better for images with uneven lighting or non-uniform backgrounds.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    h, w = img.shape[:2]
    pixels_per_cm = dpi / 2.54
    
    # Convert to grayscale and apply adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert: bloodstains are dark, background is light
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stains = []
    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px < min_area_px or len(contour) < 5:
            continue
        if area_px / (h * w) > 0.1:
            continue
        
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (width, height), angle = ellipse
        major = max(width, height)
        minor = max(min(width, height), 1.0)
        
        perimeter = cv2.arcLength(contour, True)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        
        stains.append(StainInfo(
            centroid_x=cx, centroid_y=cy,
            area_px=area_px,
            area_mm2=area_px / (pixels_per_cm ** 2) * 100,
            major_axis=major, minor_axis=minor,
            axis_ratio=major / minor,
            eccentricity=np.sqrt(1.0 - (minor / major) ** 2),
            orientation_deg=angle,
            solidity=area_px / hull_area if hull_area > 0 else 0,
            circularity=(4 * np.pi * area_px / (perimeter ** 2)) if perimeter > 0 else 0,
            contour=contour,
        ))
    
    metadata = {
        "image_path": str(image_path),
        "image_width": w, "image_height": h,
        "dpi": dpi, "pixels_per_cm": pixels_per_cm,
        "stain_count": len(stains),
        "total_stained_area_px": sum(s.area_px for s in stains),
        "total_stained_area_mm2": sum(s.area_mm2 for s in stains),
    }
    
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        debug_img = img.copy()
        for s in stains:
            cv2.ellipse(debug_img,
                       ((s.centroid_x, s.centroid_y),
                        (s.major_axis, s.minor_axis), s.orientation_deg),
                       (0, 255, 0), 2)
        stem = Path(image_path).stem
        cv2.imwrite(str(debug_path / f"{stem}_adaptive_debug.jpg"), debug_img)
    
    return stains, metadata


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stain_detection.py <image_path> [dpi] [debug_dir]")
        sys.exit(1)
    
    img_path = sys.argv[1]
    dpi = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    debug = sys.argv[3] if len(sys.argv) > 3 else None
    
    stains, meta = detect_stains(img_path, dpi=dpi, debug_dir=debug)
    print(f"Detected {meta['stain_count']} stains")
    print(f"Total stained area: {meta['total_stained_area_mm2']:.1f} mm²")
    for i, s in enumerate(stains[:5]):
        print(f"  Stain {i}: area={s.area_mm2:.2f}mm², ratio={s.axis_ratio:.2f}, "
              f"ecc={s.eccentricity:.3f}, orient={s.orientation_deg:.1f}°")
    if len(stains) > 5:
        print(f"  ... and {len(stains) - 5} more")
