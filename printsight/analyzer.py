"""Printsight analyzer — runs all defect detectors and produces quality report."""

import cv2
import numpy as np

from .detectors import detect_stringing, detect_layer_quality, detect_warping


def analyze(image_path: str) -> dict:
    """Full analysis pipeline: load image → run all detectors → aggregate score.

    Args:
        image_path: Path to the 3D print photo

    Returns:
        dict with quality_score, per-detector results, and summary
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # Clarity check: skip if image is too blurry
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if laplacian_var < 10:
        return {
            "quality_score": 0.0,
            "error": f"Image too blurry (sharpness: {laplacian_var:.1f}). Please take a clearer photo.",
        }

    # Run all detectors
    stringing_score, stringing_details = detect_stringing(img_gray)
    layer_score, layer_details = detect_layer_quality(img_gray)
    warping_score, warping_details = detect_warping(img_gray)

    # Weighted overall score
    # Stringing: 30%, Layers: 40%, Warping: 30%
    weights = {
        "stringing": 0.30,
        "layer_quality": 0.40,
        "warping": 0.30,
    }
    overall = (
        stringing_score * weights["stringing"]
        + layer_score * weights["layer_quality"]
        + warping_score * weights["warping"]
    )

    # Grading
    grade = _grade(overall)
    image_size = len(cv2.imencode('.png', img)[1])

    return {
        "quality_score": round(overall, 3),
        "grade": grade,
        "image": {
            "path": image_path,
            "width": w,
            "height": h,
            "size_bytes": image_size,
        },
        "stringing": {
            "score": round(stringing_score, 3),
            "details": stringing_details,
        },
        "layer_quality": {
            "score": round(layer_score, 3),
            "details": layer_details,
        },
        "warping": {
            "score": round(warping_score, 3),
            "details": warping_details,
        },
        "weights": weights,
    }


def _grade(score: float) -> str:
    """Convert numeric score to grade."""
    if score >= 0.90:
        return "Excellent"
    elif score >= 0.75:
        return "Good"
    elif score >= 0.50:
        return "Fair"
    elif score >= 0.25:
        return "Poor"
    else:
        return "Failed"
