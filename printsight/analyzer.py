"""Printsight analyzer — runs all defect detectors and produces quality report."""

import cv2
import numpy as np
import os

from .detectors import detect_stringing, detect_layer_quality, detect_warping
from .annotator import draw_annotations


def analyze(image_path: str, annotate: bool = False) -> dict:
    """Full analysis pipeline: load image → run all detectors → aggregate score.

    Args:
        image_path: Path to the 3D print photo
        annotate: If True, also produce annotated image with defect markings

    Returns:
        dict with quality_score, per-detector results, and if annotate=True,
        also includes "annotated_path" key with the output image path
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # Clarity check
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if laplacian_var < 10:
        return {
            "quality_score": 0.0,
            "error": f"Image too blurry (sharpness: {laplacian_var:.1f}). Please take a clearer photo.",
        }

    # Run all detectors — now each returns (score, details, annotation)
    stringing_score, stringing_details, stringing_ann = detect_stringing(img_gray)
    layer_score, layer_details, layer_ann = detect_layer_quality(img_gray)
    warping_score, warping_details, warping_ann = detect_warping(img_gray)

    # Weighted overall score
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

    result = {
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

    # Generate annotated image if requested
    if annotate:
        annotated_img = draw_annotations(
            img,
            stringing_data=stringing_ann,
            layer_data=layer_ann,
            warping_data=warping_ann,
        )
        # Save next to the original
        base, ext = os.path.splitext(image_path)
        annotated_path = f"{base}_annotated{ext}"
        cv2.imwrite(annotated_path, annotated_img)
        result["annotated_path"] = annotated_path

    return result


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
