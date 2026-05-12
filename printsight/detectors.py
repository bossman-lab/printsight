"""Individual defect detectors for 3D print quality inspection.

Each detector takes a grayscale image (numpy.ndarray) and returns:
    - score: float 0.0 (worst) to 1.0 (perfect)
    - details: dict with per-defect metrics (JSON-safe values)
    - annotation: dict with drawing primitives (contours, points, etc.)
"""

import cv2
import numpy as np


def detect_stringing(img_gray: np.ndarray) -> tuple[float, dict, dict]:
    """Detect stringing (thin wisps of plastic between parts).

    Algorithm:
        1. Dual approach: edge-based + morphology-based for best coverage
        2. Find thin line-like features using both methods
        3. Count and measure stringing instances

    Returns:
        score: 1.0 = no stringing, 0.0 = severe stringing
        details: {"count": N, "total_length_px": L, "max_length_px": L}
        annotation: {"contours": [np.array(...), ...]}
    """
    h, w = img_gray.shape
    img_area = h * w

    # ── Approach 1: Canny edge + Hough lines ──
    edges = cv2.Canny(img_gray, 30, 100)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=20, minLineLength=15, maxLineGap=10)

    line_lengths = []
    line_contours = []  # for annotation
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle > 15:  # not horizontal
                line_lengths.append(length)
                line_contours.append(np.array([[x1, y1], [x2, y2]], dtype=np.int32))

    # ── Approach 2: Morphological thin feature detection ──
    thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(thresh, erode_kernel, iterations=1)
    thin_features = cv2.subtract(thresh, eroded)
    thin_features = cv2.dilate(thin_features, kernel, iterations=1)

    morph_contours_all, _ = cv2.findContours(thin_features, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

    morph_lengths = []
    morph_contours = []
    for cnt in morph_contours_all:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
        perimeter = cv2.arcLength(cnt, closed=True)
        thinness = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 0
        if thinness > 2.0 and perimeter > 15:
            morph_lengths.append(perimeter)
            morph_contours.append(cnt)

    # Combine both approaches
    all_lengths = line_lengths + morph_lengths
    count = len(all_lengths)
    total_length = sum(all_lengths)
    max_length = max(all_lengths) if all_lengths else 0

    severity = total_length / img_area if img_area > 0 else 0
    score = max(0.0, 1.0 - severity * 30)

    return score, {
        "count": count,
        "total_length_px": round(total_length, 1),
        "max_length_px": round(max_length, 1),
        "severity_ratio": round(severity, 5),
    }, {
        "score": score,
        "contours": line_contours + morph_contours,
    }


def detect_layer_quality(img_gray: np.ndarray) -> tuple[float, dict, dict]:
    """Detect layer quality issues (inconsistent extrusion, Z-banding).

    Returns:
        score: 1.0 = perfect layers, 0.0 = severe layer issues
        details: {"peak_std": S, "dominant_freq": F, "regularity": R}
        annotation: {"bands": [(y1,y2), ...], "regularity": R, "peak_count": N}
    """
    h, w = img_gray.shape

    # Horizontal gradient — captures layer boundaries
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_mag = np.abs(grad_x)

    # Vertical projection
    row_profile = np.sum(grad_mag, axis=1)
    row_profile = row_profile / np.max(row_profile) if np.max(row_profile) > 0 else row_profile

    # Find peaks
    peaks = []
    window = max(5, h // 50)
    for i in range(window, h - window):
        segment = row_profile[i - window:i + window + 1]
        if row_profile[i] == np.max(segment) and row_profile[i] > np.mean(row_profile) * 1.3:
            peaks.append(i)

    # Peak spacing regularity
    if len(peaks) > 3:
        spacings = np.diff(peaks)
        peak_std = float(np.std(spacings))
        peak_mean = float(np.mean(spacings))
        cv_peaks = peak_std / peak_mean if peak_mean > 0 else 1.0
        regularity = max(0.0, 1.0 - cv_peaks)
    else:
        peak_std = 0
        regularity = 0.5

    # FFT analysis
    fft = np.fft.rfft(row_profile)
    fft_mag = np.abs(fft[1:])
    if len(fft_mag) > 0:
        dominant_idx = np.argmax(fft_mag) + 1
        dominant_freq = dominant_idx / h
        total_energy = np.sum(fft_mag)
        energy_ratio = fft_mag[dominant_idx - 1] / total_energy if total_energy > 0 else 0
    else:
        dominant_freq = 0
        energy_ratio = 0

    score = 0.5 * regularity + 0.5 * min(1.0, energy_ratio * 20)

    # Find problematic bands: areas with irregular peak spacing
    bad_bands = []
    if len(peaks) > 4:
        for i in range(1, len(peaks) - 1):
            local_spacing = peaks[i + 1] - peaks[i]
            mean_spacing = np.mean(spacings)
            if abs(local_spacing - mean_spacing) > mean_spacing * 0.5:
                # This peak area has irregular spacing
                y_start = max(0, peaks[i] - window)
                y_end = min(h, peaks[i] + window)
                bad_bands.append((y_start, y_end))

    # If no bad bands found but score is low, flag worst areas
    if not bad_bands and score < 0.6 and len(peaks) > 2:
        # Find the most irregular region
        max_gap_idx = np.argmax(np.abs(np.diff(peaks) - np.mean(spacings)))
        if max_gap_idx < len(peaks) - 1:
            y_start = max(0, peaks[max_gap_idx] - window * 2)
            y_end = min(h, peaks[max_gap_idx + 1] + window * 2)
            bad_bands.append((y_start, y_end))

    # Limit bands to avoid overwhelming the display
    bad_bands = bad_bands[:5]

    return score, {
        "peak_count": len(peaks),
        "peak_std_px": round(peak_std, 1),
        "dominant_freq_norm": round(float(dominant_freq), 4),
        "regularity": round(regularity, 3),
    }, {
        "score": score,
        "bands": bad_bands,
        "regularity": regularity,
        "peak_count": len(peaks),
    }


def detect_warping(img_gray: np.ndarray) -> tuple[float, dict, dict]:
    """Detect warping (corners lifting from build plate).

    Returns:
        score: 1.0 = no warping, 0.0 = severe warping
        details: {"bottom_deviation": D, "lifted_corners": N}
        annotation: {"bottom_points": ndarray, "deviations": list,
                     "line_vx/vy/cx/cy": float, "avg_deviation": float,
                     "lifted_count": int}
    """
    h, w = img_gray.shape

    _, thresh = cv2.threshold(img_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.sum(thresh == 255)
    total_pixels = h * w
    if white_pixels > total_pixels * 0.5:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.5, {"bottom_deviation": 0, "lifted_corners": 0, "error": "no contour found"}, {}

    main_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(main_contour)
    bottom_y_thresh = y + int(bh * 0.8)

    bottom_points = []
    for point in main_contour:
        px, py = point[0]
        if py >= bottom_y_thresh:
            bottom_points.append([px, py])

    annotation_data = {}

    if len(bottom_points) < 5:
        return 0.5, {"bottom_deviation": 0, "lifted_corners": 0,
                     "error": "not enough bottom points"}, annotation_data

    bottom_points_arr = np.array(bottom_points, dtype=np.float32)

    [vx, vy, cx, cy] = cv2.fitLine(bottom_points_arr, cv2.DIST_L2, 0, 0.01, 0.01)

    direction = np.array([vx[0], vy[0]], dtype=np.float32)
    center = np.array([cx[0], cy[0]], dtype=np.float32)
    direction_norm = np.linalg.norm(direction)

    deviations = []
    for pt in bottom_points_arr:
        vec = pt - center
        dist = abs(vec[0] * direction[1] - vec[1] * direction[0]) / direction_norm
        deviations.append(float(dist))

    if not deviations:
        return 0.5, {"bottom_deviation": 0, "lifted_corners": 0}, annotation_data

    max_deviation = max(deviations)
    avg_deviation = np.mean(deviations)

    deviation_ratio = max_deviation / bh if bh > 0 else 0
    score = max(0.0, 1.0 - deviation_ratio * 10)

    x_coords = bottom_points_arr[:, 0]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    x_range = x_max - x_min

    lifted_count = 0
    for i, pt in enumerate(bottom_points_arr):
        if deviations[i] > avg_deviation * 1.5:
            if pt[0] - x_min < x_range * 0.15 or x_max - pt[0] < x_range * 0.15:
                lifted_count += 1

    # Build annotation data
    annotation_data = {
        "bottom_points": bottom_points_arr,
        "deviations": deviations,
        "avg_deviation": avg_deviation,
        "lifted_count": lifted_count,
        "line_vx": float(vx[0]),
        "line_vy": float(vy[0]),
        "line_cx": float(cx[0]),
        "line_cy": float(cy[0]),
        "score": score,
    }

    return score, {
        "bottom_deviation_px": round(float(max_deviation), 1),
        "avg_deviation_px": round(float(avg_deviation), 2),
        "lifted_corners": lifted_count,
        "deviation_ratio": round(float(deviation_ratio), 4),
    }, annotation_data
