"""Annotator — draws defect markings on the original image for visual diagnosis."""

import cv2
import numpy as np


def draw_annotations(img_bgr: np.ndarray,
                     stringing_data: dict | None = None,
                     layer_data: dict | None = None,
                     warping_data: dict | None = None) -> np.ndarray:
    """Draw colored markings on the image showing detected defects.

    Colors:
        🔴 Red    — stringing (thin strand contours)
        🟡 Yellow — layer quality issues (horizontal bands)
        🔵 Blue   — warping (bottom edge deviation markers)

    Returns:
        Annotated BGR image (draws on a copy of the input)
    """
    h, w = img_bgr.shape[:2]
    ann = img_bgr.copy()

    # ── Stringing: red polylines on detected thin features ──
    if stringing_data and stringing_data.get("contours"):
        for cnt in stringing_data["contours"]:
            if len(cnt) >= 2:
                cv2.polylines(ann, [cnt], False, (0, 0, 255), 1)  # red

        # Count label
        count = len(stringing_data["contours"])
        cv2.putText(ann, f"Stringing: {count} strand(s)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ── Layer issues: yellow horizontal bands ──
    if layer_data and layer_data.get("bands"):
        for y_start, y_end in layer_data["bands"]:
            alpha = 0.15
            overlay = ann.copy()
            cv2.rectangle(overlay, (0, y_start), (w, y_end), (0, 255, 255), -1)
            cv2.addWeighted(overlay, alpha, ann, 1 - alpha, 0, ann)
            cv2.rectangle(ann, (0, y_start), (w, y_end), (0, 255, 255), 1)

        if layer_data.get("regularity", 1.0) < 0.5:
            cv2.putText(ann, f"Layer irregularity detected", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif layer_data.get("peak_count", 0) > 0 and layer_data.get("regularity", 1.0) < 0.8:
            cv2.putText(ann, f"Minor layer variation", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ── Warping: blue markers on lifted bottom edge ──
    if warping_data and warping_data.get("bottom_points") is not None:
        pts = warping_data["bottom_points"]
        deviations = warping_data.get("deviations", [])
        avg_dev = warping_data.get("avg_deviation", 0)

        # Draw ideal bottom line in green
        if warping_data.get("line_vx") is not None:
            vx, vy, cx, cy = (warping_data["line_vx"],
                              warping_data["line_vy"],
                              warping_data["line_cx"],
                              warping_data["line_cy"])
            # Extend the line across the image
            x1_line = int(cx - vx * 500)
            y1_line = int(cy - vy * 500)
            x2_line = int(cx + vx * 500)
            y2_line = int(cy + vy * 500)
            cv2.line(ann, (x1_line, y1_line), (x2_line, y2_line),
                     (0, 255, 0), 1, cv2.LINE_AA)

        # Mark each bottom point — red if lifted, green if flat
        for i, pt in enumerate(pts):
            px, py = int(pt[0]), int(pt[1])
            if i < len(deviations) and deviations[i] > avg_dev * 1.5:
                # Lifted corner — red circle
                cv2.circle(ann, (px, py), 4, (255, 0, 0), -1)  # blue
                cv2.circle(ann, (px, py), 4, (255, 255, 255), 1)
            else:
                # Normal point — small green dot
                cv2.circle(ann, (px, py), 2, (0, 255, 0), -1)

        lifted = warping_data.get("lifted_count", 0)
        if lifted > 0:
            cv2.putText(ann, f"Warping: {lifted} lifted corner(s)", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ── Score overlay in top-right ──
    overall = stringing_data.get("score", 0) * 0.3 + \
              layer_data.get("score", 0) * 0.4 + \
              warping_data.get("score", 0) * 0.3

    # Score box background
    cv2.rectangle(ann, (w - 220, 5), (w - 5, 40), (20, 20, 20), -1)
    cv2.rectangle(ann, (w - 220, 5), (w - 5, 40), (100, 100, 100), 1)
    cv2.putText(ann, f"Quality: {overall:.0%}", (w - 210, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return ann
