"""Generate synthetic 3D print test images for Printsight testing."""

import cv2
import numpy as np
import sys
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")


def _create_base_print(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple '3D print' shape (cube-like) on a clean background."""
    img = np.ones((h, w), dtype=np.uint8) * 230  # light gray background

    # Main body: a rectangle (the print)
    body_w, body_h = w // 3, h // 2
    cx, cy = w // 2, h // 2 + h // 8
    x1, y1 = cx - body_w // 2, cy - body_h // 2

    print_body = np.zeros_like(img, dtype=np.uint8)
    cv2.rectangle(print_body, (x1, y1), (x1 + body_w, y1 + body_h), 255, -1)

    # Add some layer-like lines
    for i in range(y1 + 5, y1 + body_h, 6):
        cv2.line(print_body, (x1 + 2, i), (x1 + body_w - 2, i), 200, 1)

    mask = print_body > 0
    img[mask] = 180  # medium gray print

    return img, print_body


def render_good_print(path: str) -> None:
    """Synthetic print with good quality."""
    h, w = 400, 600
    img, _ = _create_base_print(h, w)
    # Add subtle texture for sharpness
    noise = np.random.default_rng(0).normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    print(f"  Good print: {path}")


def render_stringing(path: str) -> None:
    """Synthetic print with noticeable stringing defects."""
    h, w = 400, 600
    img, body = _create_base_print(h, w)

    # Get the print region
    ys, xs = np.where(body > 0)
    if len(ys) == 0:
        ys, xs = np.array([100, 300]), np.array([100, 500])
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    rng = np.random.default_rng(42)

    # Draw dense stringing — thick, visible wisps
    # Set the background between the towers to have stringing
    for _ in range(20):
        start_y = rng.integers(y1 + 10, y1 + y2 // 2)
        start_x = x1 + 5
        end_x = x1 + x2 - 5
        # Draw stringing as curved lines
        pts = []
        for t in np.linspace(0, 1, 30):
            px = int(start_x + t * (end_x - start_x))
            py = int(start_y + rng.integers(-2, 3) + t * 30 * np.sin(t * np.pi * 4))
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], False, int(rng.integers(80, 140)), 1)

    # Add thick diagonal stringing strands
    for i in range(8):
        y_offset = rng.integers(0, y2 - y1 - 20)
        cv2.line(img,
                 (x1 + 5, y1 + 10 + y_offset),
                 (x1 + x2 - 5, y1 + y_offset + rng.integers(5, 15)),
                 int(rng.integers(80, 120)), 1)

    # Add very visible spiderweb-like stringing
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 3
    for angle in np.linspace(0, 2 * np.pi, 12):
        ex = int(cx + 60 * np.cos(angle))
        ey = int(cy + 40 * np.sin(angle))
        cv2.line(img, (cx, cy), (max(x1, min(x2, ex)),
                                  max(y1, min(y2, ey))), 110, 1)

    # Add subtle texture
    noise = rng.normal(0, 2, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(path, img)
    print(f"  Stringing test: {path}")


def render_poor_layers(path: str) -> None:
    """Synthetic print with obvious layer quality defects."""
    h, w = 400, 600
    img = np.ones((h, w), dtype=np.uint8) * 230

    # Main print body
    body_w, body_h = w // 3, h // 2
    cx, cy = w // 2, h // 2 + h // 8
    x1, y1 = cx - body_w // 2, cy - body_h // 2

    # Draw base print
    cv2.rectangle(img, (x1, y1), (x1 + body_w, y1 + body_h), 180, -1)

    rng = np.random.default_rng(42)

    # ── 1. Inconsistent layer spacing/thickness ──
    # Instead of uniform 6px spacing, vary the gaps
    y = y1 + 3
    while y < y1 + body_h:
        layer_thickness = int(rng.integers(2, 8))
        spacing = int(rng.integers(4, 14))

        # Draw the layer boundary
        brightness = int(rng.integers(155, 195))
        cv2.line(img, (x1, y), (x1 + body_w, y), brightness, layer_thickness)

        y += spacing

    # ── 2. Severe banding (z-band) ──
    # Alternating dark/light bands every ~40px
    for y in range(y1, y1 + body_h, 40):
        band_height = rng.integers(5, 12)
        if y + band_height < y1 + body_h:
            # Dark band
            cv2.rectangle(img, (x1, y), (x1 + body_w, y + band_height),
                          int(rng.integers(90, 130)), -1)
            # Add a bright line at band boundary
            if y + band_height + 1 < y1 + body_h:
                cv2.line(img, (x1, y + band_height), (x1 + body_w, y + band_height),
                         210, 1)

    # ── 3. Layer shifts ──
    shift_rows = [(y1 + body_h // 4, 4),
                  (y1 + body_h // 2, 7),
                  (y1 + 3 * body_h // 4, -3)]

    for shift_y, shift_amt in shift_rows:
        if shift_amt > 0:
            img[shift_y:y1 + body_h, x1:x1 + body_w - shift_amt] = \
                img[shift_y:y1 + body_h, x1 + shift_amt:x1 + body_w].copy()
            # Fill gap with different shade
            img[shift_y:y1 + body_h, x1 + body_w - shift_amt:x1 + body_w] = 150
        else:
            shift_amt = abs(shift_amt)
            img[shift_y:y1 + body_h, x1 + shift_amt:x1 + body_w] = \
                img[shift_y:y1 + body_h, x1:x1 + body_w - shift_amt].copy()
            img[shift_y:y1 + body_h, x1:x1 + shift_amt] = 150

    # ── 4. Surface blobs/zits ──
    for _ in range(rng.integers(5, 10)):
        bx = rng.integers(x1 + 5, x1 + body_w - 5)
        by = rng.integers(y1 + 10, y1 + body_h - 10)
        br = rng.integers(3, 7)
        cv2.circle(img, (bx, by), br, int(rng.integers(100, 140)), -1)

    cv2.imwrite(path, img)
    print(f"  Poor layers test: {path}")


def render_warping(path: str) -> None:
    """Synthetic print with warped corners."""
    h, w = 400, 600
    img = np.ones((h, w), dtype=np.uint8) * 230

    # Draw a shape with lifted bottom corners
    cx, cy = w // 2, h // 2 + h // 8
    body_w, body_h = w // 3, h // 2
    x1, y1 = cx - body_w // 2, cy - body_h // 2

    # Rectangle with curved bottom edge (warped)
    pts = np.array([
        [x1, y1],  # top-left
        [x1 + body_w, y1],  # top-right
        [x1 + body_w - 5, y1 + body_h - 10],  # right edge, lifted
        [x1 + body_w - 15, y1 + body_h - 5],  # corner lifted
        [cx, y1 + body_h + 5],  # center of bottom (stays down)
        [x1 + 15, y1 + body_h - 5],  # left corner lifted
        [x1 + 5, y1 + body_h - 10],  # left edge, lifted
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts], 180)
    cv2.polylines(img, [pts], True, 120, 1)
    # Add texture
    noise = np.random.default_rng(1).normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(path, img)
    print(f"  Warping test: {path}")


def render_all_good(path: str) -> None:
    """Clean print with no defects — should score high on all metrics."""
    h, w = 400, 600
    img = np.ones((h, w), dtype=np.uint8) * 230

    # Clean rectangular print with regular horizontal lines (good layers)
    cx, cy = w // 2, h // 2 + h // 8
    body_w, body_h = w // 3, h // 2
    x1, y1 = cx - body_w // 2, cy - body_h // 2

    cv2.rectangle(img, (x1, y1), (x1 + body_w, y1 + body_h), 180, -1)

    # Regular, clean layer lines
    for i in range(y1 + 4, y1 + body_h, 6):
        cv2.line(img, (x1 + 1, i), (x1 + body_w - 1, i), 175, 1)

    cv2.imwrite(path, img)
    print(f"  All good test: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating test images...")
    render_good_print(os.path.join(OUTPUT_DIR, "good_print.png"))
    render_stringing(os.path.join(OUTPUT_DIR, "stringing_test.png"))
    render_poor_layers(os.path.join(OUTPUT_DIR, "poor_layers.png"))
    render_warping(os.path.join(OUTPUT_DIR, "warping_test.png"))
    render_all_good(os.path.join(OUTPUT_DIR, "perfect_print.png"))
    print("Done.")


if __name__ == "__main__":
    main()
