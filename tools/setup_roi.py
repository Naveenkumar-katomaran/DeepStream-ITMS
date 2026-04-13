"""
setup_roi.py — ROI Polygon Setup Tool
======================================
Run once per camera to configure the Region of Interest (ROI) polygon dynamically.

Usage:
    python tools/setup_roi.py --camera_name visitor_camera  # reads from config.json

Controls (after the window opens):
    Left-click      → add a point to the polygon (minimum 3 points)
    R               → reset / clear points
    S               → save to config.json as fractional coordinates
    Q               → quit without saving
"""

import cv2
import json
import os
import sys
import argparse
import numpy as np

# ── Allow running from the project root ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_JSON = os.path.join(PROJECT_ROOT, "config.json")
MAX_DISPLAY_WIDTH  = 1280
MAX_DISPLAY_HEIGHT = 720

# ── Colours ───────────────────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
GREEN   = (0, 220,  60)
YELLOW  = (0, 220, 220)
BLACK   = (0,   0,   0)
CYAN    = (220, 200,  0)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scale_factors(native_w, native_h):
    """Return (display_w, display_h, scale_x, scale_y)."""
    scale = min(MAX_DISPLAY_WIDTH / native_w, MAX_DISPLAY_HEIGHT / native_h, 1.0)
    dw = int(native_w * scale)
    dh = int(native_h * scale)
    return dw, dh, native_w / dw, native_h / dh

def _render(display_frame, points, scale_x, scale_y):
    """
    Rebuild the annotation overlay on *display_frame*.
    points: list of (native_x, native_y).
    """
    vis = display_frame.copy()

    # Instructions bar
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (20, 20, 20), cv2.FILLED)
    hint = (f"Click to add points ({len(points)} set) | S=Save  R=Reset  Q=Quit")
    cv2.putText(vis, hint, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)

    if not points:
        return vis

    # Draw small dots for each point and lines between them
    dp_list = [(int(p[0] / scale_x), int(p[1] / scale_y)) for p in points]
    
    for i, dp in enumerate(dp_list):
        cv2.circle(vis, dp, 5, YELLOW, -1)
        if i > 0:
            cv2.line(vis, dp_list[i-1], dp, GREEN, 2, cv2.LINE_AA)
            
    # Close the polygon if 3+ points
    if len(points) >= 3:
        cv2.line(vis, dp_list[-1], dp_list[0], GREEN, 2, cv2.LINE_AA)
        
        # Draw translucent fill
        overlay = vis.copy()
        cv2.fillPoly(overlay, [np.array(dp_list)], GREEN)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

    return vis

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ROI polygon setup tool")
    parser.add_argument("--camera_name", required=True,
                        help="Camera name as defined in config.json")
    args = parser.parse_args()

    if not os.path.exists(CONFIG_JSON):
        print(f"[ERROR] config.json not found at {CONFIG_JSON}")
        sys.exit(1)

    with open(CONFIG_JSON, "r") as f:
        config = json.load(f)

    if args.camera_name not in config.get("camera_url", {}):
        print(f"[ERROR] Camera '{args.camera_name}' not found closely in config.json 'camera_url'.")
        sys.exit(1)

    source = config["camera_url"][args.camera_name]
    print(f"[SETUP ROI] Opening source for {args.camera_name}: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(int(source))
        except ValueError:
            pass
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    for _ in range(10):
        cap.grab()

    ret, native_frame = cap.read()
    cap.release()

    if not ret or native_frame is None:
        print("[ERROR] Failed to read a frame from the source.")
        sys.exit(1)

    native_h, native_w = native_frame.shape[:2]
    disp_w, disp_h, scale_x, scale_y = _scale_factors(native_w, native_h)
    display_base = cv2.resize(native_frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    WIN = f"LPR ROI Setup - {args.camera_name} [click to add points | S=Save | R=Reset | Q=Quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    points = []

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nx = int(mx * scale_x)
            ny = int(my * scale_y)
            points.append((nx, ny))
            print(f"[SETUP ROI] Point added: native({nx},{ny}) total={len(points)}")

    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        vis = _render(display_base, points, scale_x, scale_y)
        cv2.imshow(WIN, vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('r') or key == ord('R'):
            points.clear()
            print("[SETUP ROI] Reset.")

        elif key == ord('s') or key == ord('S'):
            if len(points) < 3:
                print("[SETUP ROI] Need at least 3 points for a polygon.")
                continue

            # Convert to relative fractional coordinates
            relative_points = [(round(p[0] / native_w, 4), round(p[1] / native_h, 4)) for p in points]
            
            if "direction_config" not in config:
                config["direction_config"] = {}
                
            if args.camera_name not in config["direction_config"]:
                config["direction_config"][args.camera_name] = {}
            
            config["direction_config"][args.camera_name]["roi"] = relative_points
            config["direction_config"][args.camera_name]["enabled"] = True

            with open(CONFIG_JSON, "w") as f:
                json.dump(config, f, indent=2)

            print(f"[SETUP ROI] ✅ Saved ROI with {len(points)} relative points to {CONFIG_JSON}")
            break

        elif key == ord('q') or key == ord('Q'):
            print("[SETUP ROI] Quit.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
