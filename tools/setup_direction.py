"""
setup_direction.py — Zone-to-Zone Setup Tool
============================================
Draw two zones to define entry/exit direction.

Usage:
    python tools/setup_direction.py --camera_name visitor_camera

Controls:
    Left-click      → Add point to current zone
    SPACE           → Switch between Zone A (Gate) and Zone B (Approach)
    R               → Reset current zone
    S               → Save both zones to config.json
    Q               → Quit without saving
"""

import cv2
import json
import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CONFIG_JSON = os.path.join(PROJECT_ROOT, "config.json")
MAX_DISPLAY_WIDTH  = 1280
MAX_DISPLAY_HEIGHT = 720

# Colors (BGR)
ZONE_A_COLOR = (255, 0, 0)   # Blue (Gate)
ZONE_B_COLOR = (0, 0, 255)   # Red (Approach)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)

def _scale_factors(native_w, native_h):
    scale = min(MAX_DISPLAY_WIDTH / native_w, MAX_DISPLAY_HEIGHT / native_h, 1.0)
    return int(native_w * scale), int(native_h * scale), native_w / (native_w * scale), native_h / (native_h * scale)

def _render(display_frame, zone_a, zone_b, active_mode, scale_x, scale_y):
    vis = display_frame.copy()
    
    # Header
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 45), (30, 30, 30), -1)
    mode_text = "ZONE A (GATE)" if active_mode == 'A' else "ZONE B (APPROACH)"
    mode_color = ZONE_A_COLOR if active_mode == 'A' else ZONE_B_COLOR
    
    hint = f"MODE: {mode_text} | SPACE=Switch | S=Save | R=Reset | Q=Quit"
    cv2.putText(vis, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    cv2.rectangle(vis, (vis.shape[1]-200, 10), (vis.shape[1]-10, 35), mode_color, -1)

    # Draw Zone A
    if zone_a:
        pts = np.array([(int(p[0]/scale_x), int(p[1]/scale_y)) for p in zone_a])
        cv2.polylines(vis, [pts], True, ZONE_A_COLOR, 2)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], ZONE_A_COLOR)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.putText(vis, "GATE (A)", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_A_COLOR, 2)

    # Draw Zone B
    if zone_b:
        pts = np.array([(int(p[0]/scale_x), int(p[1]/scale_y)) for p in zone_b])
        cv2.polylines(vis, [pts], True, ZONE_B_COLOR, 2)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], ZONE_B_COLOR)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.putText(vis, "APPROACH (B)", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_B_COLOR, 2)

    return vis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_name", required=True)
    args = parser.parse_args()

    with open(CONFIG_JSON, "r") as f:
        config = json.load(f)

    source = config["camera_url"][args.camera_name]
    cap = cv2.VideoCapture(source)
    ret, native_frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] Could not read frame")
        return

    nh, nw = native_frame.shape[:2]
    dw, dh, sx, sy = _scale_factors(nw, nh)
    display_base = cv2.resize(native_frame, (dw, dh))

    zone_a, zone_b = [], []
    active_mode = 'A'

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nx, ny = int(mx * sx), int(my * sy)
            if active_mode == 'A': zone_a.append((nx, ny))
            else: zone_b.append((nx, ny))

    cv2.namedWindow("Setup Zones")
    cv2.setMouseCallback("Setup Zones", on_mouse)

    while True:
        vis = _render(display_base, zone_a, zone_b, active_mode, sx, sy)
        cv2.imshow("Setup Zones", vis)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            active_mode = 'B' if active_mode == 'A' else 'A'
        elif key == ord('r'):
            if active_mode == 'A': zone_a.clear()
            else: zone_b.clear()
        elif key == ord('s'):
            if len(zone_a) < 3 or len(zone_b) < 3:
                print("[ERROR] Both zones need at least 3 points")
                continue
            
            # Save relative coordinates
            rel_a = [(round(p[0]/nw, 4), round(p[1]/nh, 4)) for p in zone_a]
            rel_b = [(round(p[0]/nw, 4), round(p[1]/nh, 4)) for p in zone_b]
            
            if "direction_config" not in config: config["direction_config"] = {}
            if args.camera_name not in config["direction_config"]:
                config["direction_config"][args.camera_name] = {}
            
            cam_conf = config["direction_config"][args.camera_name]
            cam_conf["enabled"] = True
            cam_conf["method"] = "zone_transition"
            cam_conf["zone_a"] = rel_a
            cam_conf["zone_b"] = rel_b
            cam_conf["mode"] = "entry"
            
            with open(CONFIG_JSON, "w") as f:
                json.dump(config, f, indent=2)
            print(f"[SUCCESS] Saved Zones for {args.camera_name}")
            break
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
