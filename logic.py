from shapely.geometry import Point, Polygon
import logging

class LogicEngine:
    def __init__(self, config, submitter=None):
        self.config = config
        self.submitter = submitter
        
        self.camera_map = {}
        # vehicle_states: {id: {crops, v_history, frame_count, zone_history, last_seen, source_id, status}}
        self.vehicle_states = {} 
        self.finalized_ids = set() # To prevent double-finalization
        
        self._load_config()

    def _load_config(self):
        dir_cfg = self.config.get("direction_config", {})
        enabled_cams = self.config.get("enabled_cameras", [])
        
        for cam_name in enabled_cams:
            cfg = dir_cfg.get(cam_name, {})
            if not cfg: continue
            
            roi_poly = Polygon(cfg.get("roi", []))
            zone_a = Polygon(cfg.get("zone_a", []))
            zone_b = Polygon(cfg.get("zone_b", []))
            
            self.camera_map[cam_name] = {
                "roi": roi_poly,
                "zone_a": zone_a,
                "zone_b": zone_b,
                "name": cam_name
            }

    def check_polygons(self, track_id, source_id, x, y, w, h, fw, fh, frame_num, class_id=None):
        enabled_cams = self.config.get("enabled_cameras", [])
        if source_id >= len(enabled_cams):
            return "UNKNOWN_SOURCE"
            
        cam_name = enabled_cams[source_id]
        
        if track_id not in self.vehicle_states:
            self.vehicle_states[track_id] = {
                "crops": [],
                "v_history": [],
                "frame_count": 0,
                "zone_history": [],
                "last_seen": 0,
                "source_id": source_id,
                "class_id": class_id,
                "status": "TRACKING",
                "max_persons_seen": 0,
                "best_img": None,
                "best_img_conf": -1.0,
                # Pass 2: Frame buffer for burst OCR + evidence video
                "frame_buffer": [],
                "bbox_list": [],
                "class_history": [],
                "area_history": []
            }
        
        state = self.vehicle_states[track_id]
        state["frame_count"] += 1
        state["last_seen"] = frame_num 
        
        rel_x, rel_y = (x + w/2) / fw, (y + h/2) / fh
        pos = Point(rel_x, rel_y)
        
        res = "TRACKING"
        if cam_name not in self.camera_map:
            return res
            
        cam_info = self.camera_map.get(cam_name, {})
        is_in_roi = False
        if cam_info:
            roi = cam_info["roi"]
            is_in_roi = roi.contains(pos)
            
            if is_in_roi:
                in_a = cam_info["zone_a"].contains(pos)
                in_b = cam_info["zone_b"].contains(pos)
                
                if in_a and "zone_a" not in state["zone_history"]:
                    state["zone_history"].append("zone_a")
                    logging.info(f"[Logic] Vehicle {track_id} entered Zone A on {cam_name}")
                if in_b and "zone_b" not in state["zone_history"]:
                    state["zone_history"].append("zone_b")
                    logging.info(f"[Logic] Vehicle {track_id} entered Zone B on {cam_name}")
                    
                if len(state["zone_history"]) >= 2:
                    if state["zone_history"][-2:] == ["zone_b", "zone_a"]:
                        if state["status"] != "VALID_ENTRY":
                            logging.info(f"🚦 [EVENT] Vehicle {track_id} VALID ENTRY detected on {cam_name}")
                        state["status"] = "VALID_ENTRY"
                    elif state["zone_history"][-2:] == ["zone_a", "zone_b"]:
                        if state["status"] != "WRONG_WAY":
                            logging.warning(f"🚨 [VIOLATION] Vehicle {track_id} WRONG WAY detected on {cam_name}")
                        state["status"] = "WRONG_WAY"
        
        return state["status"], is_in_roi

    def get_stale_tracks(self, current_frame, timeout=30):
        """Identifies vehicles that haven't been seen for 'timeout' frames."""
        stale_ids = []
        for tid, state in self.vehicle_states.items():
            if tid in self.finalized_ids: continue
            
            diff = current_frame - state["last_seen"]
            if diff > timeout:
                # Inclusive finalization: if they entered a zone, or we are in testing mode and have crops
                is_testing = self.config.get("application", {}).get("testing", False)
                if len(state["zone_history"]) > 0 or state["status"] != "TRACKING" or (is_testing and state.get("crops")):
                    stale_ids.append(tid)
                else:
                    # Silent cleanup for noise/ghosts
                    self.finalized_ids.add(tid)
                    
        return stale_ids

    def finalize(self, track_id):
        """Marks a track as finalized and returns its full data for OCR."""
        if track_id not in self.vehicle_states: return None
        state = self.vehicle_states[track_id]
        
        enabled_cams = self.config.get("enabled_cameras", [])
        cam_name = enabled_cams[state["source_id"]] if state["source_id"] < len(enabled_cams) else "Unknown"
        
        # Enhanced ITMS Logging
        logging.info(f"[Check] ID:{track_id} Track Finalized. Total Detections: {state['frame_count']}, Plate Images: {len(state.get('frame_buffer', []))}")
        
        self.finalized_ids.add(track_id)
        return state
