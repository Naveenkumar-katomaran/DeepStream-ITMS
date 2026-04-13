import os
import cv2
import numpy as np
import logging
import threading
import time
from queue import Queue
from collections import Counter
from utils.ocr import consolidate_ocr_results
from utils.logging_utils import log_context

class OCRWorker:
    def __init__(self, config, api_submitter, evidence_submitter):
        self.config = config
        self.api_submitter = api_submitter
        self.evidence_submitter = evidence_submitter
        
        # Split Pipeline Initiation
        from batch_engine import BatchInferenceEngine
        self.batch_engine_car = BatchInferenceEngine(config, "car")
        self.batch_engine_bike = BatchInferenceEngine(config, "motorcycle")
        
        self.engines = {
            "car": self.batch_engine_car,
            "motorcycle": self.batch_engine_bike
        }
        
        p_cfg = self.config.get("pipeline", {})
        self.queue = Queue(maxsize=p_cfg.get("OCR_QUEUE_MAXSIZE", 50))
        self.chunk_size = p_cfg.get("ocr_chunk_size", 16)
        
        models_cfg = self.config.get("models", {})
        thresholds = models_cfg.get("conf_thresholds", {})
        
        self.secondary_threshold = thresholds.get("secondary", 0.4)
        self.ocr_threshold = thresholds.get("ocr", 0.4)
        self.labels = models_cfg.get("ocr_labels", "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
        self.min_plate_len = models_cfg.get("min_plate_length", 2)
        self.validate_indian = models_cfg.get("validate_indian_plate", False)

        # Consensus & Class Config
        self.secondary_meta = models_cfg.get("secondary_models", {})
        logic_cfg = self.config.get("itms_logic", {})
        self.consensus_threshold = logic_cfg.get("violation_consensus_threshold", 0.6)
        self.freq_threshold = logic_cfg.get("violation_frequency_threshold", 0.3)
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logging.info("[OCR Worker] Initialized as Metadata Consolidator (Hardware-Offloaded).")

    def add_to_batch(self, camera_name, vehicle_type, image_list, obj_id, full_image=None, violation_history=None, total_frames=1, frame_buffer=None, detection_time=None, bbox_list=None):
        if self.queue.full():
            logging.warning("[OCR Worker] Queue full, dropping event batch.")
            return
        self.queue.put({
            "cam": camera_name,
            "type": vehicle_type,
            "images": image_list,
            "id": obj_id,
            "full_image": full_image,
            "violations": violation_history or [],
            "total_frames": max(1, total_frames),
            "frame_buffer": frame_buffer,
            "detection_time": detection_time or time.time(),
            "bbox_list": bbox_list
        })

    def _worker_loop(self):
        while True:
            item = self.queue.get()
            if item is None: break
            
            log_context.camera_name = item.get("cam", "System")
            
            try:
                # 1. Hardware Batch Preparation (Scout Mode Extraction)
                obj_id = item["id"]
                det_time = item.get("detection_time", time.time())
                
                v_counts = Counter()
                v_conf_sums = Counter()
                voter_input = []
                
                # 1. Resolve Dominant Vehicle Type (Voting)
                class_history = item.get("type", [])  # This is the list from handoff
                if isinstance(class_history, list) and len(class_history) > 0:
                    most_common = Counter(class_history).most_common(1)[0][0]
                    v_type = "motorcycle" if most_common in [1, 3] else "car"
                else:
                    v_type = "car"
                
                # Use the best available wide crop from Pass 1 for the 'Vehicle Image'
                best_vehicle_crop = item["images"][-1] if item.get("images") else item.get("full_image")
                best_plate_crop = None
                total_samples = 1
                
                target_engine = self.engines.get(v_type)
                
                # --- STAGE 2: ASYNCHRONOUS BURST INFERENCE (Pass 2) ---
                if target_engine and item.get("frame_buffer") and item.get("bbox_list"):

                    # 2. Extract Specialized Ghost Crops (Pass 2 Logic)
                    ocr_pad = self.config["pipeline"].get("ocr_padding", 100)
                    numpy_crops = target_engine.generate_ghost_crops(
                        item["frame_buffer"], 
                        item["bbox_list"], 
                        padding=ocr_pad
                    )
                    
                    # Memory Cleanup: Drop raw frames immediately after cropping
                    
                    if numpy_crops:
                        # Burst Inference (Violations + Plates + OCR)
                        burst_results = target_engine.process_batch(numpy_crops)
                        total_samples = len(burst_results)
                        max_chars_found = -1
                        all_plate_confs = []
                        max_plate_conf = -1.0
                        best_plate_img = None
                        best_vehicle_img = None
                        best_scene_img = None
                        
                        # --- DEBUG SETUP ---
                        debug_enabled = self.config["application"].get("testing", False)
                        if debug_enabled:
                            base_debug = f"/app/debug/vehicle_{obj_id}"
                            inf_dir = "bike_inference" if v_type == "motorcycle" else "car_inference"
                            os.makedirs(f"{base_debug}/{inf_dir}", exist_ok=True)
                            os.makedirs(f"{base_debug}/vehicle_crops", exist_ok=True)
                            os.makedirs(f"{base_debug}/plate_crops", exist_ok=True)
                            os.makedirs(f"{base_debug}/violations", exist_ok=True)
                            
                             # Save all Source Crops from Pass 1 (if any)
                            imgs_to_save = item.get("images") or []
                            for i, sc in enumerate(imgs_to_save):
                                cv2.imwrite(f"{base_debug}/vehicle_crops/pass1_{i}.jpg", cv2.cvtColor(sc, cv2.COLOR_RGB2BGR))

                        for i, res in enumerate(burst_results):
                            # Crop to draw on (RGB to match numpy_crops)
                            draw_crop = numpy_crops[i].copy()
                            
                            # OCR Characters (Verification Mode)
                            chars = res.get("characters", [])
                            plate_str = "UNKNOWN"
                            
                            # Plate Legitimacy Filter: 
                            # Only accept a "Plate" if it has at least 2 valid characters (eliminates hoods/noise)
                            is_legit_plate = len(chars) >= 2
                            
                            if is_legit_plate:
                                chars.sort(key=lambda x: x.get("x", 0))
                                plate_str = "".join([c["char"].replace("char_", "") for c in chars])
                                # Calculate real OCR confidence: average of all character confidences
                                # Key is "conf" (from batch_engine.py line 241), NOT "confidence"
                                char_confs = [c.get("conf", 0.0) for c in chars]
                                avg_char_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
                                voter_input.append((plate_str, round(avg_char_conf, 4)))
                                
                                # Update Best Plate Image if this one has more characters
                                if len(chars) > max_chars_found:
                                    max_chars_found = len(chars)
                                    px1 = max(0, min([c["x"] for c in chars]) - 10)
                                    py1 = max(0, min([c["y"] for c in chars]) - 10)
                                    px2 = min(640, max([c["x"] + c["w"] for c in chars]) + 10)
                                    py2 = min(640, max([c["y"] + c["h"] for c in chars]) + 10)
                                    crop = numpy_crops[i][int(py1):int(py2), int(px1):int(px2)].copy()
                                    if crop.size > 0:
                                        best_plate_crop = crop
                            
                            # Best Plate Selection Logic: HIGH CONFIDENCE BASED (Clean Extraction)
                            all_plates = res.get("plates", [])
                            for p_item in all_plates:
                                px, py, pw, ph = [int(c) for c in p_item[:4]]
                                p_conf = p_item[4] if len(p_item) > 4 else 0.9
                                all_plate_confs.append(float(p_conf))
                                
                                if p_conf > max_plate_conf:
                                    max_plate_conf = p_conf
                                    # Extract clean tight crop from the original raw numpy_crop
                                    if pw > 0 and ph > 0:
                                        crop = numpy_crops[i][py:py+ph, px:px+pw].copy()
                                        if crop.size > 0:
                                            best_plate_img = crop
                                            # Capture the clean wide vehicle crop (Primary Evidence)
                                            best_vehicle_img = numpy_crops[i].copy()
                                            # Also capture the corresponding clean full scene (Audit Context)
                                            best_scene_img = item["frame_buffer"][i].copy()

                            largest_plate = max(all_plates, key=lambda p: p[2] * p[3]) if all_plates else None

                            # --- VIOLATION & VISUAL OVERLAY ---
                            # Violations (Class-Specific Enforcement)
                            raw_violations = res.get("violations", [])
                            heads_in_frame = 0
                            
                            # Audit Mode: Show ALL detected plates for diagnostic tracking
                            for p_item in all_plates:
                                px, py, pw, ph = [int(c) for c in p_item[:4]]
                                p_conf = p_item[4] if len(p_item) > 4 else 0.9
                                p_area = pw * ph
                                color = (255, 255, 0) # Yellow for all plates
                                is_main = (p_item == largest_plate)
                                label_text = f"plate {p_conf:.2f} [{p_area}px]"
                                
                                if is_legit_plate and is_main:
                                    cv2.rectangle(draw_crop, (px, py), (px + pw, py + ph), color, 3)
                                    cv2.putText(draw_crop, f"PLATE (MAIN) {p_conf:.2f} [{p_area}px]", (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                elif debug_enabled:
                                    cv2.rectangle(draw_crop, (px, py), (px + pw, py + ph), (100, 100, 0), 1)
                                    cv2.putText(draw_crop, label_text, (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 0), 1)
                            
                            for v_item in raw_violations:
                                v_label = v_item["label"]
                                v_bbox = v_item["bbox"]
                                vx, vy, vw, vh = [int(c) for c in v_bbox]
                                
                                v_conf = v_item.get("conf", 0.9)
                                is_valid = False
                                
                                # Category 1: Identifying Info (Make/Logo) - Always Draw in Cyan
                                if v_label == "make":
                                    cv2.rectangle(draw_crop, (vx, vy), (vx + vw, vy + vh), (255, 255, 0), 2) # Cyan BGR: (255,255,0)
                                    cv2.putText(draw_crop, f"Make {v_conf:.2f}", (vx, vy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    continue

                                if v_type == "motorcycle":
                                    if v_label in ["helmet", "no_helmet"]:
                                        # Spatial Cluster Filter: Only count heads in the logical Bike-Zone 
                                        in_head_zone = (vy < 450) and (100 < vx < 540)
                                        if in_head_zone:
                                            is_valid = True
                                            heads_in_frame += 1
                                        else:
                                            if debug_enabled:
                                                cv2.rectangle(draw_crop, (vx, vy), (vx + vw, vy + vh), (100, 0, 100), 1)
                                    elif v_label == "mobile_phone_usage":
                                        is_valid = True
                                else:
                                    if v_label in ["no_seatbelt", "mobile_phone_usage", "drinking"]:
                                        is_valid = True
                                
                                if is_valid:
                                    v_counts[v_label] += 1
                                    v_conf_sums[v_label] += v_conf
                                    
                                    # Map colors: Red for violations, Green for compliance
                                    color = (0, 0, 255) # BGR: Red
                                    if v_label in ["helmet", "seatbelt"]:
                                        color = (0, 255, 0) # BGR: Green
                                        
                                    cv2.rectangle(draw_crop, (vx, vy), (vx + vw, vy + vh), color, 2)
                                    cv2.putText(draw_crop, v_label, (vx, vy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    
                                    # Save Violation Evidence
                                    if debug_enabled:
                                        cv2.imwrite(f"{base_debug}/violations/{v_label}_{i}.jpg", cv2.cvtColor(draw_crop, cv2.COLOR_RGB2BGR))

                            # Triple Riding Detection (Heuristic: 3+ People Detected)
                            if v_type == "motorcycle" and heads_in_frame >= 3:
                                v_counts["triples"] += 1
                                v_conf_sums["triples"] += 0.9
                                if debug_enabled:
                                    cv2.putText(draw_crop, "TRIPLES", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                    cv2.imwrite(f"{base_debug}/violations/triples_{i}.jpg", cv2.cvtColor(draw_crop, cv2.COLOR_RGB2BGR))

                            # DEBUG SAVE (Comprehensive Harvesting with Overlays)
                            if debug_enabled:
                                # Save the raw crop with overlays
                                cv2.imwrite(f"{base_debug}/{inf_dir}/crop_{i}.jpg", cv2.cvtColor(draw_crop, cv2.COLOR_RGB2BGR))
                                # Save the tight plate crop if we had one
                                if best_plate_crop is not None:
                                    cv2.imwrite(f"{base_debug}/plate_crops/plate_{i}.jpg", cv2.cvtColor(best_plate_crop, cv2.COLOR_RGB2BGR))
                    
                    # --- EVIDENCE GENERATION (CLEAN / UNANNOTATED) ---
                    # Use the high-confidence clean images identified during the burst
                    # Fallback to latest available if no plate confidence peak was found
                    vehicle_img = best_vehicle_img if best_vehicle_img is not None else numpy_crops[-1].copy()
                    plate_img = best_plate_img if best_plate_img is not None else numpy_crops[-1].copy()

                    # Cleanup the large batch of OCR crops now that we've harvested the best ones
                    del numpy_crops

                    # Diagnostic Drawing: ONLY for debug directory evidence, NOT for final payload
                    if debug_enabled:
                        # Draw on context copy (Full scene for developer auditing)
                        audit_frame = (best_scene_img if best_scene_img is not None else item["frame_buffer"][-1]).copy()
                        (bx1, by1), (bx2, by2) = item["bbox_list"][-1]
                        color = (0, 0, 255) if v_type == "motorcycle" else (0, 255, 0)
                        cv2.rectangle(audit_frame, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 4)
                        cv2.putText(audit_frame, f"{v_type.upper()} ID:{obj_id}", (int(bx1), int(by1)-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                        cv2.imwrite(f"{base_debug}/vehicle_crops/evidence_context.jpg", cv2.cvtColor(audit_frame, cv2.COLOR_RGB2BGR))
                        
                        # Save the actual clean harvest used in the payload to debug
                        cv2.imwrite(f"{base_debug}/vehicle_crops/payload_vehicle.jpg", cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{base_debug}/plate_crops/payload_plate.jpg", cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR))
                    
                    # Capture Plate Dimensions for Meta
                    ph, pw = plate_img.shape[:2] if plate_img is not None else (0, 0)
                    plate_meta = {"height": ph, "width": pw, "all_confs": all_plate_confs}
                    
                else:
                    logging.warning(f"[OCR Worker] ID:{obj_id} No batch engine or images provided. Skipping Pass 2.")
                    vehicle_img = best_vehicle_crop
                    plate_img = best_plate_crop

                # --- STAGE 4: OCR CONSOLIDATION ---
                final_plate = "UNKNOWN"
                if voter_input:
                    # UPDATED: consolidate_ocr_results now returns [text, summary, conf], status
                    [final_plate, group_summary, ocr_conf], status = consolidate_ocr_results(
                        voter_input, validate_indian_plate=self.validate_indian
                    )
                    # Use OCR consensus confidence as the primary metric
                    final_plate_confidence = ocr_conf
                else:
                    group_summary = []
                    final_plate_confidence = 0.0
                
                if final_plate is None:
                    final_plate = "UNKNOWN"
                
                # Report if we have a plate or just need to submit the event regardless
                if final_plate:
                    confirmed_violations = []
                    
                    # --- STAGE 5: VIOLATION CONSENSUS ---
                    # v_counts and v_conf_sums are populated during the burst loop below if violations are detected.
                    
                    sub_meta = self.secondary_meta.get(v_type, {})
                    reportable = sub_meta.get("reportable", [])
                    
                    disabled_cfg = self.config.get("application", {}).get("disabled_violations", {})
                    global_disabled = disabled_cfg.get("all_camera", [])
                    camera_disabled = disabled_cfg.get(item["cam"], [])

                    for v_type_name, count in v_counts.items():
                        if v_type_name not in reportable: continue
                        frequency = count / total_samples
                        
                        if frequency >= self.freq_threshold:
                            if v_type_name in global_disabled or v_type_name in camera_disabled:
                                continue

                            avg_conf = v_conf_sums[v_type_name] / count
                            confirmed_violations.append({
                                "violation": v_type_name,
                                "confidence": round(avg_conf, 2)
                            })
                            logging.info(f"[Consensus] ID:{obj_id} CONFIRMED {v_type_name} ({count}/{total_samples} frames)")

                    if self.api_submitter:
                        # 1. Start API/Image Submission (Fast)
                        event_id = self.api_submitter.submit_event(
                            item["cam"], final_plate, final_plate_confidence, obj_id, 
                            vehicle_type=v_type,
                            violations_data=confirmed_violations, 
                            image_data=vehicle_img,
                            plate_crop=plate_img,
                            detection_time=det_time,
                            ocr_results=group_summary,
                            plate_meta=plate_meta
                        )
                        
                        # 2. Trigger Video Evidence Synthesis (Background)
                        if event_id and self.evidence_submitter:
                            has_violation = len(confirmed_violations) > 0
                            self.evidence_submitter.add_task(
                                item["cam"], event_id, item.get("frame_buffer", []), 
                                final_plate, has_violation, det_time, obj_id
                            )
                        
                        # Memory Cleanup: Drop OCR worker's reference to the large video buffer
                        if "frame_buffer" in item:
                            item["frame_buffer"] = None
                
            except Exception as e:
                logging.error(f"[OCR Worker] Loop Error: {e}", exc_info=True)
            
            self.queue.task_done()
