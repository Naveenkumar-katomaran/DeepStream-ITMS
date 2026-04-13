import requests
import json
import logging
import threading
from queue import Queue
from datetime import datetime
import time
import uuid
import cv2
import os
import subprocess
import mimetypes
from difflib import SequenceMatcher
from utils.minio_utils import MinioClient
from utils.rabbitmq_utils import RabbitMQProducer
from utils.logging_utils import log_context

class ApiSubmitter:
    def __init__(self, config):
        self.config = config
        self.queue = Queue(maxsize=100)
        
        outbound_cfg = self.config.get("outbound", {})
        
        self.rabbitmq_producer = None
        if "rabbitmq" in outbound_cfg:
            self.rabbitmq_producer = RabbitMQProducer(outbound_cfg["rabbitmq"])
            
        self.minio_client = None
        if "minio" in outbound_cfg:
            self.minio_client = MinioClient(outbound_cfg["minio"])
            
        self.seen_history = {}
        self.last_sent_plates = []
        logic_cfg = self.config.get("itms_logic", {})
        self.dedupe_cfg = logic_cfg.get("deduplication", {})
        
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        logging.info(f"[ApiSubmitter] Initialized.")

    def _is_duplicate(self, plate, confidence):
        if not self.dedupe_cfg.get("enabled", True): return False
        clean_plate = plate.strip().upper()
        if clean_plate == "UNKNOWN": return False
        current_time = time.time()
        cooldown = self.dedupe_cfg.get("cooldown_seconds", 300)
        threshold = self.dedupe_cfg.get("similarity_threshold", 0.85)
        self.seen_history = {p: v for p, v in self.seen_history.items() if current_time - v[0] < cooldown}
        for seen_plate, (last_time, last_conf) in self.seen_history.items():
            if SequenceMatcher(None, clean_plate, seen_plate).ratio() >= threshold: return True
        for last_plate in self.last_sent_plates:
            if SequenceMatcher(None, clean_plate, last_plate).ratio() >= threshold: return True
        self.seen_history[clean_plate] = (current_time, confidence)
        self.last_sent_plates.append(clean_plate)
        if len(self.last_sent_plates) > 2: self.last_sent_plates.pop(0)
        return False

    def submit_event(self, cam_name, plate_text, plate_conf, obj_id, vehicle_type="car", violations_data=None, image_data=None, plate_crop=None, detection_time=None, ocr_results=None, plate_meta=None):
        log_context.camera_name = cam_name
        if self._is_duplicate(plate_text, plate_conf):
            logging.info(f"Discarding duplicate plate: {plate_text}")
            return None

        det_dt = datetime.fromtimestamp(detection_time) if detection_time else datetime.now()
        date_str = det_dt.strftime("%Y%m%d")
        hex_id = hex(obj_id)[2:].upper().zfill(4)
        event_id = f"VA-ENT-{date_str}-{hex_id}"
        short_id = hex_id.lower()
        
        base_minio = self.config.get("outbound", {}).get("minio", {}).get("public_url", "https://itms-s3.katomaran.tech")
        bucket = self.config.get("outbound", {}).get("minio", {}).get("bucket", "traffic-api")
        date_path = det_dt.strftime("%Y/%m/%d")
        storage_prefix = f"{base_minio}/{bucket}/entries/{date_path}/{event_id}"
        
        cam_id_map = self.config.get("camera_id", {})

        payload = {
            "req_id": str(uuid.uuid4()),
            "event_id": event_id,
            "camera_id": cam_id_map.get(cam_name, cam_name),
            "vehicle_type": vehicle_type,
            "number_plate": plate_text,
            "detected_at": det_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "offline_entry": False,
            "plate_confidence": round(plate_conf, 2),
            "vehicle_image_url": f"{storage_prefix}/vehicle_{short_id}.jpg",
            "plate_image_url": f"{storage_prefix}/plate_{short_id}.jpg",
            "violations": []
        }
        
        for v in (violations_data or []):
            v_name = v.get("violation")
            v_uuid = self.config.get("violation_ids", {}).get(v_name, "00000000-0000-0000-0000-000000000000")
            payload["violations"].append({
                "violation": v_name,
                "violation_id": v_uuid,
                "confidence": round(v.get("confidence", 0.8), 2),
                "evidence_clip_url": f"{storage_prefix}/violations/event_{short_id}.mp4",
                "evidence_image_url": f"{storage_prefix}/vehicle_{short_id}.jpg"
            })

        if not self.queue.full():
            self.queue.put({
                "payload": payload, 
                "image": image_data, 
                "plate_crop": plate_crop,
                "obj_id": obj_id,
                "cam_name": cam_name,
                "detection_time": detection_time or time.time(),
                "ocr_results": ocr_results or [],
                "plate_meta": plate_meta or {}
            })
        
        return event_id

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None: break
            payload, image, plate_crop = item["payload"], item["image"], item.get("plate_crop")
            obj_id = item.get("obj_id", "??")
            cam_name = item.get("cam_name", "System")
            det_time = item.get("detection_time", time.time())
            event_id = payload["event_id"]
            short_id = event_id.split("-")[-1].lower()
            
            log_context.camera_name = cam_name
            det_dt = datetime.fromtimestamp(det_time)
            
            try:
                # 1. Local Storage (Harmonized with EvidenceWorker if debug enabled)
                debug_enabled = self.config.get("application", {}).get("testing", False)
                debug_root = self.config.get("application", {}).get("debug_folder", "debug")
                
                if debug_enabled:
                    status_folder = "violations" if len(payload.get("violations", [])) > 0 else "others"
                    local_dir = os.path.join(debug_root, "detections", status_folder, f"ID{obj_id}_{event_id}")
                    os.makedirs(local_dir, exist_ok=True)
                    img_filename = f"vehicle_{short_id}.jpg"
                    plate_img_filename = f"plate_{short_id}.jpg"
                    json_filename = "metadata.json"
                else:
                    date_folder = det_dt.strftime("%Y-%m-%d")
                    time_filename = det_dt.strftime("%H:%M:%S")
                    local_dir = os.path.join(cam_name, date_folder)
                    os.makedirs(local_dir, exist_ok=True)
                    img_filename = f"{time_filename}-{payload['number_plate']}-{short_id}.jpg"
                    plate_img_filename = f"{time_filename}-plate-{short_id}.jpg"
                    json_filename = f"{time_filename}-{payload['number_plate']}-{short_id}.json"
                
                img_path = os.path.join(local_dir, img_filename)
                plate_img_path = os.path.join(local_dir, plate_img_filename)
                json_path = os.path.join(local_dir, json_filename)

                # Save Plate Image Locally
                if plate_crop is not None and plate_crop.size > 0:
                    plate_bgr = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(plate_img_path, plate_bgr)
                
                # Save Vehicle Image Locally (if in debug)
                if debug_enabled and image is not None and image.size > 0:
                    cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                plate_meta = item.get("plate_meta", {})
                local_metadata = {
                    "height": plate_meta.get("height", 0),
                    "width": plate_meta.get("width", 0),
                    "time": det_dt.strftime("%Y-%m-%d %H:%M:%S.%f") + " +0530",
                    "lp_confs": plate_meta.get("all_confs", []),
                    "ocr_batch_results": item.get("ocr_results", [])
                }
                with open(json_path, "w") as f:
                    json.dump(local_metadata, f, indent=2)

                # 2. MinIO Upload
                if self.minio_client:
                    date_path = det_dt.strftime("%Y/%m/%d")
                    if image is not None and image.size > 0:
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        image_key = f"entries/{date_path}/{event_id}/vehicle_{short_id}.jpg"
                        success, buff = cv2.imencode(".jpg", image_bgr)
                        if success: self.minio_client.upload_bytes(buff.tobytes(), image_key)
                    
                    if plate_crop is not None and plate_crop.size > 0:
                        plate_bgr = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)
                        plate_key = f"entries/{date_path}/{event_id}/plate_{short_id}.jpg"
                        success, p_buff = cv2.imencode(".jpg", plate_bgr)
                        if success: self.minio_client.upload_bytes(p_buff.tobytes(), plate_key)

                    json_key = f"entries/{date_path}/{event_id}/metadata.json"
                    self.minio_client.upload_bytes(json.dumps(payload, indent=4).encode('utf-8'), json_key)

                # 3. RabbitMQ Publish
                if self.rabbitmq_producer: 
                    self.rabbitmq_producer.publish(payload)
                    logging.info(f"[ApiSubmitter] ID:{obj_id} Published successfully.")

                logging.info(f"[ApiSubmitter] ID:{obj_id} FINAL SUBMITTED DATA:\n{json.dumps(payload, indent=4)}")

            except Exception as e:
                logging.error(f"[ApiSubmitter] Error processing {event_id}: {e}")
            self.queue.task_done()
            time.sleep(0.05)


class EvidenceWorker:
    def __init__(self, config):
        self.config = config
        self.queue = Queue(maxsize=50)
        
        outbound_cfg = self.config.get("outbound", {})
        self.minio_client = None
        if "minio" in outbound_cfg:
            self.minio_client = MinioClient(outbound_cfg["minio"])
            
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logging.info(f"[EvidenceWorker] Initialized.")


    def add_task(self, cam_name, event_id, frame_buffer, plate_text, has_violation, detection_time, obj_id):
        if not self.queue.full():
            self.queue.put({
                "cam_name": cam_name,
                "event_id": event_id,
                "frame_buffer": frame_buffer,
                "plate_text": plate_text,
                "has_violation": has_violation,
                "detection_time": detection_time,
                "obj_id": obj_id
            })

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None: break
            
            cam_name = item["cam_name"]
            event_id = item["event_id"]
            frames = item["frame_buffer"]
            plate_text = item["plate_text"]
            has_violation = item["has_violation"]
            det_time = item["detection_time"]
            obj_id = item["obj_id"]
            
            log_context.camera_name = cam_name
            short_id = event_id.split("-")[-1].lower()
            det_dt = datetime.fromtimestamp(det_time)
            debug_enabled = self.config.get("application", {}).get("testing", False)

            if (has_violation or debug_enabled) and frames:
                try:
                    temp_dir = f"/tmp/itms_evidence/{event_id}"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    v_filename = f"event_{short_id}.mp4"
                    raw_path = os.path.join(temp_dir, v_filename)
                    web_v_filename = f"web_{v_filename}"
                    
                    # Grouped local storage for easier debugging
                    debug_root = self.config.get("application", {}).get("debug_folder", "debug")
                    status_folder = "violations" if has_violation else "others"
                    local_group_dir = os.path.join(debug_root, "detections", status_folder, f"ID{obj_id}_{event_id}")
                    if debug_enabled:
                        os.makedirs(local_group_dir, exist_ok=True)
                        final_path = os.path.join(local_group_dir, v_filename)
                    else:
                        final_path = os.path.join(temp_dir, web_v_filename)

                    # Step 1: Write raw frames (8.0 FPS)
                    h, w = frames[0].shape[:2]
                    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), 8.0, (w, h))
                    for frm in frames:
                        out.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
                    out.release()
                    
                    # Step 2: Convert to H.264
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", raw_path,
                        "-vcodec", "libx264", 
                        "-acodec", "aac",
                        "-pix_fmt", "yuv420p",
                        "-preset", "fast", 
                        "-crf", "23",
                        "-loglevel", "error", 
                        "-movflags", "+faststart",
                        final_path
                    ]
                    try:
                        subprocess.run(ffmpeg_cmd, check=True, timeout=30)
                        if os.path.exists(raw_path) and raw_path != final_path:
                            os.remove(raw_path)
                    except Exception as fe:
                        logging.warning(f"[EvidenceWorker] FFmpeg failed, using raw: {fe}")
                        if debug_enabled and raw_path != final_path:
                            try:
                                import shutil
                                shutil.copy(raw_path, final_path)
                                logging.info(f"[EvidenceWorker] Saved RAW clip to debug folder: {final_path}")
                            except Exception: pass
                        final_path = raw_path

                    # Step 3: Upload
                    if has_violation and self.minio_client and os.path.exists(final_path):
                        date_path_v = det_dt.strftime("%Y/%m/%d")
                        v_key = f"entries/{date_path_v}/{event_id}/violations/{v_filename}"
                        self.minio_client.upload_file(final_path, v_key, content_type="video/mp4")
                    
                    if not debug_enabled:
                        import shutil
                        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

                except Exception as e:
                    logging.error(f"[EvidenceWorker] Error for {event_id}: {e}")
            
            self.queue.task_done()
