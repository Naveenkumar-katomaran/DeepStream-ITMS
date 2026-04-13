import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import numpy as np
import threading
import queue
import logging
import time
import cv2
from utils.interpolation import interpolate_bboxes

class BatchInferenceEngine:
    """
    A standalone GStreamer pipeline for asynchronous burst inference.
    Accepts batches of images via appsrc and processes them through SGIEs.
    """
    def __init__(self, config, model_type="car"):
        self.config = config
        self.model_type = model_type
        self.pipeline = None
        self.appsrc = None
        self.appsink = None
        self.loop = None
        self.thread = None
        self.results_queue = queue.Queue()
        self.batch_metadata = {}
        self.processed_count = 0 
        
        self._init_pipeline()

    def _init_pipeline(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("batch-inference-pipeline")
        
        # Diagnostics
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        
        # Elements
        self.appsrc = Gst.ElementFactory.make("appsrc", f"source-{self.model_type}")
        vconv_bridge = Gst.ElementFactory.make("videoconvert", f"vconv-bridge-{self.model_type}")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv-{self.model_type}")
        caps_appsrc = Gst.ElementFactory.make("capsfilter", f"caps-appsrc-{self.model_type}")
        
        # Streammux is required to form batches for nvinferserver
        streammux = Gst.ElementFactory.make("nvstreammux", f"streammux-{self.model_type}")
        
        # SGIE Dynamic Loading
        sgie_primary = Gst.ElementFactory.make("nvinferserver", f"sgie-primary-{self.model_type}")
        sgie_ocr = Gst.ElementFactory.make("nvinferserver", f"sgie-ocr-{self.model_type}")
        
        # FIX: Added a queue as a shock absorber for the 50-frame burst
        self.queue = Gst.ElementFactory.make("queue", f"queue-{self.model_type}")
        
        self.appsink = Gst.ElementFactory.make("appsink", f"sink-{self.model_type}")
        
        if not all([self.appsrc, vconv_bridge, nvvidconv, caps_appsrc, streammux, sgie_primary, sgie_ocr, self.appsink]):
            logging.error(f"[BatchEngine {self.model_type}] Failed to create elements")
            return

        # Config
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", False)
        
        # FIX: Explicit AppSrc Caps to prevent "not-negotiated" error
        src_caps = Gst.Caps.from_string("video/x-raw, format=RGB, width=640, height=640, framerate=0/1")
        self.appsrc.set_property("caps", src_caps)
        self.appsrc.set_property("max-bytes", 200000000) # 200MB buffer for large bursts
        self.appsrc.set_property("block", False)
        
        # Trace Step 1: Source Probe
        self.appsrc.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._trace_probe, "STEP 1: Source Out")
        
        # dGPU Memory Alignment: CPU -> Bridge -> GPU(NVMM)
        # nvstreammux on dGPU (especially DS 9.0) prefers RGBA for its input buffers.
        # nvinfer will handle the RGB conversion internally if needed.
        caps_appsrc.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=640, height=640"))
        
        # Streammux config - Force Batch-1 to match Triton model constraints (Stable)
        mem_type = self.config.get("pipeline", {}).get("nvbuf_memory_type", 3)
        streammux.set_property("width", 640)
        streammux.set_property("height", 640)
        streammux.set_property("batch-size", 1) 
        streammux.set_property("batched-push-timeout", -1) # Send as soon as 1 buffer arrives
        streammux.set_property("nvbuf-memory-type", mem_type)
        
        # Queue config
        self.queue.set_property("max-size-buffers", 100)
        self.queue.set_property("max-size-bytes", 0)
        self.queue.set_property("max-size-time", 0)
        
        # SGIE config loading dynamically based on model parameter
        if self.model_type == "motorcycle":
            sgie_primary.set_property("config-file-path", "configs/sgie_bike.txt")
        else:
            sgie_primary.set_property("config-file-path", "configs/sgie_car.txt")
            
        sgie_ocr.set_property("config-file-path", "configs/sgie_ocr.txt")
        
        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", self._on_new_sample)

        # Add & Link
        for el in [self.appsrc, vconv_bridge, nvvidconv, caps_appsrc, self.queue, streammux, sgie_primary, sgie_ocr, self.appsink]:
            self.pipeline.add(el)
            
        # appsrc -> bridge -> nvvidconv -> caps -> queue -> streammux
        self.appsrc.link(vconv_bridge)
        vconv_bridge.link(nvvidconv)
        nvvidconv.link(caps_appsrc)
        caps_appsrc.link(self.queue)
        
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad = self.queue.get_static_pad("src")
        
        # Trace Step 3: Muxer Input Probe
        sinkpad.add_probe(Gst.PadProbeType.BUFFER, self._trace_probe, "STEP 2: Muxer In")
        
        srcpad.link(sinkpad)
        
        streammux.link(sgie_primary)
        sgie_primary.link(sgie_ocr)
        sgie_ocr.link(self.appsink)

        # Start Loop
        self.loop = GLib.MainLoop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        self.pipeline.set_state(Gst.State.PLAYING)
        logging.info("[BatchEngine] Native Batch Pipeline (Ultimate Stabilized) Initialized.")

    def _trace_probe(self, pad, info, step_name):
        logging.info(f"[Batch-Trace] {step_name}")
        return Gst.PadProbeReturn.OK

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logging.error(f"[BatchEngine Pipeline ERROR] {message.src.get_name()}: {err}")
            logging.error(f"[BatchEngine Pipeline DEBUG] {debug}")
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logging.warning(f"[BatchEngine Pipeline WARNING] {message.src.get_name()}: {err}")
        elif t == Gst.MessageType.EOS:
            logging.info("[BatchEngine] Pipeline reached EOS")
        return True

    def _run_loop(self):
        try:
            self.loop.run()
        except Exception as e:
            logging.error(f"[BatchEngine] Loop Error: {e}")

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list
        
        results = []
        while l_frame:
            # DETERMINISTIC SYNC: Use a manual counter because DeepStream may reset 
            # frame_num to 0 when batch-size is 1.
            idx = self.processed_count
            self.processed_count += 1
            
            # Get transformation metadata (scale, offset_x, offset_y)
            r, ox, oy = self.batch_metadata.get(idx, (1.0, 0, 0))
            
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_results = {"violations": [], "characters": []}
            
            # Helper to map coordinates back to pre-letterbox space
            def restore(left, top, width, height):
                return [
                    int((left - ox) / r),
                    int((top - oy) / r),
                    int(width / r),
                    int(height / r)
                ]

            # 1. Check Frame Level User Meta
            if hasattr(frame_meta, "frame_user_meta_list"):
                l_user = frame_meta.frame_user_meta_list
                while l_user:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    l_user = l_user.next

            # 2. Check Object Meta (Standard for Detector/Classifier SGIEs)
            l_obj = frame_meta.obj_meta_list if hasattr(frame_meta, "obj_meta_list") else None
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                # Violations (Classifiers)
                if hasattr(obj_meta, "classifier_meta_list"):
                    l_class = obj_meta.classifier_meta_list
                    while l_class:
                        c_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
                        l_label = c_meta.label_info_list
                        while l_label:
                            label = pyds.NvDsLabelInfo.cast(l_label.data)
                            rx, ry, rw, rh = restore(obj_meta.rect_params.left, obj_meta.rect_params.top, 
                                                     obj_meta.rect_params.width, obj_meta.rect_params.height)
                            frame_results["violations"].append({
                                "label": label.result_label,
                                "bbox": [rx, ry, rw, rh],
                                "conf": obj_meta.confidence
                            })
                            l_label = l_label.next
                        l_class = l_class.next
                
                # Check for Plates & Violations directly attached as YOLO objects
                if obj_meta.obj_label and obj_meta.unique_component_id != 4:
                    rx, ry, rw, rh = restore(obj_meta.rect_params.left, obj_meta.rect_params.top, 
                                             obj_meta.rect_params.width, obj_meta.rect_params.height)
                    
                    if obj_meta.obj_label == "plate":
                        if "plates" not in frame_results:
                            frame_results["plates"] = []
                        
                        frame_results["plate_detected"] = True
                        frame_results["plates"].append([rx, ry, rw, rh, obj_meta.confidence])
                    else:
                        frame_results["violations"].append({
                            "label": obj_meta.obj_label,
                            "bbox": [rx, ry, rw, rh],
                            "conf": obj_meta.confidence
                        })

                # OCR (Objects/Labels from SGIE 4)
                if obj_meta.obj_label and obj_meta.unique_component_id == 4:
                    rx, ry, rw, rh = restore(obj_meta.rect_params.left, obj_meta.rect_params.top, 
                                             obj_meta.rect_params.width, obj_meta.rect_params.height)
                    frame_results["characters"].append({
                        "char": obj_meta.obj_label,
                        "conf": obj_meta.confidence,
                        "x": rx, "y": ry, "w": rw, "h": rh
                    })
                l_obj = l_obj.next
            
            results.append(frame_results)
            l_frame = l_frame.next
            
        for res_list in results:
            self.results_queue.put(res_list)
        return Gst.FlowReturn.OK

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Ultralytics-style mathematical letterboxing. Returns (img, scale, offset_x, offset_y)"""
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, (r, left, top)

    def process_batch(self, images):
        """
        Pushes a list of images (numpy arrays) through the pipeline.
        Returns aggregated results.
        """
        if not images: return []
        
        # Clear queue for fresh batch
        while not self.results_queue.empty():
            try: self.results_queue.get_nowait()
            except: pass

        self.batch_metadata = {} 
        self.processed_count = 0 
        start_time = time.time()
        for i, img in enumerate(images):
            # STANDARD RESOLUTION: Ultralytics Letterboxing (Zero-Warp)
            meta = (1.0, 0, 0) # Default: No change
            if img.shape[0] != 640 or img.shape[1] != 640:
                img, meta = self._letterbox(img, new_shape=(640, 640))
            
            self.batch_metadata[i] = meta # Store for coordinate restoration

            # Ensure RGB to match IMAGE_FORMAT_RGB in sgie_*.txt
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] == 3:
                # Assuming incoming is BGR from OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            data = img.tobytes()
            # FIX: Memory Leak Prevention
            # Gst.Buffer.new_wrapped leaks python references in PyGObject. 
            # We must explicitly allocate and fill the memory buffer.
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            
            # Metadata for appsrc (timing)
            buf.pts = int(i * (Gst.SECOND / 30))
            buf.duration = int(Gst.SECOND / 30)
            
            # Safety: Preventing Buffer Overflow during massive bursts
            time.sleep(0.01) 
            self.appsrc.emit("push-buffer", buf)
        
        # Wait for all results
        all_res = []
        timeout_at = time.time() + 30.0 # 30s timeout for massive 50-frame bursts
        
        while len(all_res) < len(images) and time.time() < timeout_at:
            try:
                res_item = self.results_queue.get(timeout=0.2)
                all_res.append(res_item)
            except queue.Empty:
                continue
        
        if len(all_res) < len(images):
            logging.error(f"[BatchEngine] Timed out. Collected {len(all_res)}/{len(images)} results.")
        else:
            logging.info(f"[BatchEngine] Hardware processed {len(all_res)} images in {time.time()-start_time:.2f}s")
        return all_res

    def generate_ghost_crops(self, frame_buffer, bbox_list, padding=30):
        """
        Extracts high-resolution crops directly from the best-frame buffer.
        Interpolation is now removed as we accumulate frames natively.
        """
        if not frame_buffer or not bbox_list:
            return []
            
        crops = []
        # Total frames to process (Matches user's accumulation limit)
        for i in range(len(frame_buffer)):
            frame = frame_buffer[i]
            bbox = bbox_list[i]
            
            if frame is None: continue
            
            fh, fw = frame.shape[:2]
            (x1, y1), (x2, y2) = bbox
            
            # Apply Padding (The 9apr Context Fix)
            px1 = max(0, int(x1) - padding)
            py1 = max(0, int(y1) - padding)
            px2 = min(fw, int(x2) + padding)
            py2 = min(fh, int(y2) + padding)
            
            if px2 > px1 and py2 > py1:
                crop = frame[py1:py2, px1:px2].copy()
                crops.append(crop)
                
        # Sustainable Limit (Matches config)
        max_limit = self.config.get("pipeline", {}).get("track_accumulation_limit", 50)
        if len(crops) > max_limit:
            # Thinning logic only if buffer exceeds configuration
            idx = np.round(np.linspace(0, len(crops) - 1, max_limit)).astype(int)
            crops = [crops[i] for i in idx]
            
        return crops

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()
