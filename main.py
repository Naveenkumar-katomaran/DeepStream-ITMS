import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import numpy as np
import cv2
import json
import time
from collections import deque
import logging
import threading
from logic import LogicEngine
from ocr_worker import OCRWorker
from submitter import ApiSubmitter, EvidenceWorker
from utils.logging_utils import setup_itms_logging, log_context, log_rotation_worker, cleanup_old_logs

# Global Engines
logic_engine = None
api_submitter = None
evidence_worker = None
ocr_worker = None
source_id_map = {} # {id: name}
source_bin_map = {} # {name: bin}
source_fail_counts = {} # {name: int}
source_resolutions = {} # {name: (w, h)}
batch_engine = None
ALLOWED_CLASSES = [2, 3, 5, 7] 
# Global counter for periodic cleanup
probe_frame_count = 0 
pipeline_ptr = None
def bus_call(bus, message, user_data):
    loop, pipeline = user_data
    t = message.type
    if t == Gst.MessageType.EOS:
        # We handle per-source looping via probes now, so global EOS means all is done.
        # However, since we drop EOS in probes, this global EOS may only trigger on manual shutdown.
        logging.info("🏁 [Pipeline] Global End-of-stream reached.")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logging.warning(f"⚠️ [GStreamer Warning] {message.src.get_name()}: {err} | Debug: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        src_name = message.src.get_name()
        logging.error(f"❌ [GStreamer ERROR] {src_name}: {err}")
        
        # Identify if error is from a specific source bin
        found_cam = None
        for cam_name, s_bin in source_bin_map.items():
            if s_bin.get_name() in src_name or (message.src.get_parent() and s_bin.get_name() in message.src.get_parent().get_name()):
                found_cam = cam_name
                break
        
        if found_cam:
            source_fail_counts[found_cam] += 1
            f_count = source_fail_counts[found_cam]
            logging.warning(f"🔄 [Recovery] Source '{found_cam}' failed ({f_count}/5). Attempting independent restart...")
            
            if f_count <= 5:
                # Independent source restart logic: toggle state
                source_bin_map[found_cam].set_state(Gst.State.NULL)
                GLib.timeout_add_seconds(2, source_bin_map[found_cam].set_state, Gst.State.PLAYING)
            else:
                logging.error(f"💀 [Recovery] Source '{found_cam}' reached max retries. System will continue with others.")
                # We don't quit the loop, just keep running for remaining cameras
        elif "renderer" in src_name or "egl" in src_name:
            logging.critical("‼️ Visual Dashboard failed, but attempting to maintain Logic Branch...")
            message.src.set_state(Gst.State.NULL)
        else:
            logging.critical("💀 Fatal Pipeline Error. Shutting down.")
            loop.quit()
    elif t == Gst.MessageType.STATE_CHANGED:
        # Check if the message source is the pipeline (to avoid logging every element change)
        if isinstance(message.src, Gst.Pipeline):
            old_state, new_state, pending_state = message.parse_state_changed()
            logging.info(f"🔄 [Pipeline] State changed from {old_state.value_nick.upper()} to {new_state.value_nick.upper()}")
    return True

def source_eos_probe(pad, info, u_data):
    """
    Independent Looping Probe:
    Catches EOS events at the source level, performs a seek-to-start,
    and drops the event so downstream (muxer/pgie) never stops.
    """
    event = info.get_event()
    if event and event.type == Gst.EventType.EOS:
        cam_name, bin_element = u_data
        logging.info(f"🏁 [Looping] Source '{cam_name}' reached end of file. Restarting...")
        # Perform a flushing seek to jump back to time 0
        bin_element.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0)
        return Gst.PadProbeReturn.DROP
    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not produce NVMM caps \n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if name.find("rtspsrc") != -1:
        Object.set_property("protocols", 0x00000004) # TCP
        Object.set_property("latency", 500) # ms
        Object.set_property("tcp-timeout", 5000000) # 5s in microseconds
        Object.set_property("drop-on-latency", True)
    # Force hardware decoding where possible
    if name.find("nvv4l2decoder") != -1:
        try:
            Object.set_property("drop-frame-interval", 0)
            Object.set_property("num-extra-surfaces", 5)
        except Exception:
            pass

def create_source_bin(index, uri):
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    return nbin

def class_filter_probe(pad, info, u_data):
    """
    Metadata filter probe: Removes objects not in the ALLOWED_CLASSES list
    BEFORE they reach the tracker. This stops tracking/detecting persons.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            # SAFETY: Advance iterator BEFORE potentially removing the current object
            curr_l_obj = l_obj
            l_obj = l_obj.next
            
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(curr_l_obj.data)
            except StopIteration:
                break
            
            if obj_meta.class_id not in ALLOWED_CLASSES:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
        
        l_frame = l_frame.next
        
    return Gst.PadProbeReturn.OK

def pgie_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
            
        sid = frame_meta.source_id
        cam_name = source_id_map.get(sid, f"cam_{sid}")
        log_context.camera_name = cam_name

        try:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            fh_buf, fw_buf = n_frame.shape[:2]
            fh_src, fw_src = frame_meta.source_frame_height, frame_meta.source_frame_width
            
            # Metadata coordinates are automatically scaled by DeepStream to match the buffer resolution. 
            # Manual scaling is not required if the buffer is matched to the Meta's current frame.

            if frame_meta.frame_num % 1000 == 0:
                logging.info(f"[Calibrate] Cam:{cam_name} | Buffer:{fw_buf}x{fh_buf} | Source:{fw_src}x{fh_src}")
            
            # Resolution Recovery check
            last_res = source_resolutions.get(cam_name)
            if last_res and (fw_src != last_res[0] or fh_src != last_res[1]):
                logging.warning(f"🔄 [Recovery] Resolution changed for {cam_name}: {last_res} -> ({fw_src}, {fh_src})")
                if logic_engine:
                    logic_engine.update_resolution(cam_name, fw_src, fh_src)
            source_resolutions[cam_name] = (fw_src, fh_src)
        except Exception as e:
            logging.error(f"[Probe] Error mapping buffer: {str(e)}")
            continue

        # 0. Global Scout-Mode Crop Collection (Optimize RAM to prevent OOM)
        # We only want to encode the full 1080p frame ONCE per video frame, not per object.
        pass

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            # Primary Detection (Relative to Buffer Resolution, e.g., 1080p)
            rect = obj_meta.rect_params
            x, y, w, h = int(rect.left), int(rect.top), int(rect.width), int(rect.height)
            
            if logic_engine:
                res, is_in_roi = logic_engine.check_polygons(
                    obj_meta.object_id, sid, x, y, w, h, 
                    frame_meta.source_frame_width, frame_meta.source_frame_height,
                    frame_meta.frame_num, obj_meta.class_id
                )
                
                # Diagnostic log for every 100 frames to prove processing is active
                if frame_meta.frame_num % 100 == 0:
                    logging.info(f"[Detector] Processing {cam_name} | Obj:{obj_meta.object_id} | Class:{obj_meta.class_id} | Res:{res} | InROI:{is_in_roi}")
                
                if is_in_roi:
                    state = logic_engine.vehicle_states.get(obj_meta.object_id)
                    if state:
                        # 1. Best Snapshot Selection (Confidence-Based)
                        if obj_meta.confidence > state["best_img_conf"]:
                            state["best_img"] = n_frame[:, :, :3].copy()
                            state["best_img_conf"] = obj_meta.confidence

                        # 2. Accumulate frames for Pass 2 (Multi-frame OCR stabilization)
                        # We implement an "Area-Prioritized" buffer to always keep the best/closest shots.
                        
                        # Boundary Filter: Ignore frames where the vehicle is too close to edges (vulnerable to cutoff)
                        if y > 50 and (y + h) < 1000:
                            curr_area = w * h
                            max_acc = config.get("pipeline", {}).get("track_accumulation_limit", 15)
                            
                            if len(state["frame_buffer"]) < max_acc:
                                state["frame_buffer"].append(n_frame[:, :, :3].copy())
                                state["bbox_list"].append([(x, y), (x + w, y + h)])
                                state["class_history"].append(obj_meta.class_id)
                                state["area_history"].append(curr_area)
                            else:
                                # Replace the smallest area frame if the current one is larger
                                min_area = min(state["area_history"])
                                if curr_area > min_area:
                                    min_idx = state["area_history"].index(min_area)
                                    state["frame_buffer"][min_idx] = n_frame[:, :, :3].copy()
                                    state["bbox_list"][min_idx] = [(x, y), (x + w, y + h)]
                                    state["class_history"][min_idx] = obj_meta.class_id
                                    state["area_history"][min_idx] = curr_area


            
            l_obj = l_obj.next
        
        # Periodic Finalization Loop (Every 30 frames check for vehicles that left)
        if frame_meta.frame_num % 30 == 0:
            stale_ids = logic_engine.get_stale_tracks(frame_meta.frame_num, timeout=30)
            for tid in stale_ids:
                final_data = logic_engine.finalize(tid)
                if final_data:
                    cam_id = final_data["source_id"]
                    cam_real_name = source_id_map.get(cam_id, f"cam_{cam_id}")
                    
                    logging.info(f"🏁 [Handoff] Finalizing tracks for Vehicle:{tid} on {cam_real_name} (Status: {final_data['status']})")
                    ocr_worker.add_to_batch(
                        cam_real_name,
                        final_data.get("class_history", []),
                        [],  # Pass 1 crops removed (sustainable)
                        obj_id=tid,
                        full_image=final_data["best_img"],
                        violation_history=final_data["v_history"],
                        total_frames=final_data["frame_count"],
                        frame_buffer=final_data.get("frame_buffer", []),
                        bbox_list=final_data.get("bbox_list", []),
                        detection_time=time.time()
                    )
                    
                    # Cleanup local state dictionary to prevent linear memory growth
                    if tid in logic_engine.vehicle_states:
                        del logic_engine.vehicle_states[tid]

        # Explicitly Unmap PyDs surface to fix the massive memory leak!
        try:
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        except Exception as e:
            pass

        l_frame = l_frame.next
        
    return Gst.PadProbeReturn.OK

def main():
    global logic_engine, api_submitter, ocr_worker, config, pipeline_ptr
    print("🚀 [STARTUP] ITMS Pipeline Initializing...")
    
    if not os.path.exists("config.json"):
        sys.stderr.write("config.json not found\n")
        return

    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Initialize professional logging
    setup_itms_logging(config)
    cleanup_old_logs(config)
    threading.Thread(target=log_rotation_worker, args=(config,), daemon=True, name="LogRotation").start()
    
    log_context.camera_name = "System"
    logging.info("🚀 [STARTUP] ITMS Pipeline Initializing...")
    logging.info("✅ [STARTUP] Configuration loaded.")

    logging.info("⏳ [STARTUP] Initializing ApiSubmitter...")
    api_submitter = ApiSubmitter(config)
    logging.info("✅ [STARTUP] ApiSubmitter online.")

    logging.info("⏳ [STARTUP] Initializing EvidenceWorker...")
    evidence_worker = EvidenceWorker(config)
    logging.info("✅ [STARTUP] EvidenceWorker online.")

    logging.info("⏳ [STARTUP] Initializing OCRWorker...")
    ocr_worker = OCRWorker(config, api_submitter, evidence_worker)
    logging.info("✅ [STARTUP] OCRWorker online.")

    logging.info("⏳ [STARTUP] Initializing LogicEngine...")
    logic_engine = LogicEngine(config, submitter=api_submitter)
    logging.info("✅ [STARTUP] LogicEngine online.")

    logging.info("⏳ [STARTUP] GStreamer Init (Registry Scan)...")
    Gst.init(None)
    logging.info("✅ [STARTUP] GStreamer Initialized.")
    
    pipeline = Gst.Pipeline()
    pipeline_ptr = pipeline
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
        return

    # Create Elements
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    # Primary Inference & Tracking
    pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    
    # Logic Branch elements
    queue_logic = Gst.ElementFactory.make("queue", "queue-logic")
    nvvidconv_probe = Gst.ElementFactory.make("nvvideoconvert", "convertor-probe")
    caps_probe = Gst.ElementFactory.make("capsfilter", "capsfilter-probe")
    sink_logic = Gst.ElementFactory.make("fakesink", "sink-logic")

    # Display Branch elements (Standard GUI Dashboard)
    show_video = config["application"].get("show_video", False)
    tee = Gst.ElementFactory.make("tee", "nvs-tee")
    queue_display = Gst.ElementFactory.make("queue", "queue-display")
    
    if show_video:
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        nvvidconv_display = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-display")
        nvosd = Gst.ElementFactory.make("nvdsosd", "nvosd")
        sink_display = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink_display.set_property("sync", 0)  # Do not block X11 vsync
    else:
        sink_display = Gst.ElementFactory.make("fakesink", "sink-display-fake")
        sink_display.set_property("sync", 0)

    if not all([streammux, pgie, tracker, tee, queue_logic, 
                nvvidconv_probe, caps_probe, sink_logic, queue_display, sink_display]):
        sys.stderr.write(" Unable to create one or more elements \n")
        return
    
    if show_video and not all([tiler, nvvidconv_display, nvosd]):
        sys.stderr.write(" Unable to create display elements \n")
        return

    # Config Elements
    mem_type = config.get("pipeline", {}).get("nvbuf_memory_type", 3) # 3: Unified
    
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", len(config.get("enabled_cameras", [])))
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("nvbuf-memory-type", mem_type)
    
    # Dynamically map config.json 'inference_interval' to the underlying Triton config
    frame_skip = config.get("pipeline", {}).get("inference_interval", 0)
    with open("configs/pgie_config.txt", "r") as config_file:
        pgie_txt = config_file.read()
    import re
    pgie_txt = re.sub(r'interval:\s*\d+', f'interval: {frame_skip}', pgie_txt)
    with open("configs/pgie_config.txt", "w") as config_file:
        config_file.write(pgie_txt)
    
    pgie.set_property("config-file-path", "configs/pgie_config.txt")
    tracker.set_property("ll-config-file", "configs/tracker_itms_pro.txt")
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")

    # Tiler setup (Match Reference)
    if show_video:
        tiler_rows = int(np.sqrt(len(config.get("enabled_cameras", []))))
        tiler_cols = int(np.ceil(len(config.get("enabled_cameras", [])) / tiler_rows))
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_cols)
        tiler.set_property("width", 1920)
        tiler.set_property("height", 1080)

    # Logic branch memory type should be Unified (3) for stable pyds access on dGPU
    nvvidconv_probe.set_property("nvbuf-memory-type", 3)
    caps_probe.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080"))

    sink_logic.set_property("sync", 0)
    sink_display.set_property("sync", 0)
    sink_display.set_property("qos", 0)

    # Add to Pipeline
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(tee)
    pipeline.add(queue_logic)
    pipeline.add(nvvidconv_probe)
    pipeline.add(caps_probe)
    pipeline.add(sink_logic)
    
    pipeline.add(queue_display)
    pipeline.add(sink_display)
    if show_video:
        pipeline.add(tiler)
        pipeline.add(nvvidconv_display)
        pipeline.add(nvosd)

    # Link Sources
    for i, cam_name in enumerate(config.get("enabled_cameras", [])):
        uri = config.get("camera_url", {}).get(cam_name)
        if not uri: continue
        
        # Resolve relative local paths to absolute file:// URIs
        if not (uri.startswith("rtsp://") or uri.startswith("http://") or uri.startswith("file://")):
            uri = "file://" + os.path.abspath(uri)
            
        source_bin = create_source_bin(i, uri)
        source_bin_map[cam_name] = source_bin
        source_fail_counts[cam_name] = 0
        
        # Attach Independent Looping Probe
        src_pad = source_bin.get_static_pad("src")
        src_pad.add_probe(Gst.PadProbeType.EVENT_BOTH, source_eos_probe, (cam_name, source_bin))
        
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    # Link Common Chain (Inference + Tracker)
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(tee)

    # Link Logic Branch (src_0)
    tee_src_pad_logic = tee.request_pad_simple("src_%u")
    queue_logic_sink_pad = queue_logic.get_static_pad("sink")
    tee_src_pad_logic.link(queue_logic_sink_pad)
    
    queue_logic.link(nvvidconv_probe)
    nvvidconv_probe.link(caps_probe)
    caps_probe.link(sink_logic)

    # Link Display Branch (src_1)
    # Link Display Branch (src_1)
    tee_src_pad_display = tee.request_pad_simple("src_%u")
    queue_display_sink_pad = queue_display.get_static_pad("sink")
    tee_src_pad_display.link(queue_display_sink_pad)
    
    if show_video:
        logging.info("📺 [STARTUP] Linking Display Branch (Tiler + OSD + EGL)...")
        queue_display.link(tiler)
        tiler.link(nvvidconv_display)
        nvvidconv_display.link(nvosd)
        nvosd.link(sink_display)
    else:
        logging.info("🙈 [STARTUP] Side-loading Display Branch into fakesink (Headless)...")
        queue_display.link(sink_display)

    logging.info("✅ [STARTUP] All elements linked successfully.")

    # Populate source ID map for log routing
    for i, cam_name in enumerate(config.get("enabled_cameras", [])):
        source_id_map[i] = cam_name

    # Class filter probe: removes non-vehicle objects before tracker
    pgie_src_pad = pgie.get_static_pad("src")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, class_filter_probe, 0)

    # Main detection probe: attached after nvvidconv so buffer is CPU-mapped RGBA
    probe_pad = caps_probe.get_static_pad("src")
    probe_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, (loop, pipeline))

    logging.info("🚀 [STARTUP] Pipeline starting MainLoop...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
