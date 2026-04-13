import logging as log
import logging.handlers as lh
import os
import threading
import sys
import time
import shutil
from datetime import datetime, timezone, timedelta
from pygelf import GelfUdpHandler

# -----------------------------
# Resolve BASE_DIR (PyInstaller / Python)
# -----------------------------
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Adjusting to project root since logging_utils is in utils/
    BASE_DIR = os.path.dirname(BASE_DIR)

IST = timezone(timedelta(hours=5, minutes=30))

# Thread-local storage for camera context
log_context = threading.local()

class ISTFormatter(log.Formatter):
    """Formatter that forces timestamps to Indian Standard Time with milliseconds."""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="milliseconds")

class CameraIdFilter(log.Filter):
    """
    Filter that injects camera_id into log records and routes them
    to the correct file handler in multi-threaded mode.
    """
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        
    def filter(self, record):
        record.camera_id = self.camera_id
        current_cam = getattr(log_context, 'camera_name', 'System')
        
        if self.camera_id == 'System':
            return current_cam == 'System' or not hasattr(log_context, 'camera_name')
        return current_cam == self.camera_id

class CustomISTFormatter(ISTFormatter):
    """Combines IST timing with the specific [COMPONENT] format requested."""
    def __init__(self, use_color=True):
        self.use_color = use_color
        # Format: 2024-04-10 15:30:05,123 | INFO | [COMPONENT] message
        self.fmt = "%(asctime)s | %(levelname)-8s | [%(camera_id)s] %(message)s"
        super().__init__(fmt=self.fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record):
        if not hasattr(record, 'camera_id'):
            record.camera_id = getattr(log_context, 'camera_name', 'System')
        return super().format(record)

def _get_log_dir(config):
    """Creates and returns logs/YYYY-MM-DD/ path."""
    logs_root = config.get("application", {}).get("log_path", "logs")
    if not os.path.isabs(logs_root):
        logs_root = os.path.join(BASE_DIR, logs_root)
        
    current_date = datetime.now(IST).strftime("%Y-%m-%d")
    log_dir = os.path.join(logs_root, current_date)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def cleanup_old_logs(config, retention_months=3):
    """Delete log folders older than retention_months."""
    try:
        logs_root = config.get("application", {}).get("log_path", "logs")
        if not os.path.isabs(logs_root):
            logs_root = os.path.join(BASE_DIR, logs_root)
            
        if not os.path.exists(logs_root):
            return

        now = datetime.now(IST)
        retention_days = retention_months * 30
        
        for folder_name in os.listdir(logs_root):
            folder_path = os.path.join(logs_root, folder_name)
            if not os.path.isdir(folder_path): continue
            
            try:
                # Parse folder name as date
                folder_date = datetime.strptime(folder_name, "%Y-%m-%d").replace(tzinfo=IST)
                if (now - folder_date).days > retention_days:
                    shutil.rmtree(folder_path)
                    log.getLogger().info(f"[CLEANUP] Deleted old logs folder: {folder_name}")
            except (ValueError, Exception):
                continue
    except Exception as e:
        print(f"Error during log cleanup: {e}")

def setup_itms_logging(config):
    """Configure or re-configure logging to a date-wise folder."""
    root_logger = log.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    
    verbose = config.get("application", {}).get("verbose", False)
    root_logger.setLevel(log.DEBUG if verbose else log.INFO)
    
    log_dir = _get_log_dir(config)

    # 1. Console Handler
    console_handler = log.StreamHandler()
    console_handler.setFormatter(CustomISTFormatter(use_color=True))
    root_logger.addHandler(console_handler)

    # 2. System Log Handler (Global)
    sys_log_path = os.path.join(log_dir, "System.log")
    sys_handler = log.FileHandler(sys_log_path, encoding="utf-8")
    sys_handler.addFilter(CameraIdFilter("System"))
    sys_handler.setFormatter(CustomISTFormatter(use_color=False))
    root_logger.addHandler(sys_handler)

    # 3. Per-Camera Log Handlers
    enabled_cams = config.get("enabled_cameras", [])
    for cam_name in enabled_cams:
        cam_log_path = os.path.join(log_dir, f"{cam_name}.log")
        cam_handler = log.FileHandler(cam_log_path, encoding="utf-8")
        cam_handler.addFilter(CameraIdFilter(cam_name))
        cam_handler.setFormatter(CustomISTFormatter(use_color=False))
        root_logger.addHandler(cam_handler)

    # 4. Graylog GELF Handler
    outbound = config.get("outbound", {})
    gl_cfg = outbound.get("graylog", {})
    if gl_cfg.get("enabled"):
        try:
            gl_handler = GelfUdpHandler(
                host=gl_cfg.get("endpoint"),
                port=gl_cfg.get("port", 12201),
                include_extra_fields=True,
                facility=gl_cfg.get("facility", "itms-va"),
                _environment=config.get("application", {}).get("environment", "production")
            )
            gl_handler.setFormatter(log.Formatter("%(message)s"))
            root_logger.addHandler(gl_handler)
            log.info(f"✅ Graylog connectivity enabled via {gl_cfg['endpoint']}")
        except Exception as e:
            log.error(f"❌ Failed to initialize Graylog: {e}")

    log.info(f"ITMS Logging initialized. Day folder: {log_dir}")
    return root_logger

def log_rotation_worker(config):
    """Daily worker to rotate log folder at midnight and cleanup old logs."""
    while True:
        try:
            now = datetime.now(IST)
            next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            wait_seconds = (next_run - now).total_seconds()
            
            # Wait until midnight
            time.sleep(wait_seconds + 5)
            
            setup_itms_logging(config)
            cleanup_old_logs(config)
        except Exception as e:
            try:
                log.getLogger().error(f"[SYSTEM] Error in daily log rotation: {e}")
            except:
                print(f"Error in daily log rotation: {e}")
            time.sleep(60)
