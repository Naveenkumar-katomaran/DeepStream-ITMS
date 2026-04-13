import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

CUSTOM_LIB_DIR = "nvinfer_custom_yolo"

def run_command(cmd, shell=False):
    logging.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Command failed: {result.stderr}")
        return False
    return True

def setup_hardware():
    # 1. Compile Custom Parser
    logging.info("--- [CLEANUP] Compiling Custom YOLO Parser ---")
    if os.path.exists(CUSTOM_LIB_DIR):
        run_command(["make", "-C", CUSTOM_LIB_DIR, "clean"])
        if run_command(["make", "-C", CUSTOM_LIB_DIR]):
            # Copy to libs/
            os.makedirs("libs", exist_ok=True)
            run_command(["cp", f"{CUSTOM_LIB_DIR}/libnvdsinfer_custom_impl_Yolo.so", "libs/"])
            logging.info("✅ Custom parser compiled and moved to libs/")
        else:
            logging.error("❌ Failed to compile custom parser.")
            return

    logging.info("--- [CLEANUP] Hardware setup complete (DeepStream will handle engines) ---")

if __name__ == "__main__":
    setup_hardware()
