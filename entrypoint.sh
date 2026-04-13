#!/bin/bash
set -e

echo "--- [ENTRYPOINT] Starting ITMS System ---"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/app/libs:$LD_LIBRARY_PATH

# Force dynamic linker cache refresh to pick up new apt-installed libraries
ldconfig 2>/dev/null || true

# Enable hardware decoding (NVMM) as verified in the reference project
# export GST_PLUGIN_FEATURE_RANK="nvv4l2decoder:NONE" (Removed to restore HW decoding)
# Stability: Force Legacy Streammux (required for AMD/Nvidia Hybrid Laptop stability)
export USE_NEW_NVSTREAMMUX=0

# Logging: Increase GStreamer debug levels for better crash diagnostics
export GST_DEBUG=3
export GST_DEBUG_FILE=/app/logs/gst_debug.log
mkdir -p /app/logs
nvidia-smi || echo "⚠️ GPU not visible to container"

# 1. Run Hardware Setup (Model Conversion & Library Compilation)
echo "--- [ENTRYPOINT] Initializing Hardware Acceleration ---"
python3 tools/setup_hardware.py
python3 tools/setup_triton_repo.py

# 2. Launch Main Pipeline
echo "--- [ENTRYPOINT] Launching DeepStream Pipeline ---"
exec python3 main.py
