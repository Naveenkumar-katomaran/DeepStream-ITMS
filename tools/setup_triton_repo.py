import os
import shutil
import logging
import subprocess

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | [Triton-Setup] %(message)s')

def run_cmd(cmd):
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Command failed: {result.stderr}")
        return False
    return True

# Ironclad Triton config for Laptop GPUs (RTX 5060)
def get_config_pbtxt(name, input_shape, output_dims):
    # Standardizing for 640x640 and enabling TensorRT Accelerator
    config = f"""name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: {input_shape}
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: {output_dims}
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

optimization {{
  execution_accelerators {{
    gpu_execution_accelerator {{
      name: "tensorrt"
      parameters {{ key: "precision_mode" value: "FP16" }}
      parameters {{ key: "max_workspace_size_bytes" value: "1073741824" }}
    }}
  }}
}}

parameters [
  {{
    key: "intra_op_parallelism",
    value: {{ string_value: "1" }}
  }},
  {{
    key: "inter_op_parallelism",
    value: {{ string_value: "1" }}
  }}
]
"""
    return config

def setup_model(name, pt_path, input_shape, output_shape):
    repo_root = "/app/model_repo"
    model_dir = os.path.join(repo_root, name)
    version_dir = os.path.join(model_dir, "1")
    
    if os.path.exists(os.path.join(version_dir, "model.onnx")):
        logging.info(f"✅ Model {name} already exists in repo. Skipping export.")
        return True

    logging.info(f"🚀 Exporting {name} ({pt_path}) to ONNX...")
    os.makedirs(version_dir, exist_ok=True)
    
    # Export using yolo CLI (installed in Dockerfile)
    # We use opset=12 for maximum compatibility with DeepStream 9.0
    export_cmd = ["yolo", "export", f"model={pt_path}", "format=onnx", "opset=12", "simplify=True"]
    if not run_cmd(export_cmd):
        return False
    
    # Move the exported onnx to the repo
    onnx_src = pt_path.replace(".pt", ".onnx")
    if os.path.exists(onnx_src):
        shutil.move(onnx_src, os.path.join(version_dir, "model.onnx"))
        logging.info(f"📦 Moved {name} to Triton Repo.")
    else:
        logging.error(f"❌ Failed to find exported ONNX at {onnx_src}")
        return False

    # Write config.pbtxt
    config_content = get_config_pbtxt(name, input_shape, output_shape)
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
        f.write(config_content)
    logging.info(f"📝 Generated config.pbtxt for {name}")
    
    return True

def main():
    repo_root = "/app/model_repo"
    
    # CRITICAL FIX: Removed Total Wipe
    # Previously, this script wiped the model_repo every single run.
    # This forced TensorRT to re-compile 4 massive AI engines locally, which maxed out
    # 100% of the Katomaran laptop GPU and CPU for 5 minutes causing the entire system to freeze.
    # By preserving the repository, DeepStream caches the .engine files and boots instantly!
    
    os.makedirs(repo_root, exist_ok=True)
    
    logging.info("--- Starting Triton Auto-Onboarding ---")
    
    # Define models to onboard
    # Format: (Triton Name, Local Path, Input Shape, Output Shape)
    # UNIFIED SHAPE: All models set to 640x640 to match working backup (/app/batch_backup)
    models = [
        ("yolov8n", "/app/model/vehicle_detection/yolov8n.pt", "[1, 3, 640, 640]", "[1, 84, 8400]"),
        ("car", "/app/model/car_model/car_model.pt", "[1, 3, 640, 640]", "[1, 10, 8400]"),
        ("bike", "/app/model/bike_model/bike_model.pt", "[1, 3, 640, 640]", "[1, 8, 8400]"),
        ("ocr", "/app/model/ocr/my_model.pt", "[1, 3, 640, 640]", "[1, 40, 8400]")
    ]
    
    for name, path, in_s, out_s in models:
        # FORCE re-export once to synchronize everything to 640x640
        # setup_model(name, path, in_s, out_s) 
        
        # Restoration logic:
        if not os.path.exists(path):
            logging.warning(f"⚠️ Source model {path} not found. Skipping {name}.")
            continue
        
        setup_model(name, path, in_s, out_s)

    logging.info("--- Triton Auto-Onboarding Complete ---")

if __name__ == "__main__":
    main()
