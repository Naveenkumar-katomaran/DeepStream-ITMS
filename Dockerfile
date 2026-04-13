FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# Layer 1: System and DeepStream Dependencies (STABLE)
RUN apt-get update && apt-get install -y \
    python3-gi python3-gst-1.0 python3-pip \
    cmake git build-essential libglib2.0-dev \
    python3-dev libgstrtspserver-1.0-dev libgstreamer1.0-dev \
    pybind11-dev python3-pybind11 wget xz-utils \
    libdca0 libmjpegutils-2.1-0 libmp3lame0 libmpg123-0 \
    libflac12 libdvdread8 libjbig0 libdvdnav4 \
    gstreamer1.0-libav libgl1 libglib2.0-0 && \
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xvf ffmpeg-release-amd64-static.tar.xz && \
    cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    rm -rf ffmpeg-*-amd64-static* && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: Build DeepStream Python Bindings
WORKDIR /opt/nvidia/deepstream/deepstream/sources
RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings
RUN git submodule update --init

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/build
RUN cmake .. && make -j$(nproc)

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings
# Copy the compiled .so directly into Python site-packages (pip install fails for this cmake-only project)
RUN cp build/pyds.so /usr/lib/python3/dist-packages/ && \
    cp build/pyds.so /usr/local/lib/python3.12/dist-packages/ && \
    echo "pyds.so installed to site-packages"

# Layer 3: Application Dependencies (CACHED)
RUN pip3 install shapely requests numpy ultralytics torch torchvision pika minio pytz pygelf \
                 onnx onnxruntime onnxslim pandas paho-mqtt psutil opencv-python --break-system-packages

# Layer 4: Application Runtime
WORKDIR /app
# Note: We rely on the docker-compose volume mapping (.:/app) 
# so your local changes to .py files are seen instantly without rebuilding.
