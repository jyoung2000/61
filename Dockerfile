## Stage 1: Build frontend with Node
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

## Stage 2: Runtime
FROM python:3.11-slim

# Make pip tolerant of slow/large wheel downloads (torch + nvidia-*-cu12 wheels
# can be hundreds of MB; the 15 s default socket timeout turns a slow CDN read
# into a hard build failure). Applies to every pip call in this image.
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

# Make NVIDIA GPUs visible when passed through with --gpus
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
# Force pure-Python protobuf so MediaPipe 0.10.8 graph configs parse correctly
# with protobuf>=4 (required by torch/pyannote). The C++ implementation
# rejects 3.x-format graph definitions under protobuf 4.x.
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install system dependencies (ca-certificates ensures HTTPS model downloads work)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    ca-certificates \
    fontconfig \
    fonts-dejavu-core \
    fonts-freefont-ttf \
    fonts-liberation2 \
    unzip \
    libgl1-mesa-glx libglib2.0-0 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install DM Sans font (default subtitle font) so FFmpeg/libass can find it
# Downloaded directly from the canonical Google Fonts GitHub repo (stable raw URLs)
RUN mkdir -p /usr/share/fonts/truetype/dmsans && \
    curl -fsSL -o /usr/share/fonts/truetype/dmsans/DMSans.ttf \
      "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf" && \
    curl -fsSL -o /usr/share/fonts/truetype/dmsans/DMSans-Italic.ttf \
      "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans-Italic%5Bopsz%2Cwght%5D.ttf" && \
    fc-cache -f -v

# Install popular Google Fonts for subtitle use (variable + static weight files)
RUN mkdir -p /usr/share/fonts/truetype/google-fonts && \
    cd /usr/share/fonts/truetype/google-fonts && \
    curl -fsSL -o Montserrat.ttf "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf" && \
    curl -fsSL -o OpenSans.ttf "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf" && \
    curl -fsSL -o Roboto.ttf "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf" && \
    curl -fsSL -o Poppins-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf" && \
    curl -fsSL -o Poppins-Bold.ttf "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf" && \
    curl -fsSL -o Inter.ttf "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf" && \
    curl -fsSL -o Nunito.ttf "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito%5Bwght%5D.ttf" && \
    curl -fsSL -o Lato-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf" && \
    curl -fsSL -o Lato-Bold.ttf "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Bold.ttf" && \
    curl -fsSL -o Oswald.ttf "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald%5Bwght%5D.ttf" && \
    curl -fsSL -o PlayfairDisplay.ttf "https://github.com/google/fonts/raw/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf" && \
    curl -fsSL -o BebasNeue-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf" && \
    fc-cache -f -v

# Register /data/fonts with fontconfig so libass picks up custom fonts
RUN mkdir -p /data/fonts && \
    echo '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig><dir>/data/fonts</dir></fontconfig>' \
    > /etc/fonts/conf.d/99-custom-fonts.conf

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .

# Install CPU-only torch + torchaudio FIRST, pinned to the requirements.txt
# version, so the subsequent `-r requirements.txt` treats torch==2.5.1 as
# already satisfied and does NOT pull the multi-GB CUDA (nvidia-*-cu12) wheel
# stack from PyPI as a torch dependency. Those wheels are useless in the CPU
# image, and the ~665 MB nvidia-cudnn-cu12 download was both wasteful and a
# frequent build-failure point. (Mirrors the GPU Dockerfile, which likewise
# installs torch before requirements; the requirements.txt torch pin exists
# precisely so the pre-installed wheel is treated as satisfied.)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Remaining Python deps. requirements.txt pins torch==2.5.1 / torchaudio==2.5.1
# and pyannote.audio<4, so torch stays the CPU wheel installed above (pip does
# not swap in CUDA torch) and pyannote is installed here — no separate pass.
RUN pip install --no-cache-dir -r requirements.txt

# MediaPipe (installed separately to avoid the protobuf<4 conflict with torch).
RUN pip install --no-cache-dir \
        flatbuffers>=23.1.4 \
        attrs>=23.1.0 \
        sounddevice>=0.4.6 \
        absl-py>=1.0.0 && \
    # MediaPipe itself — skip deps to avoid protobuf<4 constraint
    pip install --no-cache-dir --no-deps mediapipe==0.10.8 && \
    # Verify MediaPipe can actually load (fail build early if broken)
    python3 -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__} installed')" && \
    python3 -c "import mediapipe.python.solutions.face_mesh; print('FaceMesh available')"

# Download YuNet model for face detection fallback (~350KB, one-time)
# Download lbpcascade_animeface for v2 Phase 6 anime face detection (~110KB)
RUN mkdir -p /app/backend/models && \
    curl -sL -o /app/backend/models/face_detection_yunet_2023mar.onnx \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" && \
    curl --retry 4 --retry-delay 5 --retry-all-errors -fsSL \
    -o /app/backend/models/lbpcascade_animeface.xml \
    "https://github.com/nagadomi/lbpcascade_animeface/raw/master/lbpcascade_animeface.xml"

# v4.1: pre-download YOLOv8n weights for PersonDetector / ObjectDetector
# (~5.5MB). object_detector.py looks at /data/models/yolov8n.pt first,
# so stash the weights there at build time. Without this step the first
# analysis run stalls while ultralytics downloads from GitHub, or fails
# silently on offline containers. Also verify load so broken builds
# fail early instead of silently landing backend=none.
RUN mkdir -p /data/models && \
    curl --retry 4 --retry-delay 5 --retry-all-errors -fsSL \
      -o /data/models/yolov8n.pt \
      "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt" && \
    python3 -c "from ultralytics import YOLO; m = YOLO('/data/models/yolov8n.pt'); print(f'YOLOv8n loaded: {len(m.names)} classes')"

# Pre-download the Demucs htdemucs model (~80 MB) for vocal separation so the
# first job with VOCAL_SEPARATION_ENABLED doesn't stall on a download (and works
# offline). Cached in the torch-hub dir; "|| echo WARN" keeps the build green if
# the CDN is unreachable — the runtime loader self-heals to the full audio.
RUN python3 -c "from demucs.pretrained import get_model; get_model('htdemucs'); print('htdemucs cached')" \
    || echo "WARN: htdemucs pre-download failed — will download at runtime"

# Pre-download InsightFace buffalo_s ArcFace pack so the first run with
# CLIPAI_FACE_EMBEDDING=arcface doesn't stall while the model downloads.
# CPU-only via onnxruntime — never touches the GPU. The "|| echo" keeps
# the image build green if the deepghs CDN is unreachable; the lazy
# loader will retry at runtime.
RUN mkdir -p /app/backend/models/insightface && \
    python3 -c "from insightface.app import FaceAnalysis; \
                FaceAnalysis(name='buffalo_s', root='/app/backend/models/insightface', \
                             providers=['CPUExecutionProvider']).prepare(ctx_id=-1, det_size=(320,320))" \
    || echo "WARN: buffalo_s pre-download failed — will download at runtime"

# Install CUDA runtime libraries via pip for GPU passthrough support.
# These PyPI packages provide the CUDA shared libraries that ctranslate2
# and faster-whisper need — no NVIDIA apt repo or system CUDA required.
# The "|| true" ensures the build succeeds on non-x86 architectures
# where these wheels may not be available.
RUN pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    2>/dev/null || true

# Point LD_LIBRARY_PATH at the pip-installed NVIDIA libs so ctranslate2 finds them
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvrtc/lib:\
${LD_LIBRARY_PATH}

# Copy backend source
COPY backend/ ./backend/

# Bake the build identity into the image so the running container can log
# exactly which commit it was built from (the repo's .git isn't copied, so
# `git rev-parse` inside the container returns "unknown"). Pass at build time:
#   docker build --build-arg BUILD_SHA=$(git rev-parse --short HEAD) \
#                --build-arg BUILD_SUBJECT="$(git log -1 --pretty=%s)" .
# docker-compose passes these automatically (see docker-compose.yml args).
ARG BUILD_SHA=unknown
ARG BUILD_SUBJECT=""
RUN printf '%s\n%s\n' "$BUILD_SHA" "$BUILD_SUBJECT" > /app/BUILD_INFO

# Copy the markdown docs so the cloud storage setup guide (and friends)
# can be rendered at /docs/cloud-storage/SETUP.md by backend/main.py.
# This is a few KB of markdown; excluding it makes the Settings page's
# "Setup guide →" link land on a blank page.
COPY docs/ ./docs/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./static

EXPOSE 1353

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "1353", "--workers", "1"]
