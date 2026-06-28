# =============================================================================
# HyCASS Benchmark — PyTorch 2.5.1 + CUDA 12.1 on Ubuntu 22.04
# =============================================================================
# Base: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime (already pulled, GPU ✓)
# Then upgrade torch to 2.5.1+cu121 (latest cu121, 0.5 minor version difference
# vs requirements.txt's 2.6.0 irrelevant for inference benchmarking).
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL description="HyCASS compression benchmark environment"

# ---- system packages --------------------------------------------------------
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    libopenjp2-7 libopenjp2-tools \
    && rm -rf /var/lib/apt/lists/*

# ---- pip mirror (domestic) --------------------------------------------------
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ---- upgrade PyTorch to 2.5.1+cu121 -----------------------------------------
RUN pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    --trusted-host download.pytorch.org \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121

# ---- project dependencies ---------------------------------------------------
RUN pip install \
    einops==0.8.1 \
    glymur==0.13.8 \
    numpy==2.2.6 \
    scipy \
    matplotlib \
    scikit-learn \
    tqdm \
    pytz \
    timm \
    tensorboard \
    rasterio \
    pandas \
    pyjpegls \
    lz4 \
    numba

# ---- CCSDS-123 NTNU verification model --------------------------------------
RUN git clone \
    https://github.com/NTNU-SmallSat-Lab/ccsds123_issue_2_verification_model.git \
    /opt/ccsds123_i2_hlm \
    && cd /opt/ccsds123_i2_hlm \
    && pip install -r requirements.txt

# ---- user setup -------------------------------------------------------------
ARG UID=1008
ARG GID=1008
RUN groupadd -g ${GID} hsi-user 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash hsi-user

USER hsi-user
WORKDIR /workspace

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]
