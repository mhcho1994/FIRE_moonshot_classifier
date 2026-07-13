FROM nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04 AS base

ARG VSCODE_SERVER_COMMIT=4949701c880d4bdb949e3c0e6b400288da7f474b
ARG TARGETARCH



ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace:/workspace/drone_classifier:/workspace/drone_classifier_svm:/workspace/trajectory_processing:/workspace/drone_classifier_modules/src

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"

RUN python -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip setuptools wheel

RUN pip install --index-url "${TORCH_INDEX_URL}" torch \
    && pip install \
    lightgbm \
    matplotlib \
    numpy \
    pandas \
    pymavlink \
    pyulog \
    pywavelets \
    pyyaml \
    rdp \
    scikit-learn \
    scipy \
    seaborn \
    tqdm \
    tsfresh

COPY . /workspace

RUN chown -R "${USERNAME}:${USERNAME}" /workspace

USER "${USERNAME}"

CMD ["/bin/bash"]

