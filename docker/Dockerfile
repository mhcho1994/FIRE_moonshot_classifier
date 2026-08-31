# syntax=docker/dockerfile:1

# -----------------------------------------------------------------------------
# Runtime image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

ARG USERNAME=moonshot
ARG USER_UID=1000
ARG USER_GID=1000

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132

# Separate project source code from virtual environment to avoid conflicts with dependencies
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    HOME=/home/${USERNAME} \
    PROJECT_DIR=/home/${USERNAME}/FIRE_moonshot_classifier \
    PATH=/opt/venv/bin:${PATH}

WORKDIR ${PROJECT_DIR}

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        gzip \
        libgomp1 \
        procps \
        python3 \
        python3-dev \
        python3-venv \
        tar \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------- 
# Python virtual environment 
# 
# This is installed before user creation so that changing USER_UID/USER_GID 
# does not invalidate the large Python dependency layers. 
# -----------------------------------------------------------------------------
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade pip setuptools wheel packaging

# ----------------------------------------------------------------------------- 
# Python dependencies 
# 
# The runtime image contains dependencies but not the project source. 
# Dependencies are read from pyproject.toml. 
# PyTorch is installed separately because its CUDA wheel uses a dedicated # package index. 
# -------------------------------------------------------------------------
COPY --chown=${USER_UID}:${USER_GID} pyproject.toml /tmp/fire-moonshot-classifier/pyproject.toml

RUN TORCH_INDEX_URL="${TORCH_INDEX_URL}" python - <<'PY'
import os
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

pyproject_path = "/tmp/fire-moonshot-classifier/pyproject.toml"

with open(pyproject_path, "rb") as file:
    pyproject = tomllib.load(file)

dependencies = pyproject["project"].get("dependencies", [])

torch_dependencies = []
regular_dependencies = []

for dependency in dependencies:
    requirement = Requirement(dependency)

    if canonicalize_name(requirement.name) == "torch":
        torch_dependencies.append(dependency)
    else:
        regular_dependencies.append(dependency)

if torch_dependencies:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            os.environ["TORCH_INDEX_URL"],
            *torch_dependencies,
        ]
    )

if regular_dependencies:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *regular_dependencies,
        ]
    )
PY

RUN rm -rf /tmp/fire-moonshot-classifier 

# -----------------------------------------------------------------------------
# Non-root user setup
#
# The requested UID/GID can be supplied from the host. 
# 
# Examples: 
#   host mhcho: UID=1003, GID=1003 
#   container: UID=1003, GID=1003, username=moonshot
#
# Ubuntu 24.04 images may already contain: ubuntu:x:1000:1000.
#
# If the requested UID already exists, reuse and rename that account. 
# If the requested GID already exists, reuse the existing group.
# -----------------------------------------------------------------------------
RUN set -eux; \
    existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
    existing_group="$(getent group "${USER_GID}" | cut -d: -f1 || true)"; \
    \
    if [ -z "${existing_group}" ]; then \
        if getent group "${USERNAME}" >/dev/null 2>&1; then \ 
            groupmod \ 
                --gid "${USER_GID}" \ 
                "${USERNAME}"; \ 
        else \ 
            groupadd \ 
                --gid "${USER_GID}" \ 
                "${USERNAME}"; \ 
        fi; \ 
    fi; \ 
    \
    if id "${USERNAME}" >/dev/null 2>&1; then \
        user_at_requested_uid="$( \
            getent passwd "${USER_UID}" | cut -d: -f1 || true \
        )"; \
        \
        if [ -n "${user_at_requested_uid}" ] \
            && [ "${user_at_requested_uid}" != "${USERNAME}" ]; then \
            echo "UID ${USER_UID} is already assigned to ${user_at_requested_uid}" >&2; \
            exit 1; \
        fi; \
        \
        current_home="$( \
            getent passwd "${USERNAME}" | cut -d: -f6 \
        )"; \
        \
        usermod \
            --uid "${USER_UID}" \
            --gid "${USER_GID}" \
            --shell /bin/bash \
            "${USERNAME}"; \
        \
        if [ "${current_home}" != "${HOME}" ]; then \
            usermod \
                --home "${HOME}" \
                --move-home \
                "${USERNAME}"; \
        fi; \
    elif [ -n "${existing_user}" ]; then \
        usermod \
            --login "${USERNAME}" \
            --home "${HOME}" \
            --move-home \
            --gid "${USER_GID}" \
            --shell /bin/bash \
            "${existing_user}"; \
    else \
        useradd \
            --uid "${USER_UID}" \
            --gid "${USER_GID}" \
            --create-home \
            --home-dir "${HOME}" \
            --shell /bin/bash \
            "${USERNAME}"; \
    fi; \
    mkdir -p "${PROJECT_DIR}"; \ 
    chown -R "${USER_UID}:${USER_GID}" "${HOME}" "${VIRTUAL_ENV}"

# -----------------------------------------------------------------------------
# Development image
#
# Project source is not copied into this image. 
# setup_docker.sh bind-mounts the host repository and performs an editable 
# installation after starting the container.
# -----------------------------------------------------------------------------
FROM runtime AS dev

ARG USERNAME=moonshot

WORKDIR ${PROJECT_DIR}

USER ${USERNAME} 

CMD ["/bin/bash"]

# -----------------------------------------------------------------------------
# Development image
#
# Project source is copied into the image and installed non-editably.
# -----------------------------------------------------------------------------
FROM runtime AS release

ARG USERNAME=moonshot 
ARG USER_UID=1000 
ARG USER_GID=1000

WORKDIR ${PROJECT_DIR}

COPY --chown=${USER_UID}:${USER_GID} . .

# Dependencies were already installed in the runtime stage. 
RUN python -m pip install \ 
    --no-deps \ 
    --no-build-isolation \ 
    . 

USER ${USERNAME} 

ENTRYPOINT ["fireclassify"]
