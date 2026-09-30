#!/usr/bin/env bash
# Shared by local setup and the Docker runtime image. Does not configure remotes.
set -euo pipefail

if command -v rclone >/dev/null 2>&1; then
    echo "Using existing rclone: $(command -v rclone)"
    rclone version
    exit 0
fi

RCLONE_PRIVILEGE=()
if [[ "${EUID}" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "Error: installing rclone requires root or sudo." >&2
        exit 1
    fi
    RCLONE_PRIVILEGE=(sudo)
fi

if ! command -v curl >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Error: install curl and unzip, then rerun this script." >&2
        exit 1
    fi
    "${RCLONE_PRIVILEGE[@]}" apt-get update
    "${RCLONE_PRIVILEGE[@]}" apt-get install -y --no-install-recommends ca-certificates curl unzip
fi

RCLONE_INSTALL_DIR="$(mktemp -d)"
trap 'rm -rf "${RCLONE_INSTALL_DIR}"' EXIT
curl --fail --silent --show-error --location --retry 3 \
    https://rclone.org/install.sh --output "${RCLONE_INSTALL_DIR}/install.sh"

# The official installer returns 3 if another installation is already current.
RCLONE_INSTALL_STATUS=0
"${RCLONE_PRIVILEGE[@]}" bash "${RCLONE_INSTALL_DIR}/install.sh" || RCLONE_INSTALL_STATUS=$?
if [[ "${RCLONE_INSTALL_STATUS}" -ne 0 && "${RCLONE_INSTALL_STATUS}" -ne 3 ]]; then
    exit "${RCLONE_INSTALL_STATUS}"
fi
rclone version
