#!/usr/bin/env bash

SCRIPT_SOURCED=0

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    SCRIPT_SOURCED=1
else
    set -Eeuo pipefail
fi

# =============================================================================
# Project configuration
# =============================================================================

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

SCRIPT_COMMAND="${BASH_SOURCE[0]}"

PROJECT_ROOT="$(
    cd -- "${SCRIPT_DIR}/.."
    pwd
)"

DOCKERFILE="${PROJECT_ROOT}/docker/Dockerfile"

PROJECT_NAME="fire_moonshot_classifier"

DEV_IMAGE="${PROJECT_NAME}:dev"
RELEASE_IMAGE="${PROJECT_NAME}:release"

DEV_CONTAINER="${PROJECT_NAME}_dev"
RELEASE_CONTAINER="${PROJECT_NAME}_release"

CONTAINER_PROJECT_DIR="/home/moonshot/FIRE_moonshot_classifier"


# =============================================================================
# Build and runtime options
# =============================================================================

# Development images use the host UID/GID by default so bind-mounted files
# remain owned by the host user.
DEV_UID="${USER_UID:-$(id -u)}"
DEV_GID="${USER_GID:-$(id -g)}"

# Release images use 1000:1000 by default for portability.
# Set MATCH_HOST_ID=1 to use the host UID/GID for release builds as well.
RELEASE_UID="${RELEASE_UID:-1000}"
RELEASE_GID="${RELEASE_GID:-1000}"

USE_GPU="${USE_GPU:-1}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    SELECTED_TORCH_INDEX_URL="${TORCH_INDEX_URL}"
elif [[ "${USE_GPU}" == "1" ]]; then
    SELECTED_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu132"
else
    SELECTED_TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
fi

GPU_ARGS=()

if [[ "${USE_GPU}" == "1" ]]; then
    GPU_ARGS=(
        --gpus "${DOCKER_GPUS}"
    )
fi


# =============================================================================
# Utility functions
# =============================================================================

print_usage() {
    cat <<EOF
Usage:
  ${SCRIPT_COMMAND} dev-build
  ${SCRIPT_COMMAND} dev-run
  ${SCRIPT_COMMAND} dev-shell
  ${SCRIPT_COMMAND} dev-stop
  ${SCRIPT_COMMAND} dev-remove
  ${SCRIPT_COMMAND} dev-clean
  ${SCRIPT_COMMAND} release-build
  ${SCRIPT_COMMAND} release-run
  ${SCRIPT_COMMAND} release-shell
  ${SCRIPT_COMMAND} release-stop
  ${SCRIPT_COMMAND} release-remove
  ${SCRIPT_COMMAND} release-clean

Commands:
  dev-build
      Build the development image.

  dev-run
      Recreate the development container from the development image,
      bind-mount the host repository, and install the project editable.

  dev-shell
      Open Bash in the running development container.

  dev-stop
      Stop the development container without removing it.

  dev-remove
      Remove the development container.

  dev-clean
      Remove the development container and development image.

  release-build
      Build the release image using a normal, non-editable installation.

  release-run
      Recreate a persistent release container from the release image.

  release-shell
      Open Bash in the running release container.

  release-stop
      Stop the release container without removing it.

  release-remove
      Remove the release container.

  release-clean
      Remove the release container and release image.

Environment variables:
  USER_UID=<uid>
      Override the UID used by the development container.

  USER_GID=<gid>
      Override the GID used by the development container.

  MATCH_HOST_ID=1
      Use the host UID/GID for the release image.

  RELEASE_UID=<uid>
  RELEASE_GID=<gid>
      Override the release image UID/GID.

  USE_GPU=0
      Disable Docker GPU access and use CPU-only PyTorch by default.

  DOCKER_GPUS=<value>
      Value passed to 'docker run --gpus'. Default: all.

  TORCH_INDEX_URL=<url>
      Override the PyTorch package index.

  NO_CACHE=1
      Disable Docker build cache.
EOF
}


require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "Error: docker was not found in PATH." >&2
        return 1
    fi

    if ! docker info >/dev/null 2>&1; then
        echo "Error: the Docker daemon is unavailable or permission was denied." >&2
        return 1
    fi
}


image_exists() {
    local image="$1"

    docker image inspect "${image}" >/dev/null 2>&1
}


container_exists() {
    local container="$1"

    docker container inspect "${container}" >/dev/null 2>&1
}


container_is_running() {
    local container="$1"

    if ! container_exists "${container}"; then
        return 1
    fi

    [[ "$(
        docker container inspect \
            --format '{{.State.Running}}' \
            "${container}"
    )" == "true" ]]
}


build_image() {
    local target="$1"
    local image="$2"
    local uid="$3"
    local gid="$4"

    local build_args=(
        --file "${DOCKERFILE}"
        --target "${target}"
        --tag "${image}"
        --build-arg "USERNAME=moonshot"
        --build-arg "USER_UID=${uid}"
        --build-arg "USER_GID=${gid}"
        --build-arg "TORCH_INDEX_URL=${SELECTED_TORCH_INDEX_URL}"
    )

    if [[ "${NO_CACHE:-0}" == "1" ]]; then
        build_args+=(--no-cache)
    fi

    echo "Building ${image}"
    echo "  target: ${target}"
    echo "  UID:GID: ${uid}:${gid}"
    echo "  torch index: ${SELECTED_TORCH_INDEX_URL}"

    docker build \
        "${build_args[@]}" \
        "${PROJECT_ROOT}"
}


remove_container() {
    local container="$1"

    if container_exists "${container}"; then
        docker container rm \
            --force \
            "${container}" \
            >/dev/null

        echo "Removed container: ${container}"
    fi
}


remove_image() {
    local image="$1"

    if image_exists "${image}"; then
        docker image rm \
            --force \
            "${image}" \
            >/dev/null

        echo "Removed image: ${image}"
    fi
}


build_dev_image() {
    build_image \
        "dev" \
        "${DEV_IMAGE}" \
        "${DEV_UID}" \
        "${DEV_GID}"
}


start_dev_container() {
    if ! image_exists "${DEV_IMAGE}"; then
        echo "Error: ${DEV_IMAGE} does not exist." >&2
        echo "Build it with: ${SCRIPT_COMMAND} dev-build" >&2
        return 1
    fi

    remove_container "${DEV_CONTAINER}"

    echo "Starting development container"

    docker run \
        --detach \
        --name "${DEV_CONTAINER}" \
        --hostname "moonshot-dev" \
        --init \
        "${GPU_ARGS[@]}" \
        --mount type=bind,source="${PROJECT_ROOT}",target="${CONTAINER_PROJECT_DIR}" \
        --workdir "${CONTAINER_PROJECT_DIR}" \
        "${DEV_IMAGE}" \
        sleep infinity \
        >/dev/null

    echo "Installing project in editable mode"

    if ! docker exec \
        --workdir "${CONTAINER_PROJECT_DIR}" \
        "${DEV_CONTAINER}" \
        python -m pip install \
            --no-deps \
            --no-build-isolation \
            --editable \
            .
    then
        echo "Error: editable installation failed." >&2
        docker container logs "${DEV_CONTAINER}" || true
        remove_container "${DEV_CONTAINER}"
        return 1
    fi

    echo
    echo "Development container is ready:"
    echo "  image:     ${DEV_IMAGE}"
    echo "  container: ${DEV_CONTAINER}"
    echo "  project:   ${CONTAINER_PROJECT_DIR}"
    echo "  UID:GID:   ${DEV_UID}:${DEV_GID}"
    echo
    echo "Open a shell with:"
    echo "  ${SCRIPT_COMMAND} dev-shell"
}


open_dev_shell() {
    if ! container_is_running "${DEV_CONTAINER}"; then
        echo "Error: ${DEV_CONTAINER} is not running." >&2
        echo "Start it with: ${SCRIPT_COMMAND} dev-run" >&2
        return 1
    fi

    docker exec \
        --interactive \
        --tty \
        --workdir "${CONTAINER_PROJECT_DIR}" \
        "${DEV_CONTAINER}" \
        /bin/bash
}


stop_dev_container() {
    if container_is_running "${DEV_CONTAINER}"; then
        docker container stop "${DEV_CONTAINER}" >/dev/null
        echo "Stopped container: ${DEV_CONTAINER}"
    elif container_exists "${DEV_CONTAINER}"; then
        echo "Container is already stopped: ${DEV_CONTAINER}"
    else
        echo "Container does not exist: ${DEV_CONTAINER}"
    fi
}


clean_dev() {
    remove_container "${DEV_CONTAINER}"
    remove_image "${DEV_IMAGE}"
}


get_release_identity() {
    if [[ "${MATCH_HOST_ID:-0}" == "1" ]]; then
        RELEASE_UID="${DEV_UID}"
        RELEASE_GID="${DEV_GID}"
    fi
}


build_release_image() {
    get_release_identity

    build_image \
        "release" \
        "${RELEASE_IMAGE}" \
        "${RELEASE_UID}" \
        "${RELEASE_GID}"
}


ensure_release_image() {
    if ! image_exists "${RELEASE_IMAGE}"; then
        echo "Error: ${RELEASE_IMAGE} does not exist." >&2
        echo "Build it with: ${SCRIPT_COMMAND} release-build" >&2
        return 1
    fi
}


start_release_container() {
    ensure_release_image

    remove_container "${RELEASE_CONTAINER}"

    echo "Starting release container"

    docker run \
        --detach \
        --name "${RELEASE_CONTAINER}" \
        --hostname "moonshot-release" \
        --init \
        "${GPU_ARGS[@]}" \
        --workdir "${CONTAINER_PROJECT_DIR}" \
        --entrypoint /bin/bash \
        "${RELEASE_IMAGE}" \
        -lc "sleep infinity" \
        >/dev/null

    echo
    echo "Release container is ready:"
    echo "  image:     ${RELEASE_IMAGE}"
    echo "  container: ${RELEASE_CONTAINER}"
    echo "  project:   ${CONTAINER_PROJECT_DIR}"
    echo
    echo "Open a shell with:"
    echo "  ${SCRIPT_COMMAND} release-shell"
}


open_release_shell() {
    if ! container_is_running "${RELEASE_CONTAINER}"; then
        echo "Error: ${RELEASE_CONTAINER} is not running." >&2
        echo "Start it with: ${SCRIPT_COMMAND} release-run" >&2
        return 1
    fi

    docker exec \
        --interactive \
        --tty \
        --workdir "${CONTAINER_PROJECT_DIR}" \
        "${RELEASE_CONTAINER}" \
        /bin/bash
}


stop_release_container() {
    if container_is_running "${RELEASE_CONTAINER}"; then
        docker container stop "${RELEASE_CONTAINER}" >/dev/null
        echo "Stopped container: ${RELEASE_CONTAINER}"
    elif container_exists "${RELEASE_CONTAINER}"; then
        echo "Container is already stopped: ${RELEASE_CONTAINER}"
    else
        echo "Container does not exist: ${RELEASE_CONTAINER}"
    fi
}


clean_release() {
    remove_container "${RELEASE_CONTAINER}"
    remove_image "${RELEASE_IMAGE}"
}


# =============================================================================
# Main
# =============================================================================

main() {
    local command="${1:-help}"

    if (($# > 0)); then
        shift
    fi

    case "${command}" in
        help|-h|--help)
            print_usage
            ;;

        dev-build)
            require_docker
            build_dev_image
            ;;

        dev-run)
            require_docker
            start_dev_container
            ;;

        dev-shell)
            require_docker
            open_dev_shell
            ;;

        dev-stop)
            require_docker
            stop_dev_container
            ;;

        dev-remove)
            require_docker
            remove_container "${DEV_CONTAINER}"
            ;;

        dev-clean)
            require_docker
            clean_dev
            ;;

        release-build)
            require_docker
            build_release_image
            ;;

        release-run)
            require_docker
            start_release_container
            ;;

        release-shell)
            require_docker
            open_release_shell
            ;;

        release-stop)
            require_docker
            stop_release_container
            ;;

        release-remove)
            require_docker
            remove_container "${RELEASE_CONTAINER}"
            ;;

        release-clean)
            require_docker
            clean_release
            ;;

        *)
            echo "Error: unknown command '${command}'." >&2
            echo >&2
            print_usage >&2
            return 2
            ;;
    esac
}

if [[ "${SCRIPT_SOURCED}" == "1" ]]; then
    main "$@"
    return $?
fi

main "$@"
