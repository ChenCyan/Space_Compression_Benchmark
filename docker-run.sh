#!/bin/bash
# =============================================================================
# Launch HyCASS container with GPU + project data mounted.
# =============================================================================
set -euo pipefail

IMAGE_NAME="${HYCASS_IMAGE:-hycass-torch:2.6}"
PROJECT_ROOT="/data/cyl/space_compression/hycass"
DATA_ROOT="/data/cyl/space_compression"

docker run \
    --rm -it \
    --gpus all \
    --shm-size=16g \
    --user 1008:1008 \
    -v "${PROJECT_ROOT}:/workspace" \
    -v "${DATA_ROOT}:/data/cyl/space_compression" \
    -w /workspace \
    "${IMAGE_NAME}" \
    "${@:-bash}"
