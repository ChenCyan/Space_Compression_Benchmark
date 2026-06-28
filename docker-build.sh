#!/bin/bash
# =============================================================================
# Build the HyCASS benchmark Docker image.
#   cd /data/cyl/space_compression/hycass
#   bash docker-build.sh
# =============================================================================
set -euo pipefail

IMAGE_NAME="hycass-torch:2.6"
cd "$(dirname "$0")"

echo ">>> Building ${IMAGE_NAME} ..."

docker build \
    --build-arg UID=1008 \
    --build-arg GID=1008 \
    -t "${IMAGE_NAME}" \
    -f Dockerfile \
    .

echo ""
echo ">>> Build complete. Verifying GPU ..."
docker run --rm --gpus all "${IMAGE_NAME}" python -c "
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('count:', torch.cuda.device_count())
"
