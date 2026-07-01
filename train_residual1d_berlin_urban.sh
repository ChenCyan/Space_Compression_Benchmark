#!/bin/bash
set -e
set -o pipefail

DATASET="./datasets/berlin-urban-gradient/"
SAVE_DIR="./results/berlin-urban-gradient-res1d/"
PRETRAINED_DIR="./pretrained/berlin-urban-gradient-res1d/"
EPOCHS=200
BATCH=4
LR=1e-4
WORKERS=0

mkdir -p "${SAVE_DIR}/logs"
mkdir -p "${PRETRAINED_DIR}"

MODELS_GPU0=(
  "hycass_res_cr004_spatial0x_n1024"
  "hycass_res_cr050_spatial0x_n1024"
  "hycass_res_cr101_spatial2x_n128"
)

MODELS_GPU1=(
  "hycass_res_cr016_spatial0x_n1024"
  "hycass_res_cr444_spatial2x_n128"
  "hycass_res_cr888_spatial2x_n128"
)

train_model() {
  local MODEL=$1
  local GPU=$2

  echo "=========================================="
  echo "Training: ${MODEL} GPU=${GPU}"
  echo "=========================================="

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU} python3 train.py \
    --devices 0 \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --epochs ${EPOCHS} \
    --train-batch-size ${BATCH} \
    --val-batch-size ${BATCH} \
    --num-workers ${WORKERS} \
    --learning-rate ${LR} \
    --save-dir "${SAVE_DIR}" \
    --experiment-name "${MODEL}" \
    --loss mse 2>&1 | tee "${SAVE_DIR}/logs/${MODEL}.log"

  local BEST="${SAVE_DIR}/${MODEL}/final.pth.tar"
  if [ -f "${BEST}" ]; then
    cp "${BEST}" "${PRETRAINED_DIR}/${MODEL}.pth.tar"
    echo "Saved: ${PRETRAINED_DIR}/${MODEL}.pth.tar"
  fi
}

(
  for MODEL in "${MODELS_GPU0[@]}"; do
    train_model "${MODEL}" 0
  done
) &
PID_GPU0=$!

(
  for MODEL in "${MODELS_GPU1[@]}"; do
    train_model "${MODEL}" 1
  done
) &
PID_GPU1=$!

wait ${PID_GPU0}
wait ${PID_GPU1}

echo ""
echo "All 6 residual1d models trained. Weights saved to ${PRETRAINED_DIR}"
ls "${PRETRAINED_DIR}"
