#!/bin/bash
# 训练 6 个 HyCASS 模型，两张 GPU 各 3 个，串行执行
set -e

DATASET="./datasets/berlin-dense/"
SAVE_DIR="./results/berlin-dense/"
EPOCHS=200
BATCH=16
LR=1e-4
WORKERS=0

mkdir -p "$SAVE_DIR/logs"
mkdir -p pretrained/berlin-dense

MODELS_GPU0=(
  "hycass_cr004_spatial0x_n1024"
  "hycass_cr050_spatial0x_n1024"
  "hycass_cr101_spatial2x_n128"
)

MODELS_GPU1=(
  "hycass_cr016_spatial0x_n1024"
  "hycass_cr444_spatial2x_n128"
  "hycass_cr888_spatial2x_n128"
)

train_model() {
  local MODEL=$1
  local GPU=$2
  echo "=========================================="
  echo "Training: $MODEL  GPU=$GPU"
  echo "=========================================="
  CUDA_VISIBLE_DEVICES=$GPU python3 train.py \
    --devices $GPU \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --epochs $EPOCHS \
    --train-batch-size $BATCH \
    --val-batch-size $BATCH \
    --num-workers $WORKERS \
    --learning-rate $LR \
    --save-dir "$SAVE_DIR" \
    --experiment-name "$MODEL" \
    --loss mse 2>&1 | tee "$SAVE_DIR/logs/${MODEL}.log"

  # 把最终权重复制到 pretrained/
  BEST="$SAVE_DIR/$MODEL/final.pth.tar"
  if [ -f "$BEST" ]; then
    cp "$BEST" "pretrained/berlin-dense/${MODEL}.pth.tar"
    echo "Saved: pretrained/berlin-dense/${MODEL}.pth.tar"
  fi
}

# GPU0：3 个模型串行
for M in "${MODELS_GPU0[@]}"; do
  train_model "$M" 0 &
  PID_GPU0=$!
  wait $PID_GPU0
done &

# GPU1：3 个模型串行
for M in "${MODELS_GPU1[@]}"; do
  train_model "$M" 1 &
  PID_GPU1=$!
  wait $PID_GPU1
done &

# 等待两组都完成
wait

echo ""
echo "All 6 models trained. Weights saved to pretrained/berlin-dense/"
ls pretrained/berlin-dense/
