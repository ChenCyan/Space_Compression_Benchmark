#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DEVICES=0
NUM_WORKERS=4

MODEL=hycass_cr202_spatial2x_n128

LEARNING_RATE=1e-4
EPOCHS=200

MODE=easy

TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=16

SAVE_DIR=./results/
EXPERIMENT_NAME="trial"

LOSS=mse

nohup \
  python -u train.py \
    --devices ${DEVICES} \
    --train-batch-size ${TRAIN_BATCH_SIZE} \
    --val-batch-size ${VAL_BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --learning-rate ${LEARNING_RATE} \
    --mode ${MODE} \
    --model ${MODEL} \
    --loss ${LOSS} \
    --epochs ${EPOCHS} \
    --save-dir ${SAVE_DIR} \
    --experiment-name ${EXPERIMENT_NAME} \
  &> results/logs/${DEVICES}.log &
exit
