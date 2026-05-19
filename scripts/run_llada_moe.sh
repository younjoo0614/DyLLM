#!/bin/bash

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

MODEL_PATH="/data/model/LLaDA-MoE-7B-A1B-Instruct/"
TASK="gsm8k"
BATCH_SIZE=16
MAX_NEW_TOKENS=256
NUM_SHOT=5
NUM_STEPS=256
NUM_FULL_STEPS=4
REFRESH_INTERVAL=64
BLOCK_SIZE=32
THRESHOLD=0.99

OUTFILE="llada_moe_${TASK}_steps${NUM_STEPS}_full${NUM_FULL_STEPS}_r${REFRESH_INTERVAL}_t${THRESHOLD}"

CUDA_VISIBLE_DEVICES=0 python dyllm/eval/eval.py \
    --tasks $TASK \
    --batch-size $BATCH_SIZE \
    --model-path $MODEL_PATH \
    --max-new-tokens $MAX_NEW_TOKENS \
    --num-shot $NUM_SHOT \
    --num-steps $NUM_STEPS \
    --num-full-steps $NUM_FULL_STEPS \
    --refresh-interval $REFRESH_INTERVAL \
    --threshold $THRESHOLD \
    --output-file $OUTFILE \
    --log-samples \
    --block-size $BLOCK_SIZE
