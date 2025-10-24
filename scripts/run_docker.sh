#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=nvidia_geforce_rtx_3090:1

PHYS_DIR="/home/$(whoami)/MMTU"
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"
INPUT_FILE="${1:-mmtu.jsonl}"
MODEL="${2:-llama-3.1-8b-awq}"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e MMTU_HOME="/workspace" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    mmtu-image \
    bash -c "cd /workspace && python3 inference.py -i $INPUT_FILE -l info self_deploy --model $MODEL"
