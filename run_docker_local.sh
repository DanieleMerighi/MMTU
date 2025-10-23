#!/bin/bash

PHYS_DIR="/home/$(whoami)/MMTU"
LOCAL_CACHE_DIR="/home/$(whoami)/MMTU/models_cache"
DOCKER_INTERNAL_CACHE_DIR="/models_cache"
INPUT_FILE="${1:-mmtu.jsonl}"
MODEL="${2:-llama-3.1-8b-awq}"

# Crea directory cache se non esiste
mkdir -p "$LOCAL_CACHE_DIR"

echo "Using LOCAL cache: $LOCAL_CACHE_DIR"
echo "Dataset: $INPUT_FILE"
echo "Model: $MODEL"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LOCAL_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e MMTU_HOME="/workspace" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    mmtu-image \
    bash -c "cd /workspace && python3 inference.py -i $INPUT_FILE -l info self_deploy --model $MODEL"
