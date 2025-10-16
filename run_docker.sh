#!/bin/bash

PHYS_DIR="/home/$(whoami)/MMTU"
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e MMTU_HOME="/workspace" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    mmtu-image \
    bash -c "cd /workspace && python3 inference.py -i ${1} -l info self_deploy --model Qwen/Qwen3-8B-AWQ"
