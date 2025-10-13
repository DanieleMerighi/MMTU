#!/bin/bash

# Local testing script (no SLURM)
PHYS_DIR="$(pwd)"
LLM_CACHE_DIR="./llms"  # local llm cache
DOCKER_INTERNAL_CACHE_DIR="/llms"

# Create local llm cache if doesn't exist
mkdir -p "$LLM_CACHE_DIR"

INPUT_FILE="${1:-mmtu.jsonl}"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e MMTU_HOME="/workspace" \
    --rm \
    --memory="30g" \
    --gpus all \
    mmtu-image \
    bash -c "cd /workspace && python3 inference.py -i $INPUT_FILE -l info self_deploy"