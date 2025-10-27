#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=nvidia_geforce_rtx_3090:1

# MCP-enabled Docker run script
# Mounts data_sqlite directory for database access

PHYS_DIR="/home/$(whoami)/MMTU"
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"
INPUT_FILE="${1:-datasets/stratified_dataset.jsonl}"
MODEL="${2:-qwen3-8b-awq}"

echo "============================================"
echo "MCP-ENABLED INFERENCE"
echo "============================================"
echo "Dataset: $INPUT_FILE"
echo "Model: $MODEL"
echo "MCP: ENABLED (adaptive strategy)"
echo "============================================"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -v "$PHYS_DIR/data_sqlite":/workspace/data_sqlite \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e MMTU_HOME="/workspace" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    mmtu-image \
    bash -c "cd /workspace && python3 inference.py -i $INPUT_FILE -l info self_deploy --model $MODEL --mcp-strategy direct_sql"
