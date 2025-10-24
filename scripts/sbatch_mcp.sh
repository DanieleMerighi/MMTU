#!/bin/bash

# SLURM submission script for MCP-enabled inference
# Usage:
#   ./sbatch_mcp.sh                                    # Use defaults
#   ./sbatch_mcp.sh datasets/stratified_dataset.jsonl # Custom dataset
#   ./sbatch_mcp.sh datasets/test.jsonl qwen3-8b-awq  # Custom dataset + model

# Input file (your .jsonl dataset)
INPUT_FILE="${1:-datasets/stratified_dataset.jsonl}"

# Model name (default: qwen3-8b-awq - recommended for MCP)
MODEL="${2:-qwen3-8b-awq}"

echo "============================================"
echo "Submitting MCP inference job to SLURM"
echo "============================================"
echo "  Dataset: $INPUT_FILE"
echo "  Model: $MODEL"
echo "  MCP Strategy: Adaptive (488/1842 cells thresholds)"
echo "============================================"
echo ""

# Submit job to SLURM
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 \
    scripts/run_docker_mcp.sh \
    "$INPUT_FILE" \
    "$MODEL"

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$(whoami)"
echo ""
echo "Expected output file:"
echo "  inference_results/${INPUT_FILE##*/%.jsonl}.${MODEL}-mcp.result.jsonl"
echo ""
