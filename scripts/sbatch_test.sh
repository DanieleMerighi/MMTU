#!/bin/bash

# Input file (your .jsonl dataset)
INPUT_FILE="${1:-mmtu.jsonl}"  # Usa primo argomento o default mmtu.jsonl

# Model name
MODEL="${2:-llama-3.1-8b-awq}"  # Usa secondo argomento o default llama-3.1-8b-awq

echo "Submitting job with:"
echo "  Dataset: $INPUT_FILE"
echo "  Model: $MODEL"

# Submit job to SLURM
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 \
    run_docker.sh \
    "$INPUT_FILE" \
    "$MODEL"
