#!/bin/bash

# Input file (your .jsonl dataset)
INPUT_FILE="${1:-mmtu.jsonl}"  # Usa primo argomento o default mmtu.jsonl

# Submit job to SLURM
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 \
    run_docker.sh \
    "$INPUT_FILE"
