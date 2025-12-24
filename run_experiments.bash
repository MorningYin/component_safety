#!/bin/bash

# Multi-model experiment runner for component safety redesign
# This script executes the pipeline for Llama-3, Qwen-7B, and Qwen-14B

# Get script directory and set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="/root/LLM-Safety:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/root/autodl-tmp/hf_cache"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/hf_cache"
export TRANSFORMERS_CACHE="/root/autodl-tmp/hf_cache"
export HF_DATASETS_CACHE="/root/autodl-tmp/hf_cache"
cd "$SCRIPT_DIR"

echo "🚀 Starting multi-model experiments..."

# 1. Llama-3-8b-it
echo "------------------------------------------------"
echo "Running Llama-3-8b-it"
python main.py --config configs/llama3-8b.yaml

# 2. Qwen-7b-chat
echo "------------------------------------------------"
echo "Running Qwen-7b-chat..."
python main.py --config configs/qwen7b.yaml

# 3. Qwen-14b-chat
echo "------------------------------------------------"
echo "Running Qwen-14b-chat..."
python main.py --config configs/qwen14b.yaml

echo "------------------------------------------------"
echo "✨ All experiments finished!"
