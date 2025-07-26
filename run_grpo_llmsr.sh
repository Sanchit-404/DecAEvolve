#!/bin/bash

# Run LLMSR with GRPO training enabled
# This script demonstrates how to run the system with GRPO training at each iteration

echo "Starting LLMSR with GRPO training..."

# Install required dependencies if not already installed
echo "Installing GRPO dependencies..."
pip install trl>=0.14.0 peft>=0.14.0 datasets>=3.2.0 accelerate>=1.2.1 bitsandbytes>=0.45.2

# Run LLMSR with GRPO enabled
python main.py \
    --spec_path ./specs/specification_oscillator1_numpy.txt \
    --log_path ./logs/grpo_oscillator1 \
    --problem_name oscillator1 \
    --hf_model "microsoft/DialoGPT-medium" \
    --use_grpo True \
    --grpo_batch_size 4 \
    --grpo_learning_rate 2e-5 \
    --run_id 1

echo "GRPO training completed!" 