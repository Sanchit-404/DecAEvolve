#!/bin/bash

# Test script for running LLM-SR with HuggingFace models

echo "Running LLM-SR with HuggingFace SmolLM-135M..."

# Test with SmolLM-135M (smallest model for quick testing)
python main.py \
    --problem_name oscillator1 \
    --spec_path ./specs/specification_oscillator1_numpy.txt \
    --log_path ./logs/test_smollm_135m \
    --hf_model "HuggingFaceTB/SmolLM-135M-Instruct"

echo "Test completed. Check logs in ./logs/test_smollm_135m"

# Alternative models to try:
echo ""
echo "Other models you can test:"
echo "  SmolLM-360M: HuggingFaceTB/SmolLM-360M-Instruct"
echo "  SmolLM-1.7B: HuggingFaceTB/SmolLM-1.7B-Instruct"
echo "  DistilGPT-2: distilgpt2"
echo ""
echo "Example with different model:"
echo "  python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --hf_model HuggingFaceTB/SmolLM-360M-Instruct" 