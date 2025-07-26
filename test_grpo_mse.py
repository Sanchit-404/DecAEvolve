#!/usr/bin/env python3
"""
Test script for GRPO integration with MSE-based rewards in LLMSR.
This demonstrates how the system uses actual equation evaluation MSE as rewards for GRPO training.
"""

import os
import numpy as np
import pandas as pd
import torch

from llmsr import pipeline
from llmsr import config
from llmsr import sampler
from llmsr import evaluator


def test_grpo_mse_rewards():
    """Test GRPO integration with proper MSE-based rewards."""
    
    print("=" * 60)
    print("Testing GRPO with MSE-based rewards")
    print("=" * 60)
    
    # Configuration for GRPO training
    class_config = config.ClassConfig(
        llm_class=sampler.GRPOHuggingFaceLLM,  # Use GRPO-enabled LLM
        sandbox_class=evaluator.LocalSandbox
    )
    
    # Config with GRPO enabled
    config_obj = config.Config(
        use_api=False,
        hf_model="microsoft/DialoGPT-medium",  # Small model for testing
        use_grpo=True,
        grpo_batch_size=4,  # Train every 4 samples
        grpo_learning_rate=1e-5,  # Lower learning rate for stability
        samples_per_prompt=4,  # Generate 4 samples per iteration
        num_samplers=1,
        num_evaluators=1
    )
    
    # Load problem specification
    spec_path = "./specs/specification_oscillator1_numpy.txt"
    with open(spec_path, encoding="utf-8") as f:
        specification = f.read()
    
    # Load dataset
    problem_name = "oscillator1"
    df = pd.read_csv(f'./data/{problem_name}/train.csv')
    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict} 
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Target equation should predict acceleration from position (x) and velocity (v)")
    print()
    
    print("Configuration:")
    print(f"  Model: {config_obj.hf_model}")
    print(f"  GRPO batch size: {config_obj.grpo_batch_size}")
    print(f"  Learning rate: {config_obj.grpo_learning_rate}")
    print(f"  Samples per prompt: {config_obj.samples_per_prompt}")
    print()
    
    print("Starting GRPO-enabled LLMSR with PURE NMSE rewards...")
    print("The system will:")
    print("  1. Generate equation hypotheses")
    print("  2. Evaluate each by fitting parameters and calculating NMSE")
    print("  3. Convert NMSE to rewards (lower NMSE = higher reward)")
    print("  4. Train the model with GRPO using ONLY these pure NMSE rewards")
    print("  5. NO other heuristic rewards are used - only equation performance")
    print()
    
    # Run pipeline with GRPO
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config_obj,
        max_sample_nums=12,  # 3 iterations of 4 samples each
        class_config=class_config,
        log_dir="./logs/grpo_mse_test",
    )
    
    print()
    print("=" * 60)
    print("GRPO with MSE-based rewards test completed!")
    print("Check the logs for reward values and training progress.")
    print("=" * 60)


if __name__ == "__main__":
    test_grpo_mse_rewards() 