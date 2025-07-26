#!/usr/bin/env python3
"""
Test script for GRPO integration with LLMSR.
This demonstrates how to use GRPO training at each iteration.
"""

import os
import numpy as np
import pandas as pd
import torch

from llmsr import pipeline
from llmsr import config
from llmsr import sampler
from llmsr import evaluator


def test_grpo_integration():
    """Test GRPO integration with LLMSR pipeline."""
    
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
        grpo_learning_rate=2e-5,
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
    
    print("Starting GRPO-enabled LLMSR experiment...")
    print(f"Model: {config_obj.hf_model}")
    print(f"GRPO batch size: {config_obj.grpo_batch_size}")
    print(f"Learning rate: {config_obj.grpo_learning_rate}")
    print(f"Samples per prompt: {config_obj.samples_per_prompt}")
    
    # Run pipeline with GRPO
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config_obj,
        max_sample_nums=20,  # Small number for testing
        class_config=class_config,
        log_dir="./logs/grpo_test",
    )
    
    print("GRPO integration test completed!")


if __name__ == "__main__":
    test_grpo_integration() 