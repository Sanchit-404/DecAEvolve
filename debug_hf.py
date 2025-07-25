#!/usr/bin/env python3
"""
Debug script to see exactly what HuggingFace model generates
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmsr.sampler import HuggingFaceLLM, _extract_body
from llmsr import config as config_lib

def debug_generation():
    """Debug what the model actually generates"""
    
    config = config_lib.Config()
    
    # Create HF model
    hf_llm = HuggingFaceLLM(
        samples_per_prompt=2,
        model_name="microsoft/DialoGPT-medium",
        batch_inference=True
    )
    
    # Test prompt
    test_prompt = '''@equation.evolve
def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Simple test equation """'''
    
    print("=== INPUT PROMPT ===")
    full_prompt = '\n'.join([hf_llm._instruction_prompt, test_prompt])
    print(full_prompt)
    print("\n" + "="*50 + "\n")
    
    # Generate raw output (bypass trimming)
    print("=== RAW MODEL OUTPUT ===")
    hf_llm._trim = False  # Disable trimming to see raw output
    raw_samples = hf_llm.draw_samples(test_prompt, config)
    
    for i, sample in enumerate(raw_samples):
        print(f"Raw Sample {i+1}:")
        print(repr(sample))  # Use repr to see whitespace/newlines
        print("Actual text:")
        print(sample)
        print("-" * 40)
    
    print("\n=== TESTING EXTRACTION ===")
    for i, sample in enumerate(raw_samples):
        print(f"Sample {i+1} extraction:")
        extracted = _extract_body(sample, config)
        print(f"Extracted: {repr(extracted)}")
        print("Extracted text:")
        print(extracted)
        print("-" * 40)

if __name__ == "__main__":
    debug_generation() 