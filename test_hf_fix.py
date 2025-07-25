#!/usr/bin/env python3
"""
Test script to verify HuggingFace integration works with rope_scaling fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmsr.sampler import HuggingFaceLLM
from llmsr import config as config_lib

def test_hf_models():
    """Test different HuggingFace models to see which ones work"""
    
    # Models to test in order of preference
    test_models = [
        "microsoft/DialoGPT-medium",
        "gpt2-medium",
        "meta-llama/Llama-3.1-8B-Instruct",  # This should trigger the rope_scaling fix
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    
    config = config_lib.Config()
    
    for model_name in test_models:
        try:
            print(f"\n=== Testing {model_name} ===")
            
            # Test model loading
            hf_llm = HuggingFaceLLM(
                samples_per_prompt=2,
                model_name=model_name,
                batch_inference=True
            )
            
            print(f"✅ Model {model_name} loaded successfully!")
            
            # Test simple generation
            test_prompt = '''@equation.evolve
def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Simple test equation """'''
            
            print("Testing sample generation...")
            samples = hf_llm.draw_samples(test_prompt, config)
            print(f"✅ Generated {len(samples)} samples successfully!")
            
            # Show one sample
            if samples:
                print(f"Sample output: {samples[0][:100]}...")
            
            # Test successful, break
            print(f"\n🎉 {model_name} working perfectly!")
            break
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue
    else:
        print("\n💥 All models failed!")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing HuggingFace integration with rope_scaling fix...")
    success = test_hf_models()
    
    if success:
        print("\n✅ HuggingFace integration is working! You can now run the main pipeline.")
    else:
        print("\n❌ HuggingFace integration still has issues. Check your environment.") 