#!/usr/bin/env python3
"""
Test script for HuggingFace LLM integration with LLM-SR
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmsr import sampler, config as config_lib

def test_huggingface_llm():
    """Test the HuggingFaceLLM class with a simple equation generation task"""
    
    print("Testing HuggingFace LLM integration...")
    
    # Create a simple config
    config = config_lib.Config()
    
    # Initialize the HuggingFace LLM
    hf_llm = sampler.HuggingFaceLLM(
        samples_per_prompt=2,
        model_name="HuggingFaceTB/SmolLM-135M-Instruct",
        batch_inference=True
    )
    
    # Test prompt (similar to what LLM-SR uses)
    test_prompt = '''"""
Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity. 
"""

import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    x, v = inputs[:,0], inputs[:,1]
    
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(x, v, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss

@equation.evolve
def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """'''
    
    try:
        # Generate samples
        print("Generating equation samples...")
        samples = hf_llm.draw_samples(test_prompt, config)
        
        print(f"\nSuccessfully generated {len(samples)} samples:")
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:")
            print("-" * 50)
            print(sample)
            print("-" * 50)
            
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_huggingface_llm()
    if success:
        print("\n✅ HuggingFace LLM integration test passed!")
        print("You can now run LLM-SR with: python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/test_hf")
    else:
        print("\n❌ HuggingFace LLM integration test failed!")
        sys.exit(1) 