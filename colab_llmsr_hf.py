#!/usr/bin/env python3
"""
Standalone LLM-SR with HuggingFace integration for Colab
This script contains the complete HuggingFace LLM implementation matching LocalLLM structure
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Collection
import time
import numpy as np

class HuggingFaceLLM:
    def __init__(self, samples_per_prompt: int, model_name: str = None, 
                 batch_inference: bool = True, trim: bool = True) -> None:
        """
        Hugging Face model for equation generation - matches LocalLLM structure exactly.
        
        Args:
            samples_per_prompt: Number of samples to generate per prompt
            model_name: Hugging Face model identifier
            batch_inference: Use batch inference when sampling
            trim: Whether to trim the response to extract equation body
        """
        self._samples_per_prompt = samples_per_prompt
        
        # Default model if none specified  
        if model_name is None:
            model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Default to larger model for Colab
            
        self.model_name = model_name
        self._batch_inference = batch_inference
        self._trim = trim
        
        # Instruction prompt - exactly like LocalLLM
        instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n")
        self._instruction_prompt = instruction_prompt
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def draw_samples(self, prompt: str, use_api: bool = False) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if use_api:
            return self._draw_samples_api(prompt)
        else:
            return self._draw_samples_local(prompt)

    def _draw_samples_local(self, prompt: str) -> Collection[str]:
        """Local sampling method - matches LocalLLM structure exactly."""
        # instruction - exactly like LocalLLM
        prompt = '\n'.join([self._instruction_prompt, prompt])
        
        while True:
            try:
                all_samples = []
                # response from llm model
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [self._extract_body(sample) for sample in all_samples]
                
                return all_samples
            except Exception as e:
                print(f"Error in sampling: {e}")
                continue

    def _draw_samples_api(self, prompt: str) -> Collection[str]:
        """API sampling method - placeholder for consistency."""
        # Just call local method for now
        return self._draw_samples_local(prompt)

    def _do_request(self, content: str) -> str:
        """Generate response using HuggingFace model - matches LocalLLM _do_request signature."""
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        # Generate using HuggingFace model
        inputs = self.tokenizer(content, return_tensors="pt", truncation=True, max_length=1024)
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if self._batch_inference:
                # Generate multiple samples at once
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_return_sequences=repeat_prompt,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode all outputs
                responses = []
                for output in outputs:
                    generated_text = self.tokenizer.decode(
                        output[inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    responses.append(generated_text.strip())
                
                return responses
            else:
                # Generate single sample
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode output
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                return generated_text.strip()

    def _extract_body(self, sample: str) -> str:
        """
        Extract the function body from a response sample, removing any preceding descriptions
        and the function signature. Preserves indentation.
        """
        lines = sample.splitlines()
        func_body_lineno = 0
        find_def_declaration = False
        
        for lineno, line in enumerate(lines):
            # find the first 'def' program statement in the response
            if line[:3] == 'def':
                func_body_lineno = lineno
                find_def_declaration = True
                break
        
        if find_def_declaration:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
            
            return code
        
        return sample

# Test the implementation
def test_huggingface_llm():
    """Test function for the HuggingFace LLM"""
    print("Testing HuggingFace LLM with larger model...")
    
    # Test with different models - uncomment the one you want to test
    models_to_test = [
        # "meta-llama/Llama-3.1-8B-Instruct",      # Requires HF authentication
        # "unsloth/Meta-Llama-3.1-8B-bnb-4bit",    # Requires CUDA
        "Qwen/Qwen2.5-7B-Instruct",                # Good option for Colab
        # "microsoft/DialoGPT-large",              # Alternative option
        # "HuggingFaceTB/SmolLM-1.7B-Instruct",    # Smaller fallback
    ]
    
    for model_name in models_to_test:
        try:
            print(f"\n=== Testing {model_name} ===")
            hf_llm = HuggingFaceLLM(
                samples_per_prompt=3,
                model_name=model_name,
                batch_inference=True
            )
            
            # Test prompt (oscillator equation)
            test_prompt = '''@equation.evolve
def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for acceleration in a damped nonlinear oscillator

    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """'''
            
            print("Generating samples...")
            start_time = time.time()
            samples = hf_llm.draw_samples(test_prompt)
            elapsed = time.time() - start_time
            
            print(f"Generated {len(samples)} samples in {elapsed:.2f}s:")
            for i, sample in enumerate(samples):
                print(f"\nSample {i+1}:")
                print("-" * 60)
                print(sample)
                print("-" * 60)
            
            # Test successful, break
            print(f"\n✅ {model_name} working successfully!")
            break
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_huggingface_llm() 