"""
Test script for GRPO integration with LLM-SR
"""

import os
import sys
import pandas as pd
import numpy as np
import torch

# Add project root to path
sys.path.append('.')

from llmsr import pipeline, config as config_lib, sampler, evaluator


def test_grpo_integration():
    """Test the GRPO integration with a small model on oscillator data"""
    
    print("🚀 Testing GRPO Integration with LLM-SR")
    print("=" * 50)
    
    # 1. Check if required packages are installed
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from peft import LoraConfig
        print("✅ Required packages installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install: pip install transformers trl peft accelerate bitsandbytes")
        return False
    
    # 2. Load test dataset
    try:
        df = pd.read_csv('./data/oscillator1/train.csv')
        data = np.array(df)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        data_dict = {'inputs': X, 'outputs': y}
        dataset = {'data': data_dict}
        print(f"✅ Loaded oscillator1 dataset: {X.shape[0]} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # 3. Load specification
    try:
        with open('./specs/specification_oscillator1_numpy.txt', 'r') as f:
            specification = f.read()
        print("✅ Loaded specification template")
    except Exception as e:
        print(f"❌ Failed to load specification: {e}")
        return False
    
    # 4. Configure GRPO
    grpo_config = config_lib.GRPOConfig(
        model_path="HuggingFaceTB/SmolLM-135M-Instruct",  # Small model for testing
        use_grpo=True,
        learning_rate=5e-5,
        per_device_train_batch_size=2,  # Small batch for testing
        gradient_accumulation_steps=1,
        max_prompt_length=256,
        max_completion_length=128,
        num_generations=4,
        update_frequency=10,  # Update frequently for testing
        use_lora=True,
        kl_coeff=0.1,
        num_train_epochs=1
    )
    
    config = config_lib.Config(
        num_samplers=1,
        num_evaluators=1,
        samples_per_prompt=2,  # Small for testing
        evaluate_timeout_seconds=30,
        use_api=False,
        grpo_config=grpo_config
    )
    
    class_config = config_lib.ClassConfig(
        llm_class=sampler.LocalLLM, 
        sandbox_class=evaluator.LocalSandbox
    )
    
    print("✅ Configured GRPO settings")
    
    # 5. Test GRPO sampler creation
    try:
        print("🔄 Testing GRPO sampler initialization...")
        
        # This will test if our GRPO sampler can be created
        from llmsr.grpo_sampler import GRPOSampler, GRPOEquationLLM
        
        # Test the GRPO LLM creation with a very small model
        test_grpo_llm = GRPOEquationLLM(
            model_path="HuggingFaceTB/SmolLM-135M-Instruct",
            samples_per_prompt=2,
            grpo_config=grpo_config.__dict__,
            use_lora=True,
            update_frequency=5
        )
        
        print("✅ GRPO LLM initialized successfully")
        
        # Test sample generation
        test_prompt = """
def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    \"\"\" Mathematical function for acceleration in a damped nonlinear oscillator \"\"\"
"""
        
        print("🔄 Testing sample generation...")
        samples = test_grpo_llm._generate_samples(test_prompt)
        print(f"✅ Generated {len(samples)} samples")
        
        # Test reward computation
        print("🔄 Testing reward computation...")
        prompts = [test_prompt] * len(samples)
        rewards = test_grpo_llm.equation_reward_function(samples, prompts)
        print(f"✅ Computed rewards: {rewards}")
        
    except Exception as e:
        print(f"❌ GRPO sampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Test integration with main pipeline (limited run)
    try:
        print("🔄 Testing integration with main pipeline...")
        
        # Run pipeline with very limited iterations
        max_sample_nums = 20  # Very small for testing
        
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=config,
            max_sample_nums=max_sample_nums,
            class_config=class_config,
            log_dir="./test_logs"
        )
        
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! GRPO integration is working.")
    print("\n📋 Summary:")
    print("- GRPO LLM initialization: ✅")
    print("- Sample generation: ✅")
    print("- Reward computation: ✅") 
    print("- Pipeline integration: ✅")
    
    return True


def test_model_recommendations():
    """Test different small model recommendations"""
    
    models_to_test = [
        "HuggingFaceTB/SmolLM-135M-Instruct",  # Very small for testing
        "microsoft/DialoGPT-small",           # Alternative small model
        # "Salesforce/codegen-350M-mono",      # Code-focused (commented out for quick test)
    ]
    
    print("\n🧪 Testing Model Recommendations")
    print("=" * 40)
    
    for model_path in models_to_test:
        print(f"\n🔄 Testing model: {model_path}")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Test if model can be loaded
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float32,  # Use float32 for CPU testing
                device_map=None
            )
            
            # Get model size
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  ✅ Model loaded successfully")
            print(f"  📊 Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
            print(f"  🔤 Vocab size: {len(tokenizer)}")
            
            # Test tokenization of equation prompt
            test_text = "def equation(x, v, params): return params[0] * x + params[1] * v"
            tokens = tokenizer.encode(test_text)
            print(f"  🧮 Test equation tokens: {len(tokens)}")
            
        except Exception as e:
            print(f"  ❌ Failed to load {model_path}: {e}")


if __name__ == "__main__":
    print("🧪 GRPO-LLM-SR Integration Test Suite")
    print("====================================")
    
    # Test model loading first
    test_model_recommendations()
    
    # Then test full integration
    success = test_grpo_integration()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Run with larger datasets: python main.py --grpo_enabled")
        print("2. Experiment with different models")
        print("3. Tune GRPO hyperparameters")
        print("4. Monitor training progress with logging")
    else:
        print("\n🔧 Troubleshooting needed. Check error messages above.") 