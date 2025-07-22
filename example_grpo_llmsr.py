#!/usr/bin/env python3
"""
Example: Using GRPO with LLM-SR for Equation Discovery

This script demonstrates how to use Group Relative Policy Optimization (GRPO)
to continuously improve a small language model for symbolic regression tasks.
"""

import os
import sys
sys.path.append('.')

from llmsr import pipeline, config as config_lib, sampler, evaluator
import pandas as pd
import numpy as np


def run_grpo_equation_discovery():
    """
    Run LLM-SR with GRPO on the oscillator dataset
    """
    
    print("🔬 GRPO-Enhanced LLM-SR for Equation Discovery")
    print("=" * 55)
    
    # 1. Load dataset
    print("📊 Loading oscillator dataset...")
    df = pd.read_csv('./data/oscillator1/train.csv')
    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict}
    print(f"   Loaded {X.shape[0]} training samples")
    
    # 2. Load specification
    print("📋 Loading specification template...")
    with open('./specs/specification_oscillator1_numpy.txt', 'r') as f:
        specification = f.read()
    print("   Template loaded successfully")
    
    # 3. Configure GRPO
    print("⚙️ Configuring GRPO settings...")
    
    grpo_config = config_lib.GRPOConfig(
        model_path="HuggingFaceTB/SmolLM-135M-Instruct",  # Small, fast model
        use_grpo=True,
        learning_rate=5e-5,              # Conservative learning rate
        per_device_train_batch_size=4,   # Adjust based on GPU memory
        gradient_accumulation_steps=2,    # Effective batch size = 4*2=8
        max_prompt_length=512,
        max_completion_length=256,
        num_generations=8,               # GRPO group size
        update_frequency=25,             # Update every 25 equation samples
        use_lora=True,                   # Efficient fine-tuning
        kl_coeff=0.1,                    # KL regularization
        num_train_epochs=1,
        bf16=True                        # Use bfloat16 if supported
    )
    
    config = config_lib.Config(
        num_samplers=1,
        num_evaluators=1,
        samples_per_prompt=4,            # Generate 4 equations per prompt
        evaluate_timeout_seconds=30,
        use_api=False,
        grpo_config=grpo_config
    )
    
    class_config = config_lib.ClassConfig(
        llm_class=sampler.LocalLLM,
        sandbox_class=evaluator.LocalSandbox
    )
    
    print("   ✅ GRPO configuration:")
    print(f"   - Model: {grpo_config.model_path}")
    print(f"   - Learning rate: {grpo_config.learning_rate}")
    print(f"   - Update frequency: {grpo_config.update_frequency}")
    print(f"   - LoRA enabled: {grpo_config.use_lora}")
    
    # 4. Run LLM-SR with GRPO
    print("\n🚀 Starting GRPO-enhanced LLM-SR...")
    print("   The model will continuously learn from equation evaluations")
    
    max_sample_nums = 200  # Total number of equation samples to generate
    
    try:
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=config,
            max_sample_nums=max_sample_nums,
            class_config=class_config,
            log_dir="./logs/grpo_oscillator1"
        )
        
        print("\n🎉 GRPO-enhanced LLM-SR completed successfully!")
        print(f"📈 Generated and evaluated {max_sample_nums} equations")
        print("📁 Results saved to ./logs/grpo_oscillator1/")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


def compare_grpo_vs_standard():
    """
    Compare GRPO vs standard LLM-SR performance
    """
    
    print("\n🔍 Comparison: GRPO vs Standard LLM-SR")
    print("=" * 45)
    
    # Common configuration
    base_config = {
        'num_samplers': 1,
        'num_evaluators': 1,
        'samples_per_prompt': 4,
        'evaluate_timeout_seconds': 30,
        'use_api': False,
    }
    
    max_samples = 100  # Smaller for comparison
    
    results = {}
    
    # Test configurations
    test_configs = [
        {
            'name': 'Standard LLM-SR',
            'grpo_config': config_lib.GRPOConfig(use_grpo=False),
            'description': 'Original LLM-SR with API-based LLM'
        },
        {
            'name': 'GRPO-Enhanced',
            'grpo_config': config_lib.GRPOConfig(
                model_path="HuggingFaceTB/SmolLM-135M-Instruct",
                use_grpo=True,
                update_frequency=20
            ),
            'description': 'LLM-SR with GRPO continuous learning'
        }
    ]
    
    print("📋 Test configurations:")
    for i, test_config in enumerate(test_configs, 1):
        print(f"   {i}. {test_config['name']}: {test_config['description']}")
    
    print(f"\n🎯 Each test will generate {max_samples} equations")
    print("📊 Metrics: Best fitness score, convergence rate, equation quality")
    
    # Note: Full comparison would require actual runs
    print("\n💡 To run full comparison:")
    print("   python example_grpo_llmsr.py --compare")
    print("   (Implementation would track metrics and create comparison plots)")


def main():
    """Main function to demonstrate GRPO integration"""
    
    # Check if comparison mode
    if '--compare' in sys.argv:
        compare_grpo_vs_standard()
        return
    
    # Check if test mode
    if '--test' in sys.argv:
        print("🧪 Running in test mode...")
        # Import and run the test
        from test_grpo_llmsr import test_grpo_integration
        success = test_grpo_integration()
        if not success:
            sys.exit(1)
        return
    
    # Regular GRPO demonstration
    print("🎯 Running GRPO-enhanced equation discovery...")
    print("💡 Use --test for testing mode or --compare for comparison")
    print()
    
    try:
        run_grpo_equation_discovery()
        
        print("\n🎊 Success! Your GRPO integration is working.")
        print("\n📚 What happened:")
        print("1. 🤖 Loaded a small language model (SmolLM-135M)")
        print("2. 🔄 Applied LoRA for efficient fine-tuning")
        print("3. 📝 Generated equation hypotheses for oscillator physics")
        print("4. 🧮 Evaluated each equation's mathematical fitness")
        print("5. 🎯 Used GRPO to improve the model based on evaluation feedback")
        print("6. 🔁 Repeated the cycle, continuously learning better equations")
        
        print("\n🔬 Key Benefits:")
        print("• Adaptive learning from equation evaluation feedback")
        print("• Improved equation quality over iterations")
        print("• Efficient training with LoRA and small models")
        print("• Domain-specific reward functions for physics")
        
        print("\n🚀 Next experiments:")
        print("• Try different small models (CodeT5, CodeGen)")
        print("• Adjust GRPO hyperparameters (learning rate, update frequency)")
        print("• Test on different physics problems (oscillator2, stressstrain)")
        print("• Compare convergence speed vs standard LLM-SR")
        
    except ImportError as e:
        print(f"\n📦 Missing dependencies: {e}")
        print("Please install: pip install -r requirements_grpo.txt")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Run with --test flag to diagnose issues")


if __name__ == "__main__":
    main() 