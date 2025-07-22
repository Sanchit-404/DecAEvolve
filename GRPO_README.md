# GRPO-Enhanced LLM-SR 🚀

This implementation integrates **Group Relative Policy Optimization (GRPO)** with **LLM-SR** for continuous improvement of small language models in symbolic regression tasks.

## 🎯 Overview

GRPO allows the language model to **learn from equation evaluation feedback** at each iteration, potentially leading to:
- 📈 Better equation quality over time
- ⚡ Faster convergence to optimal solutions  
- 🎯 Domain-specific adaptation to physics problems
- 💾 Efficient training with small models and LoRA

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GRPO Sampler  │───▶│   Equation       │───▶│   Experience    │
│                 │    │   Evaluator      │    │   Buffer        │
│ • Small LLM     │    │                  │    │                 │
│ • LoRA Training │    │ • Execute eqns   │    │ • Store best    │
│ • Reward Fn     │    │ • Compute scores │    │ • Island mgmt   │
│                 │    │                  │    │                 │
└─────────┬───────┘    └──────────┬───────┘    └─────────┬───────┘
          │                       │                      │
          └───────────────────────┼──────────────────────┘
                                  │
                            ┌─────▼─────┐
                            │    GRPO   │
                            │  Training │
                            │           │
                            │ • Rewards │
                            │ • Policy  │
                            │ • Updates │
                            └───────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_grpo.txt
```

### 2. Test Installation
```bash
python test_grpo_llmsr.py
```

### 3. Run GRPO-Enhanced LLM-SR
```bash
# Basic usage with GRPO
python main.py --use_grpo --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt

# With custom model and parameters
python main.py --use_grpo \
    --grpo_model "HuggingFaceTB/SmolLM-135M-Instruct" \
    --grpo_lr 5e-5 \
    --grpo_update_freq 25 \
    --grpo_batch_size 4 \
    --problem_name oscillator1 \
    --spec_path ./specs/specification_oscillator1_numpy.txt
```

### 4. Run Example
```bash
# Full example with explanations
python example_grpo_llmsr.py

# Test mode for debugging
python example_grpo_llmsr.py --test
```

## 🔧 Configuration

### GRPO Parameters
```python
grpo_config = GRPOConfig(
    model_path="HuggingFaceTB/SmolLM-135M-Instruct",  # Base model
    use_grpo=True,                    # Enable GRPO
    learning_rate=5e-5,               # Conservative LR
    per_device_train_batch_size=4,    # Batch size
    gradient_accumulation_steps=2,     # Effective batch = 4*2=8
    max_prompt_length=512,            # Max input tokens
    max_completion_length=256,        # Max output tokens
    num_generations=8,                # GRPO group size
    update_frequency=25,              # Update every N samples
    use_lora=True,                    # Efficient fine-tuning
    kl_coeff=0.1,                     # KL regularization
    num_train_epochs=1,               # Epochs per update
    bf16=True                         # Mixed precision
)
```

### Recommended Small Models

| Model | Size | Strengths | Use Case |
|-------|------|-----------|----------|
| `HuggingFaceTB/SmolLM-135M-Instruct` | 135M | Fast, instruction-tuned | Testing, quick experiments |
| `microsoft/DialoGPT-small` | 117M | Conversation, reasoning | General equation discovery |
| `Salesforce/codegen-350M-mono` | 350M | Code generation | Complex mathematical expressions |
| `microsoft/CodeT5-small` | 60M | Code understanding | Simple equation structures |

## 🎯 Reward Function Design

The GRPO implementation uses a **multi-component reward function**:

```python
def equation_reward_function(completions, prompts):
    reward = 0.0
    
    # 1. Syntax validity (±2.0)
    reward += syntax_reward(equation)
    
    # 2. Parsimony (simpler = better) (-0.0 to -2.0)
    reward -= complexity_penalty(equation)  
    
    # 3. Physics plausibility (+0.0 to +1.5)
    reward += physics_reward(equation)
    
    # 4. Evaluation score (main signal)
    reward += evaluation_score  # From mathematical fitness
    
    return reward
```

### Key Reward Components:
- **🎯 Evaluation Score**: Primary signal from equation fitness
- **✅ Syntax Validity**: Ensures equations can be executed  
- **📐 Parsimony**: Prefers simpler equations (Occam's razor)
- **⚗️ Physics Plausibility**: Rewards use of relevant variables/functions

## 🔬 How It Works

### 1. **Sampling Phase**
- Model generates equation hypotheses
- Store prompt-completion pairs for training

### 2. **Evaluation Phase** 
- Execute equations on physics data
- Compute mathematical fitness scores
- Extract reward signals

### 3. **GRPO Training Phase** (every N samples)
- Group completions by prompt
- Compute group-relative advantages  
- Optimize policy with KL constraint
- Update model weights via LoRA

### 4. **Iteration**
- Improved model generates better equations
- Cycle continues with enhanced capabilities

## 📊 Expected Benefits

### Compared to Standard LLM-SR:
- 🎯 **Adaptive Learning**: Model improves from feedback
- ⚡ **Faster Convergence**: Fewer iterations to find good equations
- 🎨 **Domain Adaptation**: Learns physics-specific patterns
- 💾 **Efficiency**: Small models + LoRA = low compute cost

### Metrics to Track:
- **Equation Quality**: Best fitness scores over time
- **Convergence Rate**: Iterations to reach target fitness
- **Diversity**: Variety in equation structures discovered
- **Training Stability**: KL divergence from reference model

## 🧪 Experimental Validation

### Test Scenarios:
1. **Oscillator Physics**: Damped harmonic motion
2. **Bacterial Growth**: Population dynamics
3. **Stress-Strain**: Materials science
4. **Custom Datasets**: Your domain-specific problems

### Comparison Protocol:
```bash
# Standard LLM-SR baseline
python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt

# GRPO-enhanced version  
python main.py --use_grpo --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt
```

## 🚧 Troubleshooting

### Common Issues:

**1. CUDA Out of Memory**
```python
# Reduce batch size
grpo_config.per_device_train_batch_size = 2

# Use gradient accumulation
grpo_config.gradient_accumulation_steps = 4

# Use CPU if needed
grpo_config.bf16 = False
```

**2. Slow Training**
```python
# Increase update frequency
grpo_config.update_frequency = 50

# Use smaller model
grpo_config.model_path = "HuggingFaceTB/SmolLM-135M-Instruct"
```

**3. Poor Equation Quality**
```python
# Adjust reward weighting
# Increase evaluation score weight
# Tune KL coefficient

# Try different base model
grpo_config.model_path = "Salesforce/codegen-350M-mono"
```

## 🔮 Future Enhancements

### Potential Improvements:
- **🎭 Multi-Objective Rewards**: Balance accuracy, simplicity, interpretability
- **🌊 Curriculum Learning**: Start with simple equations, progress to complex
- **🔄 Online Learning**: Update model continuously during evaluation
- **📊 Advanced Metrics**: Equation novelty, physical consistency
- **🤖 Ensemble Methods**: Multiple GRPO agents with different specializations

### Research Directions:
- **GRPO Variants**: Experiment with different policy optimization algorithms
- **Reward Engineering**: Domain-specific reward functions for different physics
- **Model Architecture**: Specialized small models for equation generation
- **Transfer Learning**: Pre-train on mathematical corpora, fine-tune with GRPO

## 📚 References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) - DeepSeek-Math: Pushing the Limits of Mathematical Reasoning
- [LLM-SR Paper](https://arxiv.org/abs/2404.18400) - Scientific Equation Discovery via Programming with LLMs
- [TRL Documentation](https://huggingface.co/docs/trl/) - Transformer Reinforcement Learning
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- 🧪 New reward function designs
- 🤖 Support for additional small models  
- 📊 Better evaluation metrics
- 🔧 Performance optimizations
- 📖 Documentation improvements

## 📄 License

This GRPO integration follows the same MIT license as the original LLM-SR project.

---

**Happy equation discovering! 🧮✨** 