# LLM-SR with GRPO Integration

## Overview

This document provides a comprehensive guide to the GRPO (Group Relative Policy Optimization) integration in the LLM-SR (Large Language Model Symbolic Regression) framework. GRPO enables online reinforcement learning from human feedback (RLHF) to improve equation discovery performance over time.

## Table of Contents

- [What is GRPO?](#what-is-grpo)
- [GRPO vs Standard LLM-SR](#grpo-vs-standard-llm-sr)
- [Architecture Overview](#architecture-overview)
- [Code Walkthrough](#code-walkthrough)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## What is GRPO?

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm that trains language models by comparing multiple completions for the same prompt. Unlike PPO which learns from individual samples, GRPO learns from **relative comparisons** between multiple generations.

### Key Benefits:
- **Online Learning**: Model improves during the search process
- **Sample Efficiency**: Reuses LLM-SR samples for training (no double generation)
- **Comparative Learning**: Learns to prefer better equations over worse ones
- **Scientific Focus**: Trained specifically on mathematical equation discovery

## GRPO vs Standard LLM-SR

| Aspect | Standard LLM-SR | GRPO-Enhanced LLM-SR |
|--------|----------------|---------------------|
| **Model** | Static pre-trained model | Dynamically fine-tuned model |
| **Learning** | No learning during search | Learns from equation performance |
| **Sample Usage** | Generate → Evaluate → Discard | Generate → Evaluate → Train → Improve |
| **Performance** | Fixed capabilities | Improving capabilities over time |
| **Memory** | Lower GPU usage | Higher GPU usage (LoRA training) |

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Get Prompt    │───▶│  Generate 4      │───▶│  Evaluate &     │
│  (from buffer)  │    │  LLM-SR Samples  │    │  Compute NMSE   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                                │
         │              ┌──────────────────┐              │
         │              │  Collect Samples │◀─────────────┘
         │              │  & Rewards for   │
         │              │  GRPO Training   │
         │              └──────────────────┘
         │                        │
         │                        ▼
         │              ┌──────────────────┐    ┌─────────────────┐
         │              │  GRPO Training   │───▶│  Update Model   │
         │              │  (Every N batch) │    │  (LoRA weights) │
         │              └──────────────────┘    └─────────────────┘
         │                                                │
         └────────────────────────────────────────────────┘
```

## Code Walkthrough

### 1. GRPO Model Initialization

**File**: `llmsr/sampler.py:637-675`

```python
class GRPOHuggingFaceLLM(HuggingFaceLLM):
    def __init__(self, samples_per_prompt: int, model_name: str = None, 
                 learning_rate: float = 2e-5):
        super().__init__(samples_per_prompt, model_name, batch_inference, trim)
        
        # Setup LoRA for efficient training
        self._setup_lora()
        
        # Setup GRPO trainer
        self._setup_grpo_trainer(learning_rate=learning_rate)
```

**Key Components**:
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with minimal parameters
- **GRPO Configuration**: Training hyperparameters and batch settings
- **Model Wrapping**: Converts HuggingFace model for GRPO training

### 2. Sample Collection & Reward Computation

**File**: `llmsr/sampler.py:148-210`

```python
class GRPOSampler(Sampler):
    def sample(self, **kwargs):
        while True:
            prompt = self._database.get_prompt()  # Get evolutionary prompt
            samples = self._llm.draw_samples(prompt.code, self.config)  # Generate equations
            
            # Collect samples and evaluate them
            batch_data = []
            for sample in samples:
                # Store sample metadata for GRPO
                batch_data.append({
                    'prompt': prompt.code,
                    'completion': sample,
                    'sample_key': f"sample_{cur_global_sample_nums}",
                    # ... other metadata
                })
                
                # Evaluate sample to get NMSE score
                chosen_evaluator.analyse(sample, ...)
            
            # Trigger GRPO training when batch is full
            self._collect_rewards_and_train(batch_data, **kwargs)
```

**Process**:
1. **Generation**: Generate 4 equation samples per prompt
2. **Evaluation**: Test each equation on scientific datasets
3. **Reward Conversion**: Convert NMSE scores to GRPO rewards: `reward = exp(-mse)`
4. **Data Collection**: Store prompt-completion-reward triplets

### 3. Reward Computation Logic

**File**: `llmsr/sampler.py:218-254`

```python
def _collect_rewards_and_train(self, batch_data, **kwargs):
    for data in batch_data:
        sample_key = data['sample_key']
        if sample_key in self.sample_scores:
            score = self.sample_scores[sample_key]  # Score from evaluator
            if score is not None:
                mse = -score  # Convert from negative MSE to positive MSE
                if mse > 0:
                    reward = np.exp(-mse)  # Exponential reward scaling
                else:
                    reward = 1.0  # Perfect fit
                data['reward'] = float(reward)
            else:
                data['reward'] = 0.0  # Failed evaluation
        else:
            data['reward'] = 0.0  # No score available
```

**Reward Mapping**:
- **High-quality equations** (low MSE) → **High rewards** (close to 1.0)
- **Poor equations** (high MSE) → **Low rewards** (close to 0.0)
- **Failed equations** (syntax errors, etc.) → **Zero reward** (0.0)

### 4. GRPO Training Process

**File**: `llmsr/sampler.py:709-796`

```python
def train_with_grpo(self, training_data, evaluators=None, database=None):
    # Group samples by prompt for comparative learning
    prompt_groups = {}
    for data in training_data:
        prompt = data['prompt']
        if prompt not in prompt_groups:
            prompt_groups[prompt] = []
        prompt_groups[prompt].append(data)
    
    # Prepare dataset with existing samples (no re-generation!)
    dataset_entries = []
    for prompt, group in prompt_groups.items():
        group_completions = [item['completion'] for item in group]
        group_rewards = [item['reward'] for item in group]
        
        dataset_entries.append({
            'prompt': prompt,
            'completions': group_completions,
            'rewards': group_rewards
        })
    
    # Create flattened dataset for GRPO
    train_dataset = Dataset.from_dict({
        'prompt': flattened_prompts,
        'completion': flattened_completions,
        'reward': flattened_rewards
    })
    
    # Train with GRPO
    self.grpo_trainer = GRPOTrainer(
        model=self.model,
        reward_funcs=[reward_function],
        train_dataset=train_dataset,
        # ...
    )
    self.grpo_trainer.train()
```

**Key Features**:
- **Sample Reuse**: Uses existing LLM-SR samples (no double generation)
- **Grouped Learning**: Compares multiple completions per prompt
- **Efficient Training**: Only updates LoRA weights, not full model

### 5. Handling Edge Cases

**Empty/Failed Equations** (`sampler.py:750-752`):
```python
# If not enough valid samples, pad with fallbacks
while len(group_completions) < self.grpo_config.num_generations:
    if group_rewards:
        # Duplicate best sample
        best_idx = np.argmax(group_rewards)
        group_completions.append(group_completions[best_idx])
        group_rewards.append(group_rewards[best_idx])
    else:
        # All samples failed → use simple fallback
        group_completions.append("    return 0")
        group_rewards.append(0.0)
```

**Equation Extraction** (`sampler.py:257-307`):
```python
def _extract_body(sample: str, config: config_lib.Config) -> str:
    lines = sample.splitlines()
    for lineno, line in enumerate(lines):
        if line[:3] == 'def':  # Find function definition
            # Extract everything after 'def' line
            return '\n'.join(lines[lineno + 1:])
    
    # If no 'def' found, return original (likely to fail evaluation)
    return sample
```

## Configuration

### GRPO-Specific Parameters

**File**: `llmsr/config.py`

```python
@dataclasses.dataclass
class Config:
    # GRPO Settings
    use_grpo: bool = False
    grpo_batch_size: int = 4          # Train every N samples
    grpo_learning_rate: float = 2e-5  # LoRA learning rate
    
    # Model Settings
    hf_model: str = "microsoft/DialoGPT-medium"
    samples_per_prompt: int = 4       # Must match grpo_batch_size
```

**GRPO Training Configuration** (`sampler.py:676-704`):
```python
self.grpo_config = GRPOConfig(
    output_dir="./grpo_checkpoints",
    learning_rate=learning_rate,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=512,
    max_completion_length=256,
    num_generations=4,              # Must match samples_per_prompt
    optim="adamw_8bit",            # Memory-efficient optimizer
    num_train_epochs=1,
    greater_is_better=True,        # Higher rewards are better
)
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU Memory** | 8GB | 16GB+ |
| **System RAM** | 16GB | 32GB+ |
| **Storage** | 10GB | 50GB+ |
| **GPU** | RTX 3060 | RTX 4080+ |

## Usage Examples

### Basic GRPO Training

```bash
# Small model for testing
python main.py \
    --spec_path ./specs/specification_oscillator1_numpy.txt \
    --use_grpo True \
    --grpo_batch_size 4 \
    --grpo_learning_rate 2e-5 \
    --hf_model "HuggingFaceTB/SmolLM-135M-Instruct" \
    --problem_name oscillator1

# Larger model for production
python main.py \
    --spec_path ./specs/specification_oscillator1_numpy.txt \
    --use_grpo True \
    --grpo_batch_size 8 \
    --grpo_learning_rate 1e-5 \
    --hf_model "microsoft/DialoGPT-large" \
    --problem_name oscillator1
```

### Advanced Configuration

```bash
# Custom GRPO settings
python main.py \
    --spec_path ./specs/specification_bactgrow_numpy.txt \
    --use_grpo True \
    --grpo_batch_size 16 \
    --grpo_learning_rate 5e-6 \
    --hf_model "microsoft/DialoGPT-large" \
    --problem_name bactgrow \
    --log_path ./logs/grpo_bactgrow \
    --max_sample_nums 1000
```

### Monitoring Training Progress

**Check GRPO Training Logs**:
```bash
# View training progress
tail -f logs/grpo_training.log

# Check model checkpoints
ls -la grpo_checkpoints/

# Monitor GPU usage
nvidia-smi -l 1
```

**Example Training Output**:
```
Training with GRPO on 4 samples...
GRPO training: 1 unique prompts with existing samples
Reusing NMSE rewards - Average: 0.7486, Min: 0.0000, Max: 0.9996
Flattened dataset: 4 prompt-completion pairs
Starting GRPO training...
{'loss': 0.0, 'learning_rate': 2e-05, 'reward': 1.0, 'epoch': 1.0}
GRPO training completed
```

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```bash
# Solution: Reduce batch sizes
--grpo_batch_size 2
--samples_per_prompt 2

# Or use smaller model
--hf_model "distilgpt2"
```

**2. GRPO Dependencies Missing**
```bash
# Install required packages
pip install trl>=0.14.0 peft>=0.14.0 datasets>=3.2.0 accelerate>=1.2.1
```

**3. Empty Equations Generated**
- **Cause**: Model generates non-extractable text
- **Solution**: Increase `max_new_tokens` or use larger model
- **Automatic Handling**: System assigns 0.0 reward to teach model to avoid

**4. Training Not Triggered**
- **Cause**: `grpo_batch_size` not reached
- **Solution**: Check if enough samples are being generated
- **Debug**: Look for "Training with GRPO" message in logs

**5. Model Not Improving**
- **Cause**: Learning rate too high/low, insufficient training data
- **Solution**: Adjust `grpo_learning_rate`, increase `grpo_batch_size`
- **Monitor**: Check reward trends in training logs

### Performance Optimization

**Memory Optimization**:
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use 8-bit optimization
optim="adamw_8bit"

# Reduce sequence lengths
max_prompt_length=256
max_completion_length=128
```

**Speed Optimization**:
```python
# Increase batch sizes (if memory allows)
grpo_batch_size=16
per_device_train_batch_size=2

# Use mixed precision
bf16=True  # On supported hardware
```

## Advanced Topics

### Custom Reward Functions

You can modify the reward computation in `sampler.py:218-254`:

```python
# Current: Exponential scaling
reward = np.exp(-mse)

# Alternative: Linear scaling
reward = max(0, 1 - mse)

# Alternative: Threshold-based
reward = 1.0 if mse < 0.01 else 0.1
```

### Multi-Objective Optimization

Extend rewards to include multiple criteria:

```python
# Combine MSE with complexity penalty
complexity_penalty = len(completion.split()) * 0.01
reward = np.exp(-mse) - complexity_penalty
```

### Curriculum Learning

Gradually increase problem difficulty:

```python
# Start with simple problems, progress to complex ones
if iteration < 100:
    problem = "oscillator1"  # Simple
elif iteration < 500:
    problem = "bactgrow"     # Medium
else:
    problem = "stressstrain" # Complex
```

## Contributing

When modifying GRPO functionality:

1. **Test with small models first** (`SmolLM-135M-Instruct`)
2. **Verify sample reuse** (no double generation)
3. **Check reward computation** (exponential scaling)
4. **Monitor GPU memory usage**
5. **Validate equation extraction** (handle edge cases)

## References

- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.14767)
- **LoRA Paper**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **LLM-SR Paper**: [Large Language Models for Scientific Discovery](https://www.nature.com/articles/s41586-023-06221-2)
- **TRL Library**: [Transformer Reinforcement Learning](https://github.com/huggingface/trl)