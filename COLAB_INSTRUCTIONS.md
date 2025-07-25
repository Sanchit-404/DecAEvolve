# LLM-SR HuggingFace Integration for Google Colab

## 🎯 **Objective**
Test LLM-SR with larger language models using HuggingFace transformers in Google Colab, as small models (< 2B parameters) are not generating quality mathematical equations.

## 🚀 **Quick Start in Colab**

### 1. Install Dependencies
```bash
!pip install transformers torch accelerate numpy
```

### 2. Copy the Standalone Script
Copy the entire content of `colab_llmsr_hf.py` to a Colab cell and run it.

### 3. Recommended Models to Test

**Best Options for Colab:**
- `"Qwen/Qwen2.5-7B-Instruct"` - Excellent for math/coding (7B parameters)
- `"microsoft/DialoGPT-large"` - Alternative option (345M parameters)
- `"HuggingFaceTB/SmolLM-1.7B-Instruct"` - Smaller fallback

**Premium Options (if you have Colab Pro or authentication):**
- `"meta-llama/Llama-3.1-8B-Instruct"` - Requires HuggingFace authentication
- `"unsloth/Meta-Llama-3.1-8B-bnb-4bit"` - 4-bit quantized version

### 4. Test Different Models
Edit the `models_to_test` list in the script:

```python
models_to_test = [
    "Qwen/Qwen2.5-7B-Instruct",        # Recommended first choice
    "microsoft/DialoGPT-large",        # Backup option
    "HuggingFaceTB/SmolLM-1.7B-Instruct",  # Fallback
]
```

## 🔧 **Integration with Full LLM-SR**

### Modified Files Summary

**1. `llmsr/sampler.py`** - Added `HuggingFaceLLM` class that:
- Matches `LocalLLM` structure exactly
- Supports both batch and single inference
- Uses same `_extract_body` trimming function
- Compatible with existing LLM-SR pipeline

**2. `main.py`** - Updated to use HuggingFace models by default
**3. `llmsr/config.py`** - Added `hf_model` configuration parameter

### Key Features
- **Exact LocalLLM compatibility**: Same method signatures and behavior
- **Batch inference support**: Generates multiple samples efficiently
- **Automatic device detection**: Uses GPU if available, falls back to CPU
- **Model flexibility**: Easy to switch between different HF models
- **Error handling**: Graceful fallbacks and retry logic

## 📊 **Expected Results**

With larger models (7B+), you should see:
- **Better equation structure**: Valid mathematical expressions
- **Proper variable usage**: Correct use of `x`, `v`, and `params`
- **Diverse outputs**: Multiple creative mathematical formulations
- **Faster convergence**: Better starting points for optimization

## 🔄 **Next Steps: GRPO Integration**

Once we confirm larger models generate quality equations, we can proceed with:
1. **GRPO trainer integration** at each LLM-SR iteration
2. **Reward-based fine-tuning** using equation evaluation scores
3. **Continuous improvement** of equation generation quality

## 🐛 **Troubleshooting**

**Memory Issues:**
- Use smaller models like `DialoGPT-large`
- Enable gradient checkpointing
- Reduce `max_new_tokens`

**Authentication Errors:**
- For Meta Llama models, login to HuggingFace: `!huggingface-cli login`
- Use alternative models that don't require authentication

**Generation Quality:**
- Increase `temperature` for more creativity
- Adjust `top_p` and `top_k` for different sampling strategies
- Increase `max_new_tokens` for longer generations

## 📝 **Example Usage**

```python
# Initialize with 7B model
hf_llm = HuggingFaceLLM(
    samples_per_prompt=4,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    batch_inference=True
)

# Generate equation samples
samples = hf_llm.draw_samples(oscillator_prompt)
print(f"Generated {len(samples)} equation samples")
```

This integration provides a solid foundation for testing larger models and implementing GRPO post-training in the next phase! 