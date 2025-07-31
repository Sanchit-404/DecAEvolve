# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-SR is a scientific equation discovery and symbolic regression framework that combines Large Language Models with evolutionary search. The system generates Python program skeletons representing mathematical equations and uses LLMs to propose hypotheses, which are then evaluated against scientific datasets.

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n llmsr python=3.11.7
conda activate llmsr
pip install -r requirements.txt

# Alternative: use environment.yml
conda env create -f environment.yml
conda activate llmsr
```

### Running LLM-SR

**With HuggingFace Models (Local):**
```bash
# Test with small model
python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --hf_model "HuggingFaceTB/SmolLM-135M-Instruct"

# Standard run
python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_local
```

**With GRPO Training:**
```bash
# Install GRPO dependencies
pip install trl>=0.14.0 peft>=0.14.0 datasets>=3.2.0 accelerate>=1.2.1 bitsandbytes>=0.45.2

# Run with GRPO
python main.py --spec_path ./specs/specification_oscillator1_numpy.txt --use_grpo True --grpo_batch_size 4 --grpo_learning_rate 2e-5
```

**With OpenAI API:**
```bash
export API_KEY=[YOUR_API_KEY_HERE]
python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt
```

**With Local LLM Server:**
```bash
# Start server (in separate terminal)
bash run_server.sh
# Or manually:
python ./llm_engine/engine.py --model_path mistralai/Mixtral-8x7B-Instruct-v0.1 --gpu_ids 0 --port 5000 --quantization

# Run LLM-SR
python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt
```

### Testing
```bash
# Integration tests
python test_hf_integration.py
python test_grpo_integration.py
python test_grpo_mse.py
```

## Architecture

### Core Components

**Pipeline (`llmsr/pipeline.py`):**
- Main orchestration logic that coordinates sampling, evaluation, and evolution
- Extracts function names from specifications using decorators `@evaluate.run` and `@equation.evolve`
- Manages the iterative search process

**Sampler (`llmsr/sampler.py`):**
- Contains LLM interfaces: `HuggingFaceLLM`, `GRPOHuggingFaceLLM`, and API-based samplers
- Generates code hypotheses using LLMs based on prompts and previous successful solutions
- GRPO version supports reinforcement learning from human feedback during generation

**Evaluator (`llmsr/evaluator.py`):**
- `LocalSandbox` class executes generated code safely and measures fitness
- Evaluates mathematical expressions against datasets
- Handles timeout and error management during code execution

**Buffer (`llmsr/buffer.py`):**
- Experience buffer maintains multiple "islands" of evolved solutions for diversity
- Implements clustering and sampling strategies for selecting examples to include in prompts
- Manages reset mechanisms to prevent stagnation

**Configuration (`llmsr/config.py`):**
- `Config`: Main experiment parameters (samplers, evaluators, timeouts, model settings)
- `ExperienceBufferConfig`: Multi-population evolution settings
- `ClassConfig`: Specifies which LLM and sandbox classes to use

### Key Files

**Entry Point:**
- `main.py`: Command-line interface and experiment setup

**LLM Engine:**
- `llm_engine/engine.py`: Local LLM server for hosting HuggingFace models

**Specifications:**
- `specs/`: Problem specifications with function templates decorated for evolution
- Templates use either numpy+BFGS or torch+Adam optimizers

**Datasets:**
- `data/`: Scientific datasets (oscillator1, oscillator2, bactgrow, stressstrain)
- Each contains train.csv, test_id.csv, test_ood.csv

### Problem Specification Format

Specifications must contain exactly:
1. One function decorated with `@evaluate.run` - evaluates generated equations
2. One function decorated with `@equation.evolve` - the function structure to be discovered

The pipeline extracts these using `code_manipulation.yield_decorated()`.

### Available Datasets
- `oscillator1`: Oscillatory system equations
- `oscillator2`: Complex oscillatory dynamics  
- `bactgrow`: Bacterial growth modeling
- `stressstrain`: Materials stress-strain relationships

### Model Support
- **HuggingFace models**: Local inference with quantization support
- **OpenAI API**: GPT-3.5, GPT-4 series
- **GRPO Training**: Fine-tuning during search with reinforcement learning

### Configuration Parameters
Key settings in `llmsr/config.py`:
- `samples_per_prompt`: Number of hypotheses generated per LLM call
- `num_samplers`/`num_evaluators`: Parallelization settings
- `experience_buffer.functions_per_prompt`: How many previous solutions to include in prompts
- `experience_buffer.num_islands`: Population diversity management