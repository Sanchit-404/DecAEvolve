# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration of a LLMSR experiments
."""
from __future__ import annotations

import dataclasses
from typing import Type
import os

from llmsr import sampler
from llmsr import evaluator


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.
    
    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands in experience buffer for diversity
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
        cluster_sampling_temperature_period (int): Period for temperature decay
    """
    functions_per_prompt: int = 2 
    num_islands: int = 10 
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO training.
    
    Args:
        model_path: Path to the base model for GRPO training
        use_grpo: Whether to use GRPO training
        learning_rate: Learning rate for GRPO training
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_prompt_length: Maximum prompt length
        max_completion_length: Maximum completion length
        num_generations: Number of generations per prompt
        update_frequency: How often to perform GRPO updates
        use_lora: Whether to use LoRA for efficient training
        kl_coeff: KL coefficient for regularization
    """
    model_path: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    use_grpo: bool = False
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_prompt_length: int = 512
    max_completion_length: int = 256
    num_generations: int = 8
    update_frequency: int = 50
    use_lora: bool = True
    kl_coeff: float = 0.1
    optim: str = "adamw_8bit"
    num_train_epochs: int = 1
    bf16: bool = True


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for LLMSR experiments.
   
   Args:
       experience_buffer: Evolution multi-population settings
       num_samplers (int): Number of parallel samplers
       num_evaluators (int): Number of parallel evaluators
       samples_per_prompt (int): Number of hypotheses per prompt
       evaluate_timeout_seconds (int): Hypothesis evaluation timeout
       use_api (bool): API usage flag
       grpo_config: GRPO-specific configuration
   """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1 
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    evaluate_timeout_seconds: int = 30  
    use_api: bool = False
    api_model: str = "gpt-3.5-turbo"
    grpo_config: GRPOConfig = dataclasses.field(default_factory=GRPOConfig)


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]
    sampler_class: Type[sampler.Sampler] = sampler.Sampler
