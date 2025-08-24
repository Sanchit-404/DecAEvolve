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

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from llmsr import evaluator
from llmsr import buffer
from llmsr import config as config_lib
import requests
import json
import http.client
import os

# Conditional imports to avoid dependency issues
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from accelerate import Accelerator, infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_balanced_memory
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install transformers to use HuggingFace models.")

# GRPO imports moved to grpo_sampler.py


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
            use_distributed: bool = False,  # Add this parameter
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        
        # Import GRPO class only if needed
        try:
            from .grpo_sampler import GRPOHuggingFaceLLM, OfflineGRPOHuggingFaceLLM
            grpo_available = True
        except ImportError:
            grpo_available = False
            
        if grpo_available and llm_class.__name__ == 'GRPOHuggingFaceLLM':
            self._llm = llm_class(samples_per_prompt, model_name=config.hf_model, 
                                  learning_rate=config.grpo_learning_rate,
                                  use_distributed=use_distributed)
        elif llm_class == HuggingFaceLLM:
            self._llm = llm_class(samples_per_prompt, model_name=config.hf_model,
                                  use_distributed=use_distributed)
        elif llm_class.__name__ == 'OfflineGRPOHuggingFaceLLM':
            try:
                self._llm = llm_class(samples_per_prompt, model_name=config.hf_model,
                                      use_distributed=use_distributed)
            except TypeError:
                self._llm = llm_class(samples_per_prompt, use_distributed=use_distributed)
        else:
            self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1


# GRPOSampler moved to grpo_sampler.py


import re
from typing import Optional

def _extract_body(sample: str, config: "config_lib.Config") -> str:
    """
    Robustly extract the function body from a response sample, handling decorators,
    multi-line signatures, and various indentation styles. Returns only the function body,
    properly dedented, or the original sample if no function is found.

    Args:
        sample: The raw LLM response as a string.
        config: Configuration object (must have .use_api attribute).

    Returns:
        The extracted function body as a string, or the original sample if no function found.
    """
    lines = sample.splitlines()
    n = len(lines)
    func_start = None
    func_end = None

    # Helper: Find the first function definition (optionally after decorators)
    def find_func_start(lines):
        in_decorator = False
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                in_decorator = True
                continue
            if stripped.startswith("def "):
                return i
        return None

    func_start = find_func_start(lines)
    if func_start is None:
        # No function found, return original sample
        return sample.strip()

    # Find the end of the function signature (handles multi-line signatures)
    sig_end = func_start
    open_parens = 0
    found_colon = False
    for i in range(func_start, n):
        line = lines[i]
        open_parens += line.count("(") - line.count(")")
        if ":" in line and open_parens <= 0:
            sig_end = i
            found_colon = True
            break
    if not found_colon:
        # Malformed function, return original sample
        return sample.strip()

    # Determine indentation of the function body
    body_lines = []
    body_indent: Optional[int] = None
    for i in range(sig_end + 1, n):
        line = lines[i]
        # Skip empty lines after signature
        if not line.strip() and not body_lines:
            continue
        # Find the first non-empty line to determine indentation
        if body_indent is None and line.strip():
            body_indent = len(line) - len(line.lstrip())
        # If indentation is less than body_indent, function body ends
        if body_indent is not None and (len(line) - len(line.lstrip()) < body_indent) and line.strip():
            break
        # Only include lines that are part of the function body (including blank lines)
        if body_indent is not None:
            body_lines.append(line)
    # Remove trailing blank lines
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    # Dedent the function body
    if body_lines and body_indent is not None:
        dedented = []
        for l in body_lines:
            if l.strip():
                dedented.append(l[body_indent:] if l.startswith(" " * body_indent) else l.lstrip())
            else:
                dedented.append("")
        code = "\n".join(dedented)
    else:
        code = ""

    # If config.use_api, do not re-indent; else, optionally re-indent (legacy behavior)
    if not code and config.use_api:
        return ""
    if not code:
        return sample.strip()
    if not config.use_api:
        # Optionally re-indent to 4 spaces (legacy behavior)
        code = "\n".join(("    " + l if l.strip() else "") for l in code.splitlines())
    return code



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        url = "http://127.0.0.1:5000/completions"
        instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below with mathematical structure, considering the physical meaning and relationships of inputs.\n\n")
        self._batch_inference = batch_inference
        self._url = url
        self._instruction_prompt = instruction_prompt
        self._trim = trim


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)


    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                # response from llm server
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
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                return all_samples
            except Exception:
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        
        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    conn = http.client.HTTPSConnection("api.openai.com")
                    payload = json.dumps({
                        "max_tokens": 512,
                        "model": config.api_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                    headers = {
                        'Authorization': f"Bearer {os.environ['API_KEY']}",
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))
                    response = data['choices'][0]['message']['content']
                    
                    if self._trim:
                        response = _extract_body(response, config)
                    
                    all_samples.append(response)
                    break

                except Exception:
                    continue
        
        return all_samples
    
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        
        if response.status_code == 200: #Server status code 200 indicates successful HTTP request! 
            response = response.json()["content"]
            
            return response if self._batch_inference else response[0]


class HuggingFaceLLM(LLM):
    def __init__(self, samples_per_prompt: int, model_name: str = None, 
                 batch_inference: bool = True, trim: bool = True, 
                 use_distributed: bool = False, max_memory: dict = None) -> None:
        """
        Hugging Face model for equation generation with multi-GPU support.
        
        Args:
            samples_per_prompt: Number of samples to generate per prompt
            model_name: Hugging Face model identifier
            batch_inference: Use batch inference when sampling
            trim: Whether to extract equation body from response
            use_distributed: Whether to use accelerate distributed training
            max_memory: Dictionary specifying max memory per device
        """
        super().__init__(samples_per_prompt)
        
        if model_name is None:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
        self.model_name = model_name
        self._batch_inference = batch_inference
        self._trim = trim
        self.use_distributed = use_distributed
        
        instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             You are given an example of the function signature in the first function below. \
                             Your task is to complete the last 'equation' function with your mathematical relationship, considering the physical meaning and relationships of inputs. \
                             Only complete the body of the current 'equation' function. Do NOT give me a new function. Just give me the new mathematical relationship in function body. Do NOT use equation_v0 in your implementation.\n\n \
                             ")
        
        self._instruction_prompt = instruction_prompt
        
        # Initialize accelerator first if using distributed training
        if self.use_distributed:
            self.accelerator = Accelerator(
                mixed_precision='fp16',
                gradient_accumulation_steps=4,
            )
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self._load_model(max_memory)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print(f"Model loaded successfully")
    
    def _load_model(self, max_memory: dict = None):
        """Load model with appropriate strategy based on distributed setting."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.use_distributed:
                # For distributed training - don't use device_map
                print("Loading model for distributed training...")
                
                model_kwargs = {
                    'torch_dtype': torch.float16,
                    'trust_remote_code': True,
                }
                
                # Handle LLaMA models
                if 'llama' in self.model_name.lower():
                    try:
                        from transformers import LlamaConfig
                        config = LlamaConfig.from_pretrained(self.model_name)
                        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                            if isinstance(config.rope_scaling, dict) and 'rope_type' in config.rope_scaling:
                                config.rope_scaling = {
                                    'type': config.rope_scaling.get('rope_type', 'linear'),
                                    'factor': config.rope_scaling.get('factor', 1.0)
                                }
                        model_kwargs['config'] = config
                    except ImportError:
                        pass
                
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                
                # Apply LoRA BEFORE accelerate preparation if needed
                if hasattr(self, '_needs_lora_setup') and self._needs_lora_setup:
                    self._setup_lora()
                    self._needs_lora_setup = False
                
                # Let accelerate handle the model distribution
                self.model = self.accelerator.prepare(self.model)
                self.device = self.accelerator.device
                
            else:
                # For non-distributed inference - use device_map='auto'
                num_gpus = torch.cuda.device_count()
                print(f"Available GPUs: {num_gpus}")
                
                if num_gpus > 1:
                    print("Setting up multi-GPU inference with device_map='auto'...")
                    
                    if max_memory is None:
                        max_memory = {}
                        for i in range(num_gpus):
                            gpu_memory = torch.cuda.get_device_properties(i).total_memory
                            max_memory[i] = f"{int(gpu_memory * 0.95 / 1024**3)}GB"
                        print(f"Auto-detected memory allocation: {max_memory}")
                    
                    model_kwargs = {
                        'torch_dtype': torch.float16,
                        'trust_remote_code': True,
                        'device_map': 'auto',
                        'max_memory': max_memory,
                        'offload_folder': 'offload',
                    }
                    
                    # Handle LLaMA models
                    if 'llama' in self.model_name.lower():
                        try:
                            from transformers import LlamaConfig
                            config = LlamaConfig.from_pretrained(self.model_name)
                            if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                                if isinstance(config.rope_scaling, dict) and 'rope_type' in config.rope_scaling:
                                    config.rope_scaling = {
                                        'type': config.rope_scaling.get('rope_type', 'linear'),
                                        'factor': config.rope_scaling.get('factor', 1.0)
                                    }
                            model_kwargs['config'] = config
                        except ImportError:
                            pass
                    
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                    
                    # Apply LoRA after device_map setup if needed
                    if hasattr(self, '_needs_lora_setup') and self._needs_lora_setup:
                        self._setup_lora()
                        self._needs_lora_setup = False
                    
                    print(f"Model distributed across {num_gpus} GPUs")
                    
                else:
                    # Single GPU or CPU
                    print("Using single GPU or CPU setup...")
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    model_kwargs = {
                        'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                        'trust_remote_code': True,
                    }
                    
                    # Handle LLaMA models
                    if 'llama' in self.model_name.lower():
                        try:
                            from transformers import LlamaConfig
                            config = LlamaConfig.from_pretrained(self.model_name)
                            if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                                if isinstance(config.rope_scaling, dict) and 'rope_type' in config.rope_scaling:
                                    config.rope_scaling = {
                                        'type': config.rope_scaling.get('rope_type', 'linear'),
                                        'factor': config.rope_scaling.get('factor', 1.0)
                                    }
                            model_kwargs['config'] = config
                        except ImportError:
                            pass
                    
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                    
                    # Apply LoRA after model loading if needed
                    if hasattr(self, '_needs_lora_setup') and self._needs_lora_setup:
                        self._setup_lora()
                        self._needs_lora_setup = False
                    
                    if not torch.cuda.is_available():
                        if torch.backends.mps.is_available():
                            self.device = torch.device("mps")
                        else:
                            self.device = torch.device("cpu")
                        self.model = self.model.to(self.device)
                
                # Create a minimal accelerator for non-distributed use
                self.accelerator = Accelerator(
                    mixed_precision='fp16' if torch.cuda.is_available() else 'no',
                    gradient_accumulation_steps=4,
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to load specified model '{self.model_name}': {e}")
    
    def get_device_for_inputs(self, inputs):
        """Get the appropriate device for inputs based on model setup."""
        if self.use_distributed:
            return self.device
        elif hasattr(self.model, 'hf_device_map'):
            # Multi-GPU setup with device_map - inputs go to first device
            return torch.device(f"cuda:0")
        else:
            # Single device setup
            return self.device if hasattr(self, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_generation_model(self):
        """Get the model for generation, handling DDP wrapping."""
        if hasattr(self.model, 'module'):
            # If model is wrapped (e.g., in DDP), get the base model for generation
            return self.model.module
        else:
            return self.model

    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Local sampling method."""
        prompt = '\n'.join([self._instruction_prompt, prompt])
        
        while True:
            try:
                all_samples = []
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                return all_samples
            except Exception as e:
                print(f"Error in sampling: {e}")
                continue

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """API sampling method - falls back to local method."""
        return self._draw_samples_local(prompt, config)

    def _do_request(self, content: str) -> str:
        """Generate response using HuggingFace model."""
        content = content.strip('\n').strip()
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        inputs = self.tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to appropriate device
        target_device = self.get_device_for_inputs(inputs)
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        
        # Get the correct model for generation (handles DDP wrapping)
        generation_model = self.get_generation_model()
        
        with torch.no_grad():
            if self._batch_inference:
                outputs = generation_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_return_sequences=repeat_prompt,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                responses = []
                for output in outputs:
                    generated_text = self.tokenizer.decode(
                        output[inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    responses.append(generated_text.strip())
                
                return responses
            else:
                outputs = generation_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                return generated_text.strip()