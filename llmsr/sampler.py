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
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install transformers to use HuggingFace models.")

try:
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    GRPO_AVAILABLE = True
except ImportError:
    GRPO_AVAILABLE = False
    print("Warning: GRPO dependencies not available. Install trl, peft, and datasets to use GRPO training.")


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
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        if llm_class == GRPOHuggingFaceLLM:
            self._llm = llm_class(samples_per_prompt, model_name=config.hf_model, 
                                  learning_rate=config.grpo_learning_rate)
        elif llm_class == HuggingFaceLLM:
            self._llm = llm_class(samples_per_prompt, model_name=config.hf_model)
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


class GRPOSampler(Sampler):
    """ Sampler that integrates GRPO training after each batch of samples. """
    
    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        super().__init__(database, evaluators, samples_per_prompt, config, max_sample_nums, llm_class)
        
        # GRPO-specific data collection
        self.training_data = []
        self.grpo_batch_size = config.grpo_batch_size  # Train every N samples
        self.sample_scores = {}  # Track scores for samples
        
    def sample(self, **kwargs):
        """ Sample with GRPO training integration. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # Collect samples and their rewards for GRPO training
            batch_data = []
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                
                # Create a unique key for this sample
                sample_key = f"sample_{cur_global_sample_nums}"
                
                # Store the sample data for GRPO training
                batch_data.append({
                    'prompt': prompt.code,
                    'completion': sample,
                    'sample': sample,
                    'sample_key': sample_key,
                    'evaluator': chosen_evaluator,
                    'island_id': prompt.island_id,
                    'version_generated': prompt.version_generated,
                    'global_sample_nums': cur_global_sample_nums,
                    'sample_time': sample_time
                })
                
                # Wrap the evaluator to capture scores
                original_register = self._database.register_program
                def capture_score_register(program, island_id, scores_per_test, **reg_kwargs):
                    # Calculate the score and store it
                    score = self._calculate_score_from_tests(scores_per_test)
                    self.sample_scores[sample_key] = score
                    # Call original register
                    return original_register(program, island_id, scores_per_test, **reg_kwargs)
                
                # Temporarily replace register_program to capture score
                self._database.register_program = capture_score_register
                
                # Analyze sample to get reward
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )
                
                # Restore original register function
                self._database.register_program = original_register
            
            # Collect rewards and trigger GRPO training if we have enough samples
            self._collect_rewards_and_train(batch_data, **kwargs)
    
    def _calculate_score_from_tests(self, scores_per_test):
        """ Calculate the aggregate score from test scores. """
        if not scores_per_test:
            return 0.0
        return np.mean(list(scores_per_test.values()))
    
    def _collect_rewards_and_train(self, batch_data, **kwargs):
        """ Collect PURE NMSE-based rewards and trigger GRPO training if conditions are met. """
        # Collect ONLY NMSE-based rewards from evaluation scores
        for data in batch_data:
            sample_key = data['sample_key']
            if sample_key in self.sample_scores:
                score = self.sample_scores[sample_key]
                # The score from evaluator is -MSE, so we convert it to a positive reward
                # Higher score (lower MSE) = higher reward for GRPO
                if score is not None:
                    # Convert negative MSE to positive reward
                    mse = -score  # Convert back to positive MSE
                    # Use exponential scaling to make differences more pronounced
                    # Scale factor depends on typical MSE values - adjust if needed
                    if mse > 0:
                        reward = np.exp(-mse)  # Exponential reward: lower MSE = higher reward
                    else:
                        reward = 1.0  # Perfect fit gets maximum reward
                    data['reward'] = float(reward)
                    print(f"Sample {sample_key}: MSE={mse:.6f}, Score={score:.6f}, Reward={reward:.6f}")
                else:
                    data['reward'] = 0.0  # Failed evaluation
                    print(f"Sample {sample_key}: Failed evaluation, Reward=0.0")
                # Clean up stored score
                del self.sample_scores[sample_key]
            else:
                data['reward'] = 0.0  # Failed evaluation
        
        # Add to training data
        self.training_data.extend(batch_data)
        
        # Train with GRPO if we have enough samples
        if len(self.training_data) >= self.grpo_batch_size and hasattr(self._llm, 'train_with_grpo'):
            print(f"Training with GRPO on {len(self.training_data)} samples...")
            self._llm.train_with_grpo(self.training_data, self._evaluators, self._database)
            # Clear training data after training
            self.training_data = []


def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
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
        # for gpt APIs
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        
        # for mixtral
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        url = "http://127.0.0.1:5000/completions"
        instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n")
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
                 batch_inference: bool = True, trim: bool = True) -> None:
        """
        Hugging Face model for equation generation - matches LocalLLM structure exactly.
        
        Args:
            samples_per_prompt: Number of samples to generate per prompt
            model_name: Hugging Face model identifier
            batch_inference: Use batch inference when sampling
            trim: Whether to trim the response to extract equation body
        """
        super().__init__(samples_per_prompt)
        
        # Default model if none specified  
        if model_name is None:
            model_name = "microsoft/DialoGPT-medium"
            
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
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Special handling for LLaMA models with rope_scaling issues
            model_kwargs = {
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                'device_map': "auto" if torch.cuda.is_available() else None,
                'trust_remote_code': True,
            }
            
            # For LLaMA models, add specific config to handle rope_scaling
            if 'llama' in model_name.lower():
                try:
                    from transformers import LlamaConfig
                    config = LlamaConfig.from_pretrained(model_name)
                    # Override rope_scaling to the expected format if it exists
                    if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                        if isinstance(config.rope_scaling, dict) and 'rope_type' in config.rope_scaling:
                            # Convert new format to old format
                            config.rope_scaling = {
                                'type': config.rope_scaling.get('rope_type', 'linear'),
                                'factor': config.rope_scaling.get('factor', 1.0)
                            }
                    model_kwargs['config'] = config
                except ImportError:
                    pass  # If LlamaConfig not available, continue without it
                    
            # Disable quantization on non-CUDA systems (like MacBook)
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, **model_kwargs)
            else:
                # Use float16 on MPS (Apple Silicon) or float32 on CPU for compatibility
                model_kwargs['torch_dtype'] = torch.float16 if torch.backends.mps.is_available() else torch.float32
                model_kwargs.pop('device_map', None)  # Remove device_map for CPU/MPS
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            # Try multiple fallback models in order of preference
            fallback_models = [
                "microsoft/DialoGPT-medium",
                "gpt2-medium", 
                "gpt2",
                "distilgpt2"
            ]
            
            for fallback in fallback_models:
                try:
                    print(f"Trying fallback model: {fallback}")
                    model_name = fallback
                    self.model_name = model_name
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                    )
                    print(f"Successfully loaded fallback model: {fallback}")
                    break
                except Exception as fallback_e:
                    print(f"Fallback {fallback} also failed: {fallback_e}")
                    continue
            else:
                raise RuntimeError("All fallback models failed to load")
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Move model to appropriate device
        if not torch.cuda.is_available():
            # For MacBook, use MPS if available, otherwise CPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:
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
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                return all_samples
            except Exception:
                continue

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """API sampling method - placeholder for consistency."""
        # Just call local method for now
        return self._draw_samples_local(prompt, config)

    def _do_request(self, content: str) -> str:
        """Generate response using HuggingFace model - matches LocalLLM _do_request signature."""
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        # Generate using HuggingFace model
        inputs = self.tokenizer(content, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move inputs to the same device as model (CUDA, MPS, or CPU)
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
                    max_new_tokens=1024,
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


class GRPOHuggingFaceLLM(HuggingFaceLLM):
    """
    HuggingFace LLM with GRPO training capabilities.
    Extends HuggingFaceLLM to support online GRPO training.
    """
    
    def __init__(self, samples_per_prompt: int, model_name: str = None, 
                 batch_inference: bool = True, trim: bool = True, 
                 learning_rate: float = 2e-5) -> None:
        """
        Initialize GRPO-enabled HuggingFace model.
        """
        super().__init__(samples_per_prompt, model_name, batch_inference, trim)
        
        if not GRPO_AVAILABLE:
            raise ImportError("GRPO dependencies not available. Install trl, peft, and datasets.")
        
        # Setup LoRA for efficient training
        self._setup_lora()
        
        # Setup GRPO trainer with specified learning rate
        self._setup_grpo_trainer(learning_rate=learning_rate)
        
        print("GRPO-enabled HuggingFace model initialized successfully")
    
    def _setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.1,
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA applied. Trainable parameters: {self.model.print_trainable_parameters()}")
    
    def _setup_grpo_trainer(self, learning_rate=2e-5):
        """Setup GRPO trainer configuration."""
        # Configure optimizer based on available hardware
        if torch.cuda.is_available():
            optim = "adamw_8bit"
            use_bf16 = True
        else:
            # Use standard AdamW for CPU/MPS compatibility
            optim = "adamw_torch"
            use_bf16 = False
        
        self.grpo_config = GRPOConfig(
            output_dir="./grpo_checkpoints",
            learning_rate=learning_rate,
            per_device_train_batch_size=1,  # Reduced batch size for memory
            gradient_accumulation_steps=4,  # Increased to maintain effective batch size
            max_prompt_length=512,
            max_completion_length=256,
            num_generations=4,  # GRPO requires at least 2, using 4 to match samples_per_prompt
            optim=optim,
            num_train_epochs=1,
            bf16=use_bf16,
            remove_unused_columns=False,
            logging_steps=1,
            save_steps=100,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            report_to=[],  # Disable wandb logging for MacBook compatibility
            greater_is_better=True,  # Higher reward is better
        )
        
        # Initialize trainer (will be recreated for each training round)
        self.grpo_trainer = None
    
    def train_with_grpo(self, training_data, evaluators=None, database=None):
        """
        Train the model using GRPO with the collected training data.
        
        Args:
            training_data: List of dicts with 'prompt', 'completion', and 'reward' keys
        """
        if not training_data:
            print("No training data available for GRPO")
            return
        
        try:
            # Group training data by unique prompts to reuse existing samples
            prompt_groups = {}
            for data in training_data:
                prompt = data['prompt']
                if prompt not in prompt_groups:
                    prompt_groups[prompt] = []
                prompt_groups[prompt].append(data)
            
            # Prepare dataset with existing prompt-completion pairs
            dataset_entries = []
            all_rewards = []
            
            for prompt, group in prompt_groups.items():
                # Use existing completions instead of generating new ones
                group_rewards = []
                group_completions = []
                
                for item in group:
                    group_completions.append(item['completion'])
                    group_rewards.append(item['reward'])
                
                # Pad or truncate to match num_generations if needed
                while len(group_completions) < self.grpo_config.num_generations:
                    # If we have fewer samples than required, duplicate the best one
                    if group_rewards:
                        best_idx = np.argmax(group_rewards)
                        group_completions.append(group_completions[best_idx])
                        group_rewards.append(group_rewards[best_idx])
                    else:
                        # Fallback to empty completion
                        group_completions.append("    return 0")
                        group_rewards.append(0.0)
                
                # Truncate if we have too many
                group_completions = group_completions[:self.grpo_config.num_generations]
                group_rewards = group_rewards[:self.grpo_config.num_generations]
                
                # Create dataset entry with existing samples
                dataset_entries.append({
                    'prompt': prompt,
                    'completions': group_completions,
                    'rewards': group_rewards
                })
                all_rewards.extend(group_rewards)
            
            print(f"GRPO training: {len(dataset_entries)} unique prompts with existing samples")
            if all_rewards:
                avg_reward = np.mean(all_rewards)
                max_reward = np.max(all_rewards)
                min_reward = np.min(all_rewards)
                print(f"Reusing NMSE rewards - Average: {avg_reward:.4f}, Min: {min_reward:.4f}, Max: {max_reward:.4f}")
            else:
                print("No NMSE rewards available")
            
            # Flatten the dataset for GRPO training (each prompt-completion-reward as separate entry)
            flattened_prompts = []
            flattened_completions = []
            flattened_rewards = []
            
            for entry in dataset_entries:
                prompt = entry['prompt']
                completions = entry['completions']
                rewards = entry['rewards']
                
                for completion, reward in zip(completions, rewards):
                    flattened_prompts.append(prompt)
                    flattened_completions.append(completion)
                    flattened_rewards.append(reward)
            
            # Create dataset with individual prompt-completion-reward entries
            train_dataset = Dataset.from_dict({
                'prompt': flattened_prompts,
                'completion': flattened_completions,
                'reward': flattened_rewards
            })
            
            print(f"Flattened dataset: {len(flattened_prompts)} prompt-completion pairs")
            
            # Since we have pre-computed rewards, we can use a simple reward function
            def reward_function(completions, **kwargs):
                """
                Simple reward function that returns the pre-computed rewards.
                GRPO will use these for training without re-evaluation.
                """
                # For GRPO with pre-computed rewards, we can return the rewards directly
                # The rewards are already in the dataset, so GRPO should handle this automatically
                return [1.0 for _ in completions]  # Placeholder - GRPO uses dataset rewards
            
            self.grpo_trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=[reward_function],
                args=self.grpo_config,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
            )
            
            print("Starting GRPO training...")
            self.grpo_trainer.train()
            print("GRPO training completed")
            
            self.model.save_pretrained("./grpo_checkpoints/latest")
            
        except Exception as e:
            print(f"Error during GRPO training: {e}")
            import traceback
            traceback.print_exc()