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

# New imports for Hugging Face integration
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
        # Pass model configuration to LLM class
        if llm_class == HuggingFaceLLM:
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
            samples = self._llm.draw_samples(prompt.code,self.config)
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