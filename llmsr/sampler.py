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
from accelerate import infer_auto_device_map
from accelerate import dispatch_model

# Conditional imports to avoid dependency issues
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        # Import GRPO class only if needed
        try:
            from .grpo_sampler import GRPOHuggingFaceLLM
            grpo_available = True
        except ImportError:
            grpo_available = False
            
        if grpo_available and llm_class.__name__ == 'GRPOHuggingFaceLLM':
            self._llm = llm_class(samples_per_prompt, model_name=config.hf_model, 
                                  learning_rate=config.grpo_learning_rate)
        elif llm_class == HuggingFaceLLM:
            self._llm = llm_class(samples_per_prompt, model_name=config.hf_model)
        elif llm_class.__name__ == 'OfflineGRPOHuggingFaceLLM':
            try:
                self._llm = llm_class(samples_per_prompt, model_name=config.hf_model)
            except TypeError:
                self._llm = llm_class(samples_per_prompt)
        else:
            self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config
        self.model = self._llm.model

    
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

# extract from new function equation_v2
def _extract_body_v1(sample: str, config: "config_lib.Config") -> str:
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

def _extract_body_v2(sample: str, config: "config_lib.Config") -> str:
    """
    Extract the first function body from a response sample. This function handles two cases:
    1. If there's content before the first 'def' statement, treat it as a continuation 
       of equation_v1 and return that content.
    2. If there's no content before the first 'def', extract the body of the first 
       complete function found.

    Args:
        sample: The raw LLM response as a string.
        config: Configuration object (must have .use_api attribute).

    Returns:
        The extracted function body as a string, or the original sample if no function found.
    """
    lines = sample.splitlines()
    n = len(lines)
    
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
    
    # Case 1: Check if there's meaningful content before the first function definition
    if func_start is not None and func_start > 0:
        # Extract content before the first function
        pre_func_lines = []
        for i in range(func_start):
            line = lines[i]
            stripped = line.strip()
            # Skip empty lines and comments, but include code
            if stripped and not stripped.startswith("#"):
                pre_func_lines.append(line)
        
        # If we found meaningful content before the function, return it
        if pre_func_lines:
            # Determine the minimum indentation
            min_indent = float('inf')
            for line in pre_func_lines:
                if line.strip():  # Only consider non-empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            # Dedent the content
            if min_indent == float('inf'):
                min_indent = 0
            
            dedented = []
            for line in pre_func_lines:
                if line.strip():
                    dedented.append(line[min_indent:] if len(line) >= min_indent else line.lstrip())
                else:
                    dedented.append("")
            
            code = "\n".join(dedented).strip()
            
            # Apply indentation based on config
            if not config.use_api and code:
                lines = code.splitlines()
                if lines:
                    lines[0] = "    " + lines[0].lstrip()
                    code = "\n".join(lines)
            return code
    
    # Case 2: No meaningful content before first function, extract first function body
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
    body_indent = None
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

    # Apply indentation based on config
    if not code and config.use_api:
        return ""
    if not code:
        return sample.strip()
    if not config.use_api:
        # Optionally re-indent to 4 spaces (legacy behavior)
        code = "\n".join(("    " + l.strip() if l.strip() else "") for l in code.splitlines())
    return code


def _extract_body(sample: str, config: "config_lib.Config") -> str:
    """
    Extract the first code block that represents the continuation/body of equation_v1
    or the first function body. Always terminates immediately after the *first* actual
    `return` statement (ignoring comments/strings/docstrings), and supports multi-line
    returns.

    Works whether or not the sample contains a `def ...` at all.
    """
    lines = sample.splitlines()
    n = len(lines)

    def find_func_start(_lines):
        for i, line in enumerate(_lines):
            s = line.lstrip()
            if s.startswith("def "):
                return i
        return None

    def dedent_block(_lines):
        nonempty = [l for l in _lines if l.strip()]
        if not nonempty:
            return _lines
        min_indent = min(len(l) - len(l.lstrip()) for l in nonempty)
        return [(l[min_indent:] if len(l) >= min_indent else l.lstrip()) for l in _lines]

    def strip_trailing_blanks(_lines):
        out = _lines[:]
        while out and not out[-1].strip():
            out.pop()
        return out

    # ---- Core: truncate after first real `return` (supports multiline) ----
    def truncate_after_first_return(_lines):
        in_triple = False
        triple_delim = None
        i = 0

        def scan_string(line, j, quote):
            esc = False
            j += 1
            L = len(line)
            while j < L:
                ch = line[j]
                if ch == "\\" and not esc:
                    esc = True
                    j += 1
                    continue
                if ch == quote and not esc:
                    return j + 1
                esc = False
                j += 1
            return L  # string continues to EOL (implicit close next line not tracked)

        def net_bracket_delta(line, start_idx=0):
            # Count ()[]{} ignoring strings/comments/triple strings
            j = start_idx
            L = len(line)
            depth_delta = 0
            in_str = False
            str_q = None
            nonlocal in_triple, triple_delim
            while j < L:
                if in_triple:
                    k = line.find(triple_delim, j)
                    if k == -1:
                        return depth_delta  # stays in triple
                    in_triple = False
                    j = k + 3
                    continue

                ch = line[j]

                # triple quotes
                if j + 2 < L and (line[j:j+3] in ("'''", '"""')):
                    in_triple = True
                    triple_delim = line[j:j+3]
                    j += 3
                    continue

                # line comment
                if ch == "#":
                    break

                # strings
                if not in_str and ch in ("'", '"'):
                    j = scan_string(line, j, ch)
                    continue

                if ch in "([{":
                    depth_delta += 1
                elif ch in ")]}":
                    depth_delta -= 1
                j += 1
            return depth_delta

        def is_return_token_here(line, j):
            # word-boundary `return` not inside quotes/triple/comment
            prev = line[j-1] if j > 0 else " "
            nxt = line[j+6] if j + 6 < len(line) else ""
            is_word = (not (prev.isalnum() or prev == "_")) and (nxt == "" or nxt.isspace() or nxt in "([{")
            return is_word

        while i < len(_lines):
            line = _lines[i]
            j = 0
            L = len(line)
            # scan line for first *real* return
            in_str = False
            # (we reuse triple/docstring state carried in outer vars)

            while j < L:
                if in_triple:
                    k = line.find(triple_delim, j)
                    if k == -1:
                        j = L
                        break
                    in_triple = False
                    j = k + 3
                    continue

                ch = line[j]

                # triple quotes
                if j + 2 < L and (line[j:j+3] in ("'''", '"""')):
                    in_triple = True
                    triple_delim = line[j:j+3]
                    j += 3
                    continue

                # comment
                if ch == "#":
                    break

                # strings
                if ch in ("'", '"'):
                    j = j + 1
                    j = j if j >= L else (j + (0 if line[j-1] == ch else 0))  # fallthrough to scan below
                    # use helper to skip full string
                    def _skip_string(s, start, quote):
                        esc = False
                        k = start
                        while k < len(s):
                            c = s[k]
                            if c == "\\" and not esc:
                                esc = True
                                k += 1
                                continue
                            if c == quote and not esc:
                                return k + 1
                            esc = False
                            k += 1
                        return k
                    j = _skip_string(line, j-1, ch)
                    continue

                # return keyword?
                if j + 6 <= L and line[j:j+6] == "return" and is_return_token_here(line, j):
                    # include this line and then accumulate until the return expression ends
                    kept = _lines[:i+1]
                    # Start counting brackets after 'return'
                    depth = net_bracket_delta(line, j + 6)
                    # backslash continuation?
                    cont = line.rstrip().endswith("\\")
                    k = i
                    while True:
                        if depth == 0 and not cont and not in_triple:
                            return kept  # done at this very line
                        k += 1
                        if k >= len(_lines):
                            return kept  # EOF fallback
                        next_line = _lines[k]
                        kept.append(next_line)
                        d = net_bracket_delta(next_line, 0)
                        depth += d
                        cont = next_line.rstrip().endswith("\\")
                    # unreachable
                j += 1
            i += 1

        return _lines  # no return found

    # ---------- Main extraction logic ----------
    func_start = find_func_start(lines)

    if func_start is None:
        # No function found at all: treat entire sample as a continuation body and cut after first `return`
        trimmed = truncate_after_first_return(lines)
        # Drop entirely empty/comment/docstring-only leading stuff
        # (keep non-empty, non-pure-comment, non-triple-quote delimiter lines)
        cleaned = []
        skip_triple = False
        triple = None
        for ln in trimmed:
            s = ln.strip()
            if skip_triple:
                if triple and triple in s:
                    skip_triple = False
                continue
            if s.startswith("'''") or s.startswith('"""'):
                triple = s[:3]
                if s.count(triple) < 2:
                    skip_triple = True
                continue
            if not s or s.startswith("#"):
                continue
            cleaned.append(ln)
        cleaned = strip_trailing_blanks(cleaned)
        code = "\n".join(dedent_block(cleaned)).strip()
        if not code and config.use_api:
            return ""
        if not code:
            return sample.strip()
        code = "\n".join(("    " + l.strip() if l.strip() else "") for l in code.splitlines())
        return code

    # There is a function: first try to extract its body as before
    # 1) find end of signature
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
        return sample.strip()

    # 2) collect body by indentation
    body_lines = []
    body_indent = None
    for i in range(sig_end + 1, n):
        line = lines[i]
        if not line.strip() and not body_lines:
            continue
        if body_indent is None and line.strip():
            body_indent = len(line) - len(line.lstrip())
        if body_indent is not None and (len(line) - len(line.lstrip()) < body_indent) and line.strip():
            break
        if body_indent is not None:
            body_lines.append(line)

    body_lines = strip_trailing_blanks(body_lines)
    if not body_lines:
        if config.use_api:
            return ""
        return sample.strip()

    dedented = [(l[body_indent:] if l.startswith(" " * body_indent) else l.lstrip()) if l.strip() else "" for l in body_lines]
    dedented = strip_trailing_blanks(dedented)

    # ✂️ terminate after first syntactic `return`
    dedented = truncate_after_first_return(dedented)

    code = "\n".join(dedented).strip()
    if not code and config.use_api:
        return ""
    if not code:
        return sample.strip()
    if not config.use_api:
        code = "\n".join(("    " + l.strip() if l.strip() else "") for l in code.splitlines())
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
        
        # Default model if none specified (no fallbacks)
        if model_name is None:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
        self.model_name = model_name
        self._batch_inference = batch_inference
        self._trim = trim
        
        # Instruction prompt - exactly like LocalLLM
        instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             You are given an example of the function signature in the first function equation_v0 below. \
                             Your task is to complete the last 'equation' function with your mathematical relationship, considering the physical meaning and relationships of inputs. \
                             Only give me the completion code of the BODY of the current 'equation' FUNCTION. Do NOT give me a new function. Do NOT give me 'pass' instead of completion. Just give me the new mathematical relationship with inputs for completion of current function body. Do NOT use equation_v0 in your completion. \n\n \
                             ")

        self._instruction_prompt = instruction_prompt
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.use_multi_gpu = torch.cuda.device_count() > 1 or os.environ.get('ACCELERATE_USE_MULTI_GPU', 'false').lower() == 'true'
            
            # Special handling for LLaMA models with rope_scaling issues
            model_kwargs = {
                # "attn_implementation": "flash_attention_2",
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                # 'device_map': "auto" if torch.cuda.is_available() else None,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }
            
            # if 'llama' in model_name.lower():
            #     try:
            #         from transformers import LlamaConfig
            #         config = LlamaConfig.from_pretrained(model_name)
            #         if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            #             if isinstance(config.rope_scaling, dict) and 'rope_type' in config.rope_scaling:
            #                 config.rope_scaling = {
            #                     'type': config.rope_scaling.get('rope_type', 'linear'),
            #                     'factor': config.rope_scaling.get('factor', 1.0)
            #                 }
            #         model_kwargs['config'] = config
            #     except ImportError:
            #         pass
            
            if self.use_multi_gpu:
                # model_kwargs['device_map'] = 'auto'
                # model_kwargs['device_map'] = {"": 0}
                # Store that we're using distributed model
                self.is_distributed = True
            else:
                # Single GPU or CPU setup
                self.is_distributed = False
                
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            # device_map=infer_auto_device_map(model)
            device_map='cuda'
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True,
            )

            # self.model = dispatch_model(self.model, device_map=device_map)
            
            # # Handle device placement for non-distributed setups
            # if not self.is_distributed:
            #     if torch.cuda.is_available():
            #         self.device = torch.device("cuda:0")
            #         self.model = self.model.to(self.device)
            #     elif torch.backends.mps.is_available():
            #         self.device = torch.device("mps")
            #         self.model = self.model.to(torch.float32).to(self.device)
            #     else:
            #         self.device = torch.device("cpu")
            #         self.model = self.model.to(torch.float32).to(self.device)
            # else:
            #     # For distributed models, device is managed by accelerate
            #     self.device = next(self.model.parameters()).device
            
        except Exception as e:
            raise RuntimeError(f"Failed to load specified model '{model_name}': {e}")
        
        # breakpoint()
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.tokenizer.pad_token = "[PAD]"
        # self.tokenizer.padding_side = "left"

        
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

                # breakpoint()
                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                # breakpoint()
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
        inputs = self.tokenizer(content, return_tensors="pt", 
                                truncation=True, 
                                max_length=512,
                                padding=True,
                                add_special_tokens=True
        )
        # if self.is_distributed:
        #     # For distributed models, move to the device of the first parameter
        #     target_device = infer_auto_device_map(self.model)
        #     inputs = {k: v.to(target_device) for k, v in inputs.items()}
        # else:
        #     # For single device models
        #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Move inputs to the same device as model (CUDA, MPS, or CPU)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            if self._batch_inference:
                # self.model = self.model.bfloat16().cuda()
                # Generate multiple samples at once
                # self.model = self.model.half()
                # breakpoint()
                outputs = self.model.generate(
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
                # breakpoint()
                # Decode all outputs
                responses = []
                for output in outputs:
                    generated_text = self.tokenizer.decode(
                        output[inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    responses.append(generated_text.strip())
                # breakpoint()
                return responses
            else:
                # Generate single sample
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
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