""" Offline GRPO sampler that uses LLM-SR samples as training dataset instead of generating new samples. """
from __future__ import annotations

import os
import numpy as np
import torch
import time
from typing import Sequence, Type, List, Dict, Any, Optional
import re
from .sampler import HuggingFaceLLM, Sampler, LLM
from llmsr import evaluator, buffer, config as config_lib
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel

try:
    from accelerate.utils import is_peft_model
except ImportError:
    def is_peft_model(model):  # type: ignore[misc]
        return getattr(model, "peft_config", None) is not None
from datasets import Dataset
from dataclasses import dataclass, field
from accelerate import dispatch_model


def _vllm_server_port_external() -> int:
    """Match trl vllm-serve port when Slurm sets VLLM_SERVER_PORT."""
    return int(os.environ.get("VLLM_SERVER_PORT", "8000"))


def _vllm_group_port_base_external() -> int:
    """
    NCCL/TCP port base for TRL <-> vLLM weight sync. Must differ per concurrent job on one node.
    Override with VLLM_GROUP_PORT_BASE if needed.
    """
    raw = os.environ.get("VLLM_GROUP_PORT_BASE")
    if raw is not None and str(raw).strip() != "":
        b = int(str(raw).strip())
        return min(max(b, 1024), 64000)
    jid = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    return 52000 + (int(jid) % 12000)


def _vllm_group_port_external(episode: int) -> int:
    p = _vllm_group_port_base_external() + int(episode) * 100
    return min(max(p, 1024), 65000)


def _unwrap_to_peft_model(model: Any) -> Optional[PeftModel]:
    """Inner PeftModel if wrapped (e.g. torch.compile _orig_mod)."""
    cur: Any = model
    seen: set[int] = set()
    for _ in range(8):
        if not isinstance(cur, torch.nn.Module):
            break
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if isinstance(cur, PeftModel):
            return cur
        nxt = getattr(cur, "model", None)
        if nxt is None:
            nxt = getattr(cur, "_orig_mod", None)
        if nxt is None:
            break
        cur = nxt
    return None


def _model_already_has_peft_training(model: Any) -> bool:
    """If LoRA already applied, pass peft_config=None into GRPOTrainer (avoid stacked adapters)."""
    if _unwrap_to_peft_model(model) is not None:
        return True
    return bool(is_peft_model(model))


def _grpo_report_to() -> str | list:
    """HF/TRL W&B callback requires WANDB_API_KEY; otherwise training aborts at init."""
    if os.environ.get("WANDB_API_KEY", "").strip():
        return "wandb"
    return []


def _llmsr_reward_from_mse(mse: float) -> float:
    """
    Map validation MSE to a reward in (0, 1]: exp(-mse / sigma).
    Env LLMSR_GRPO_REWARD_SIGMA defaults to 1.0; increase (e.g. 2–5) for higher logged rewards at the same MSE
    (still monotonic in MSE for mse >= 0).
    """
    try:
        sigma = float(os.environ.get("LLMSR_GRPO_REWARD_SIGMA", "1.0").strip())
    except ValueError:
        sigma = 1.0
    sigma = max(sigma, 1e-6)
    return float(np.exp(-float(mse) / sigma))


def _llmsr_dataloader_num_workers() -> int:
    """GRPO DataLoader workers; env LLMSR_DATALOADER_NUM_WORKERS (default 4)."""
    try:
        n = int(os.environ.get("LLMSR_DATALOADER_NUM_WORKERS", "4").strip())
    except ValueError:
        n = 4
    return max(0, min(n, 32))


class OfflineGRPOHuggingFaceLLM(HuggingFaceLLM):
    """
    HuggingFace LLM with offline GRPO training capabilities.
    Uses pre-collected LLM-SR samples as training dataset instead of generating new samples.
    """
    
    def __init__(self, samples_per_prompt: int, model_name: str = None, 
                 batch_inference: bool = True, trim: bool = True, 
                 learning_rate: float = 1e-6) -> None:
        """
        Initialize offline GRPO-enabled HuggingFace model.
        """
        super().__init__(samples_per_prompt, model_name, batch_inference, trim)
        
        # self._setup_lora()
        # On Apple MPS, avoid float16 training instability
        try:
            if (not torch.cuda.is_available()) and torch.backends.mps.is_available():
                self.model = self.model.to(torch.float32).to("mps")
        except Exception:
            pass

        self.offline_dataset = []
        self.training_episodes = 0
        self.evaluators = None  # Will be set by sampler
        self._setup_grpo_trainer(learning_rate=learning_rate)
        
        print("Offline GRPO-enabled HuggingFace model initialized successfully")
    
    def set_evaluators(self, evaluators):
        """Set evaluators for reward computation during training."""
        self.evaluators = evaluators
        print(f"Set {len(evaluators)} evaluators for reward computation")
    
    def _setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            use_rslora="True",
        )
        
        self.model = get_peft_model(self.model, lora_config)
    
    def _setup_grpo_trainer(self, learning_rate=1e-6):
        """Setup GRPO trainer configuration for offline training (version-compatible)."""
        if torch.cuda.is_available():
            # optim = "adamw_8bit"
            use_bf16 = True
        # else:
        #     # Use standard AdamW for CPU/MPS compatibility
        #     optim = "adamw_torch"
        #     use_bf16 = False
        
        # Build kwargs and filter by GRPOConfig signature for compatibility across TRL versions
        import inspect
        lr_scheduler_type: str = "warmup_stable_decay"
        lr_scheduler_kwargs: dict[str, Any] = field(
            default_factory=lambda: dict(num_warmup_steps=50, num_decay_steps=0, min_lr_ratio=0.0)
        )
        lr_scheduler_kwargs_dict = lr_scheduler_kwargs.default_factory()
        token_entropy_percentile_threshold = 0.0 # from https://huggingface/papers/2506.01939
        # loss_type = `bnpo` => helps remove length bias if per_device_train_batch_size > 1
    
        
        cfg_kwargs = {
            # 'output_dir': f"./grpo_checkpoints/{self.problem_name}-adaptive/run4/episode{self.training_episodes}",
            # 'bf16': True,
            'learning_rate': learning_rate,
            'lr_scheduler_type': lr_scheduler_type,
            'warmup_steps': lr_scheduler_kwargs_dict["num_warmup_steps"],
            'lr_scheduler_kwargs': {k: v for k,v in lr_scheduler_kwargs_dict.items() if k != "num_warmup_steps"},
            'mask_truncated_completions': False,
            'temperature': 0.8,
            'top_p': 0.9,
            'loss_type': "bnpo",
            'use_liger_loss': (token_entropy_percentile_threshold == 0.0),
            'per_device_train_batch_size': 16,  # Reduced for stability
            'gradient_accumulation_steps': 4,
            'max_prompt_length': 2048,
            'max_completion_length': 768,
            'num_generations': 64,  # Reduced to match batch size
            'logging_steps': 1,
            'save_steps': 25,
            'dataloader_num_workers': _llmsr_dataloader_num_workers(),
            'greater_is_better': True,
            # Ensure finite training when dataloader has no length
            'max_steps': 500,
            'scale_rewards': True,
            # 'max_grad_norm': 1.0,
            'beta': 0.05,
            'epsilon': 0.2,
            'disable_dropout': True,
            'report_to': _grpo_report_to(),
            # vllm (127.0.0.1 + env port for Slurm dynamic bind)
            "use_vllm": True,
            "vllm_mode": "server",
            "vllm_server_base_url": f"http://127.0.0.1:{_vllm_server_port_external()}",
            "vllm_server_host": "127.0.0.1",
            "vllm_server_port": _vllm_server_port_external(),
            "vllm_server_timeout": 1200,
            "vllm_group_port": _vllm_group_port_external(0),
            # TRL default True multiplies loss by vLLM IS ratios; when train vs vLLM logprobs mismatch, grad_norm collapses.
            # Default off; set GRPO_VLLM_IS_CORRECTION=1 when train/vLLM are aligned (IS mean ~1).
            "vllm_importance_sampling_correction": os.environ.get("GRPO_VLLM_IS_CORRECTION", "").strip().lower()
            in ("1", "true", "yes"),
        }
        _model_tag = self.model_name.replace("/home/jovyan/nly934-storage/models/", "")
        cfg_kwargs["output_dir"] = (
            f"./grpo_checkpoints/{self.problem_name}-V2-adaptive-{_model_tag}-r8-ga{cfg_kwargs['gradient_accumulation_steps']}"
            f"-g{cfg_kwargs['num_generations']}-lr{learning_rate}/nprompt{self.n_prompts}"
        )

        sig = inspect.signature(GRPOConfig.__init__)
        filtered_kwargs = {k: v for k, v in cfg_kwargs.items() if k in sig.parameters}
        try:
            self.grpo_config = GRPOConfig(**filtered_kwargs)
        except TypeError:
            # Fallback: drop optional stabilizers if still incompatible
            minimal_keys = [
                'output_dir','learning_rate','per_device_train_batch_size','gradient_accumulation_steps',
                'max_prompt_length','max_completion_length','num_generations','optim',
                'num_train_epochs','bf16','remove_unused_columns','logging_steps','save_steps',
                'dataloader_num_workers','report_to','greater_is_better'
            ]
            minimal_kwargs = {k: v for k, v in cfg_kwargs.items() if k in sig.parameters and k in minimal_keys}
            self.grpo_config = GRPOConfig(**minimal_kwargs)
            print("fallback: minimal config for grpo")
        
        # Initialize trainer (will be recreated for each training round)
        self.grpo_trainer = None
    
    def _create_episode_config(self, lora_cfg):
        """Create a new GRPO config for each training episode with unique run_name."""
        import inspect
        import time
        
        # Build kwargs and filter by GRPOConfig signature for compatibility across TRL versions
        lr_scheduler_type: str = "warmup_stable_decay"
        lr_scheduler_kwargs: dict[str, Any] = field(
            default_factory=lambda: dict(num_warmup_steps=50, num_decay_steps=0, min_lr_ratio=0.0)
        )
        lr_scheduler_kwargs_dict = lr_scheduler_kwargs.default_factory()
        token_entropy_percentile_threshold = 0.0 # from https://huggingface/papers/2506.01939
        
        cfg_kwargs = {
            'learning_rate': 1e-6,
            'lr_scheduler_type': lr_scheduler_type,
            'warmup_steps': lr_scheduler_kwargs_dict["num_warmup_steps"],
            'lr_scheduler_kwargs': {k: v for k,v in lr_scheduler_kwargs_dict.items() if k != "num_warmup_steps"},
            'mask_truncated_completions': False,
            'temperature': 0.8,
            'top_p': 0.9,
            'loss_type': "bnpo",
            'use_liger_loss': (token_entropy_percentile_threshold == 0.0),
            'per_device_train_batch_size': 8,  # Reduced for stability
            'gradient_accumulation_steps': 8,
            'max_prompt_length': 2048,
            'max_completion_length': 768,
            'num_generations': 64,  # Reduced to match batch size
            'logging_steps': 1,
            'save_steps': 8,
            'dataloader_num_workers': _llmsr_dataloader_num_workers(),
            'greater_is_better': True,
            # Ensure finite training when dataloader has no length
            'max_steps': 8,
            'scale_rewards': True,
            # 'max_grad_norm': 1.0,
            'beta': 0.05,
            'epsilon': 0.2,
            'disable_dropout': True,
            'report_to': _grpo_report_to(),
            # Unique run_name for each episode
            # vllm
            "use_vllm": True,
            "vllm_mode": "server",
            "vllm_server_base_url": f"http://127.0.0.1:{_vllm_server_port_external()}",
            "vllm_server_host": "127.0.0.1",
            "vllm_server_port": _vllm_server_port_external(),
            "vllm_server_timeout": 1200,
            "vllm_group_port": _vllm_group_port_external(self.training_episodes),
            "vllm_importance_sampling_correction": os.environ.get("GRPO_VLLM_IS_CORRECTION", "").strip().lower()
            in ("1", "true", "yes"),
        }
        cfg_kwargs['output_dir']= f"./grpo_checkpoints/{self.problem_name}-adaptive-{self.model_name.replace('Qwen/', '')}-r{lora_cfg.r}-ga{cfg_kwargs['gradient_accumulation_steps']}-g{cfg_kwargs['num_generations']}/episode{self.training_episodes}"
        cfg_kwargs['run_name']= f"episode-{self.training_episodes}-{self.model_name}-r{lora_cfg.r}-ga{cfg_kwargs['gradient_accumulation_steps']}-ng{cfg_kwargs['num_generations']}-{int(time.time() * 1000)}"

        sig = inspect.signature(GRPOConfig.__init__)
        filtered_kwargs = {k: v for k, v in cfg_kwargs.items() if k in sig.parameters}
        try:
            return GRPOConfig(**filtered_kwargs)
        except TypeError:
            # Fallback: drop optional stabilizers if still incompatible
            minimal_keys = [
                'output_dir','learning_rate','per_device_train_batch_size','gradient_accumulation_steps',
                'max_prompt_length','max_completion_length','num_generations','optim',
                'num_train_epochs','bf16','remove_unused_columns','logging_steps','save_steps',
                'dataloader_num_workers','report_to','greater_is_better','run_name'
            ]
            minimal_kwargs = {k: v for k, v in cfg_kwargs.items() if k in sig.parameters and k in minimal_keys}
            return GRPOConfig(**minimal_kwargs)
    
    def collect_offline_sample(self, prompt: str, completion: str, reward: float):
        """
        Collect a single LLM-SR sample for offline training.
        
        Args:
            prompt: The prompt used to generate the completion
            completion: The LLM-SR generated completion
            reward: The evaluated reward (e.g., exp(-MSE))
        """
        # Clean completion to remove stray docstrings like \"\"\"Improved version of `equation_vX`\"\"\"
        cleaned = self._clean_completion_text(completion)
        self.offline_dataset.append({
            'prompt': prompt,
            'completion': cleaned,
            'reward': reward
        })

    def _clean_completion_text(self, text: str) -> str:
        """Remove version docstring noise and trim whitespace lines from completion body."""
        if not isinstance(text, str):
            text = str(text)
        lines = text.splitlines()
        cleaned_lines = []
        pattern = re.compile(r'^\s*"""Improved version of `equation_v\d+`\."""\s*$')
        for line in lines:
            if pattern.match(line):
                continue
            # Stop at the first new function or module-level noise
            if re.match(r'^\s*def\s+\w+\s*\(', line):
                break
            if re.match(r'^\s*if\s+__name__\s*==\s*["\"]__main__["\"]\s*:', line):
                break
            if re.match(r'^\s*(import|from)\s+\w+', line):
                break
            cleaned_lines.append(line)
        # Collapse excessive blank lines
        result = "\n".join(cleaned_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = "\n".join(("    " + l.strip() if l.strip() else "") for l in result.splitlines())
        # Hard cap lines to avoid pathological long bodies
        max_lines = 128
        result_lines = result.strip().splitlines()
        if len(result_lines) > max_lines:
            result_lines = result_lines[:max_lines]
        return "\n".join(("    " + l.strip() if l.strip() else "") for l in result_lines)
    

    def _extract_body(self, sample: str) -> str:
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
            return sample.strip()

        dedented = [(l[body_indent:] if l.startswith(" " * body_indent) else l.lstrip()) if l.strip() else "" for l in body_lines]
        dedented = strip_trailing_blanks(dedented)

        # ✂️ terminate after first syntactic `return`
        dedented = truncate_after_first_return(dedented)

        code = "\n".join(dedented).strip()
        if not code:
            return sample.strip()
        code = "\n".join(("    " + l.strip() if l.strip() else "") for l in code.splitlines())
        return code


    

    # def prepare_offline_dataset(self) -> Dict[str, List]:
    #     """
    #     Prepare the collected LLM-SR samples into offline GRPO format.
    #     Groups completions by prompt and organizes rewards accordingly.
        
    #     Returns:
    #         Formatted dataset with prompts, completions, and rewards
    #     """
    #     if not self.offline_dataset:
    #         print("No offline samples collected for training")
    #         return {}
        
    #     # Group samples by prompt
    #     prompt_groups = {}
    #     for sample in self.offline_dataset:
    #         prompt = sample['prompt']
    #         instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
    #                          You are given an example of the function signature in the first function below. \
    #                          Your task is to complete the last 'equation' function with your mathematical relationship, considering the physical meaning and relationships of inputs. \
    #                          Only complete the body of the current 'equation' function. Do NOT give me a new function. Just give me the new mathematical relationship in function body. Do NOT use equation_v0 in your implementation.\n\n \
    #                          ")
    #         prompt = '\n'.join([instruction_prompt, prompt])

    #         if prompt not in prompt_groups:
    #             prompt_groups[prompt] = {
    #                 'completions': [],
    #                 'rewards': []
    #             }
    #         prompt_groups[prompt]['completions'].append(sample['completion'])
    #         prompt_groups[prompt]['rewards'].append(sample['reward'])
        
    #     # Format for offline GRPO training
    #     # Note: TRL excludes 'prompt' and 'completion' from reward_kwargs, but NOT 'completions'
    #     # So we use 'completion' (singular) to avoid conflicts
    #     formatted_dataset = {
    #         'prompt': [],
    #         'completion': [],
    #         'rewards': []
    #     }
        
    #     for prompt, group_data in prompt_groups.items():
    #         # Normalize completions to stripped strings for robust matching later
    #         norm_completions = [str(c).strip() for c in group_data['completions']]
    #         formatted_dataset['prompt'].append(prompt)
    #         formatted_dataset['completion'].append(norm_completions)
    #         formatted_dataset['rewards'].append(group_data['rewards'])
        
    #     print(f"Prepared offline dataset: {len(formatted_dataset['prompt'])} unique prompts")
    #     print(f"Total samples: {len(self.offline_dataset)}")
        
    #     all_rewards = [reward for group_rewards in formatted_dataset['rewards'] for reward in group_rewards]
    #     # if all_rewards:
    #     #     print(f"Reward statistics - Mean: {np.mean(all_rewards):.4f}, "
    #     #           f"Std: {np.std(all_rewards):.4f}, "
    #     #           f"Min: {np.min(all_rewards):.4f}, "
    #     #           f"Max: {np.max(all_rewards):.4f}")
        
    #     return formatted_dataset


    def prepare_offline_dataset(self) -> Dict[str, List]:
        """
        Prepare the collected LLM-SR samples into offline GRPO format.
        Groups completions by prompt and organizes rewards accordingly.
        This function accumulates new samples across multiple calls.
        
        Returns:
            Formatted dataset with prompts, completions, and rewards
        """
        if not self.offline_dataset:
            print("No offline samples collected for training")
            return getattr(self, "formatted_dataset", {})

        # Initialize if first time
        if not hasattr(self, "formatted_dataset") or not self.formatted_dataset:
            self.formatted_dataset = {
                'prompt': [],
                'completion': [],
                'rewards': []
            }

        # Convert to dict for faster lookup
        prompt_to_idx = {p: i for i, p in enumerate(self.formatted_dataset['prompt'])}

        instruction_prompt = (
            "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
            "You are given an example of the function signature in the first function below. "
            "Your task is to complete the last 'equation' function with your mathematical relationship, "
            "considering the physical meaning and relationships of inputs. "
            "Only complete the body of the current 'equation' function. Do NOT give me a new function. "
            "Just give me the new mathematical relationship in function body. "
            "Do NOT use equation_v0 in your implementation.\n\n"
        )

        for sample in self.offline_dataset:
            prompt = '\n'.join([instruction_prompt, sample['prompt']])
            completion = str(sample['completion']).strip()
            reward = sample['reward']

            if prompt in prompt_to_idx:
                idx = prompt_to_idx[prompt]
                # Avoid duplicate completions if already stored
                if completion not in self.formatted_dataset['completion'][idx]:
                    self.formatted_dataset['completion'][idx].append(completion)
                    self.formatted_dataset['rewards'][idx].append(reward)
            else:
                # New prompt → add fresh entry
                self.formatted_dataset['prompt'].append(prompt)
                self.formatted_dataset['completion'].append([completion])
                self.formatted_dataset['rewards'].append([reward])
                prompt_to_idx[prompt] = len(self.formatted_dataset['prompt']) - 1

        print(f"Prepared offline dataset: {len(self.formatted_dataset['prompt'])} unique prompts")
        print(f"Total samples: {sum(len(c) for c in self.formatted_dataset['completion'])}")

        all_rewards = [reward for group_rewards in self.formatted_dataset['rewards'] for reward in group_rewards]
        if all_rewards:
            print(f"Offline - Reward statistics - Mean: {np.mean(all_rewards):.4f}, "
                  f"Std: {np.std(all_rewards):.4f}, "
                  f"Min: {np.min(all_rewards):.4f}, "
                  f"Max: {np.max(all_rewards):.4f}")

        return self.formatted_dataset

    
    
    def train_with_offline_grpo(self):
        """
        Train the model using offline GRPO with the collected LLM-SR samples.
        This method uses a custom dataset format that works with default GRPOTrainer.
        """
        if not self.offline_dataset:
            print("No offline training data available for GRPO")
            return
        
        # Prepare dataset in offline format
        dataset = self.prepare_offline_dataset()
        if not dataset:
            return
        
        # breakpoint()
        
        print(f"Starting offline GRPO training episode {self.training_episodes + 1}...")
        
        # For flattened dataset, we don't need to ensure fixed num_generations
        # Just check if we have any data
        if not dataset['completion'] or all(len(c) == 0 for c in dataset['completion']):
            print("No completions available for offline GRPO")
            return
        
        print(f"GRPO config: per_device_train_batch_size={getattr(self.grpo_config,'per_device_train_batch_size', None)}")
        
        # Create a custom dataset that works with default GRPOTrainer
        # We need to flatten the dataset so each row contains one prompt-completion-reward triplet
        flattened_dataset = {
            'prompt': [],
            'completion': [],
            'reward': []
        }
        
        for prompt, completions, rewards in zip(dataset['prompt'], dataset['completion'], dataset['rewards']):
            for completion, reward in zip(completions, rewards):
                flattened_dataset['prompt'].append(prompt)
                flattened_dataset['completion'].append(completion)
                flattened_dataset['reward'].append(reward)
        
        print(f"Flattened dataset: {len(flattened_dataset['prompt'])} total samples")
        

        # Create custom dataset for offline training
        train_dataset = Dataset.from_dict(flattened_dataset)

        # breakpoint()
        
        # Create the reward function that looks up precomputed rewards
        def llmsr_reward_function(prompts, completions, **kwargs):
            print(f"=== LLMSR REWARD FUNCTION CALLED ===")
            print(f"Prompts: {len(prompts)}, Completions: {len(completions)}")
            # print prompts 
            print(f"Prompts: {prompts}")
            
            rewards = []
            for prompt, completion in zip(prompts, completions):
                # Clean the completion to match our stored format
                cleaned_completion = self._clean_completion_text(completion)
                cleaned_completion = self._extract_body(cleaned_completion)
                # breakpoint()
                
                # Compute the reward by evaluating the completion
                try:
                    # Create a temporary program by combining prompt and completion
                    # Extract the function name from the prompt (assuming it ends with a function definition)
                    prompt_lines = prompt.strip().split('\n')
                    function_start = None
                    for i, line in enumerate(prompt_lines):
                        if line.strip().startswith('def '):
                            function_start = i
                            break
                    
                    if function_start is not None:
                        # Find the end of the function definition
                        function_end = None
                        for i in range(function_start + 1, len(prompt_lines)):
                            if prompt_lines[i].strip() == '' or prompt_lines[i].startswith('def '):
                                function_end = i
                                break
                        
                        if function_end is None:
                            function_end = len(prompt_lines)
                        
                        # Combine with the completion to create the full function
                        full_function = '\n'.join(prompt_lines[function_start:]) + '\n' + cleaned_completion
                        
                        # Create a complete program for evaluation
                        complete_program = '\n'.join(prompt_lines[:function_start]) + '\n' + full_function + '\n'
                        
                        
                        # Use actual evaluators if available, otherwise fall back to heuristics
                        if self.evaluators and len(self.evaluators) > 0:
                            # Use the first evaluator for consistency
                            chosen_evaluator = self.evaluators[0]
                            
                            # Create a mock database to capture scores
                            class MockDatabase:
                                def __init__(self):
                                    self.scores_per_test = {}
                                
                                def register_program(self, program, island_id, scores_per_test, **kwargs):
                                    self.scores_per_test = scores_per_test
                            
                            mock_db = MockDatabase()
                            
                            # Temporarily replace the evaluator's database
                            original_db = chosen_evaluator._database
                            chosen_evaluator._database = mock_db
                            
                            try:
                                # Analyze the completion to get scores
                                chosen_evaluator.analyse(
                                    cleaned_completion,
                                    "grpo_eval",  # Mock island_id
                                    0,            # Mock version
                                    global_sample_nums=0,
                                    sample_time=0.0
                                )
                                
                                # Extract scores from the mock database
                                scores_per_test = mock_db.scores_per_test
                                
                                # Calculate score from test results (same logic as in sampler)
                                if scores_per_test:
                                    score = np.mean(list(scores_per_test.values()))
                                    # The score from evaluator is -MSE
                                    mse = -score
                                    # breakpoint()
                                    
                                    # Use exponential decay for MSE to reward mapping (same as in sampler)
                                    if mse is not None and not np.isnan(mse) and not np.isinf(mse):
                                        reward = _llmsr_reward_from_mse(mse)
                                    else:
                                        reward = 0.01
                                else:
                                    # Failed evaluation gets a small floor reward
                                    reward = 0.01
                                    
                            except Exception as e:
                                print(f"Evaluation failed for completion: {e}")
                                reward = 0.01
                            finally:
                                # Restore original database
                                chosen_evaluator._database = original_db
                        
                    else:
                        # No function definition found, use default reward
                        reward = 0.01
                        
                except Exception as e:
                    print(f"Error computing reward for completion: {e}")
                    reward = 0.01
                
                # Ensure reward is in valid range [0.01, 1.0]
                # reward = max(0.01, min(1.0, float(reward)))
                rewards.append(reward)
                
                print(f"Computed reward for completion: {reward:.6f}")
            
            
            # Replace all 0.01 rewards with the minimum of the other (non-0.01) rewards, if any
            # non_floor_rewards = [r for r in rewards if r != 0.01]
            # if non_floor_rewards:
            #     min_non_floor = min(non_floor_rewards)
            #     rewards = [min_non_floor-0.1 if r == 0.01 else r for r in rewards]
            print(f"LLM-SR rewards: {[f'{r:.4f}' for r in rewards]}")
            print(f"Reward statistics - Mean: {np.mean(rewards):.4f}, "
                  f"Std: {np.std(rewards):.4f}, "
                  f"Min: {np.min(rewards):.4f}, "
                  f"Max: {np.max(rewards):.4f}")
            # INSERT_YOUR_CODE
            # Write rewards and statistics to a log file for tracking over iterations
            import os

            log_dir = "./grpo_reward_logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "rewards_log_{self.problem_name}_nprompt{self.n_prompts}.txt")

            with open(log_file, "a") as f:
                f.write(f"Episode {self.training_episodes}\n")
                f.write(f"Rewards: {[f'{r:.4f}' for r in rewards]}\n")
                f.write(f"Reward statistics - Mean: {np.mean(rewards):.4f}, "
                        f"Std: {np.std(rewards):.4f}, "
                        f"Min: {np.min(rewards):.4f}, "
                        f"Max: {np.max(rewards):.4f}\n")
                f.write("-" * 60 + "\n")
            return rewards

        try:
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules='all-linear',
                use_rslora="True",
            )

            # Create a new config for this training episode
            import os
            import wandb

            # Finish any existing WandB run to ensure clean separation
            if wandb.run is not None:
                print(f"Finishing previous WandB run: {wandb.run.name}")
                wandb.finish()

            if os.environ.get("WANDB_API_KEY", "").strip():
                os.environ["WANDB_PROJECT"] = f"llmsr-grpo-{self.problem_name}-adaptive"
                os.environ["WANDB_MODE"] = "online"
            else:
                print("WANDB_API_KEY unset: GRPO report_to=[] (no Weights & Biases logging).")
            
            # Create a fresh config with unique run_name for this episode
            # episode_config = self._create_episode_config(lora_cfg)
            # print(f"Created new GRPO config for episode {self.training_episodes} with run_name: {episode_config.run_name}")
            # print(f"WandB project: {os.environ.get('WANDB_PROJECT', 'Not set')}")
            # print(f"Current WandB run before training: {wandb.run.name if wandb.run else 'None'}")

            if hasattr(self.model, "disable_adapters"):
                try:
                    self.model.disable_adapters()
                except ValueError as e:
                    if "No adapter loaded" not in str(e):
                        raise

            _already_peft = _model_already_has_peft_training(self.model)
            train_peft_cfg = None if _already_peft else lora_cfg

            _peft_for_ref = _unwrap_to_peft_model(self.model)
            if _peft_for_ref is not None and getattr(self.grpo_config, "beta", 0.0) != 0.0:
                if "ref" in _peft_for_ref.peft_config:
                    _peft_for_ref.delete_adapter("ref")

            _mn = self.model_name.replace("/home/jovyan/nly934-storage/models/", "")
            self.grpo_config.run_name = (
                f"V2-{_mn}-prompt{self.n_prompts}-r{lora_cfg.r}-ga{self.grpo_config.gradient_accumulation_steps}"
                f"-ng{self.grpo_config.num_generations}-lr{self.grpo_config.learning_rate}-{int(time.time() * 1000)}"
            )
            self.grpo_trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=[llmsr_reward_function],
                args=self.grpo_config,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
                peft_config=train_peft_cfg,
            )
            # Pre-trainer disable_adapters() turns PEFT layers off; TRL does not re-enable them. Without this,
            # episode 2+ would forward through the frozen base while the optimizer still targets LoRA weights.
            if hasattr(self.model, "enable_adapters"):
                try:
                    self.model.enable_adapters()
                except ValueError as e:
                    if "No adapter loaded" not in str(e):
                        raise
            print(f"WandB run after trainer creation: {wandb.run.name if wandb.run else 'None'}")
            print("Starting GRPO training...")
            self.grpo_trainer.train()
            print("GRPO training completed")

        except Exception as e:
            print(f"Error during offline GRPO training: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up
            self.grpo_trainer = None
            self.model.eval()
            _pm = _unwrap_to_peft_model(self.model)
            if _pm is not None and hasattr(_pm, "enable_adapters"):
                try:
                    _pm.enable_adapters()
                except Exception:
                    pass
            if hasattr(self.model, "enable_adapters"):
                try:
                    self.model.enable_adapters()
                except Exception:
                    pass
            print("Model set to evaluation mode")
            
            # Finish WandB run for this episode
            import wandb
            if os.environ.get("WANDB_API_KEY", "").strip() and wandb.run is not None:
                print(f"Finishing WandB run for episode {self.training_episodes}: {wandb.run.name}")
                wandb.finish()
        
        # Save the trained model
        # self.model.save_pretrained(f"./grpo_checkpoints/offline_episode_{self.training_episodes}")
        
        # Clear offline dataset after training
        self.offline_dataset.clear()
        self.training_episodes += 1
        print(f"Offline training episode {self.training_episodes} completed")
            
        # except Exception as e:
        #     print(f"Error during offline GRPO training: {e}")
        #     import traceback
        #     traceback.print_exc()
            
        #     # Restore original model state even on failure
        #     # self.model.generate = original_generate
        #     self.model.eval()
        #     print("Model state restored after training failure")
    


class OfflineGRPOSampler(Sampler):
    """
    Sampler that collects LLM-SR samples for offline GRPO training.
    Instead of training during sampling, it collects data and trains periodically.
    """
    
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
        
        self.sample_scores = {}  
        self.samples_since_training = 0
        self.mse_history: list[float] = []
        self._reward_history_maxlen: int = 200
        self._reward_min_history: int = 4
        
        if not isinstance(self._llm, OfflineGRPOHuggingFaceLLM):
            print("WARNING: OfflineGRPOSampler requires OfflineGRPOHuggingFaceLLM")
        else:
            # Pass evaluators to the LLM for reward computation
            self._llm.set_evaluators(self._evaluators)
    
    def sample(self, **kwargs):
        """Sample with offline GRPO data collection."""
        while True:
            
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt
            # breakpoint()

            # Process each sample for offline training data collection
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                
                sample_key = f"sample_{cur_global_sample_nums}"
                
                # Pre-register this sample with a default failed state so failures are also collected
                self.sample_scores[sample_key] = {
                    'score': None,
                    'prompt': prompt.code,
                    'completion': sample,
                    'scores_per_test': {}
                }
                
                # Wrap the evaluator to capture scores for offline training
                original_register = self._database.register_program
                def capture_score_register(program, island_id, scores_per_test, **reg_kwargs):
                    # Calculate and store the score
                    score = self._calculate_score_from_tests(scores_per_test)
                    self.sample_scores[sample_key] = {
                        'score': score,
                        'prompt': prompt.code,
                        'completion': sample,
                        'scores_per_test': scores_per_test
                    }
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
                
                # Collect sample for offline training
                self._collect_offline_sample(sample_key)
                self.samples_since_training += 1
            
            # Train after each iteration (exactly one GRPO step per sampling iteration)
            if hasattr(self._llm, 'train_with_offline_grpo'):
                # Ensure we only train on the samples from THIS iteration
                try:
                    n = int(self.samples_since_training)
                    if n > 0 and hasattr(self._llm, 'offline_dataset'):
                        self._llm.offline_dataset = list(self._llm.offline_dataset[-n:])
                except Exception:
                    pass
                
                # breakpoint()
                print(f"Global samples nums: {self.__class__._global_samples_nums}, n_prompts: {self.config.n_prompts}", (self.__class__._global_samples_nums - 1) % self.config.n_prompts)
                if (self.__class__._global_samples_nums - 1) % self.config.n_prompts == 0:
                    print("Triggering offline GRPO training after this iteration...")
                    self._llm.train_with_offline_grpo()
                    self.samples_since_training = 0
                    break
                
                

                # breakpoint()
    

    def _calculate_score_from_tests(self, scores_per_test):
        """Calculate the aggregate score from test scores."""
        if not scores_per_test:
            return 0.0
        return np.mean(list(scores_per_test.values()))
    
    
    def _collect_offline_sample(self, sample_key: str):
        """Collect a sample for offline GRPO training."""
        if sample_key in self.sample_scores:
            sample_data = self.sample_scores[sample_key]
            score = sample_data['score']
            
            if score is not None:
                # The score from evaluator is -MSE
                mse = -score
                # Update history then compute log-normalized reward
                # self._update_mse_history(mse)
                # reward = self._reward_from_mse(mse)
                if mse is not None and not np.isnan(mse) and not np.isinf(mse):
                    reward = _llmsr_reward_from_mse(mse)
                else:
                    reward = 0.01
                # Ensure reward is in valid range [0.01, 1.0]
                # reward = np.clip(float(reward), 0.01, 1.0)
                print(f"Collected offline sample: MSE={mse:.6e}, Score={score:.6e}, Reward={reward:.6f}")
            else:
                # Failed evaluation gets a small floor reward
                reward = 0.01
                print(f"Collected offline sample: FAILED evaluation, Reward={reward:.6f}")
            
            # Always collect sample for offline training (both success and failure)
            if hasattr(self._llm, 'collect_offline_sample'):
                self._llm.collect_offline_sample(
                    prompt=sample_data['prompt'],
                    completion=sample_data['completion'],
                    reward=float(reward)
                )
            
            # Clean up stored score
            del self.sample_scores[sample_key]
        else:
            print(f"Sample {sample_key}: No score data available")
    
    def finalize_training(self):
        """Trigger final offline training with remaining samples."""
        if (self.samples_since_training > 0 and 
            hasattr(self._llm, 'train_with_offline_grpo')):
            print(f"Final offline GRPO training with {self.samples_since_training} samples...")
            self._llm.train_with_offline_grpo()
            self.samples_since_training = 0