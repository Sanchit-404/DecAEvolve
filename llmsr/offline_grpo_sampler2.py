""" Offline GRPO sampler that uses LLM-SR samples as training dataset instead of generating new samples. """
from __future__ import annotations

import numpy as np
import torch
import time
from typing import Sequence, Type, List, Dict, Any
import re
from .sampler import HuggingFaceLLM, Sampler, LLM
from llmsr import evaluator, buffer, config as config_lib
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from dataclasses import dataclass, field
from accelerate import dispatch_model



class OfflineGRPOHuggingFaceLLM(HuggingFaceLLM):
    """
    HuggingFace LLM with offline GRPO training capabilities.
    Uses pre-collected LLM-SR samples as training dataset instead of generating new samples.
    """
    
    def __init__(self, samples_per_prompt: int, model_name: str = None, 
                 batch_inference: bool = True, trim: bool = True, 
                 learning_rate: float = 2e-5) -> None:
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
        
        self._setup_grpo_trainer(learning_rate=learning_rate)
        
        self.offline_dataset = []
        self.training_episodes = 0
        
        print("Offline GRPO-enabled HuggingFace model initialized successfully")
    
    def _setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=32,
            lora_alpha=64,
            target_modules="all-linear",
            lora_dropout=0.05,
            use_rslora="True",
        )
        
        self.model = get_peft_model(self.model, lora_config)
    
    def _setup_grpo_trainer(self, learning_rate=2e-5):
        """Setup GRPO trainer configuration for offline training (version-compatible)."""
        # if torch.cuda.is_available():
        #     optim = "adamw_8bit"
        #     use_bf16 = True
        # else:
        #     # Use standard AdamW for CPU/MPS compatibility
        #     optim = "adamw_torch"
            # use_bf16 = False
        
        # Build kwargs and filter by GRPOConfig signature for compatibility across TRL versions
        import inspect
        lr_scheduler_type: str = "warmup_stable_decay"
        lr_scheduler_kwargs: dict[str, Any] = field(
            default_factory=lambda: dict(num_warmup_steps=200, num_decay_steps=0, min_lr_ratio=0.0)
        )
        lr_scheduler_kwargs_dict = lr_scheduler_kwargs.default_factory()
        token_entropy_percentile_threshold = 0.0 # from https://huggingface/papers/2506.01939
        # loss_type = `bnpo` => helps remove length bias if per_device_train_batch_size > 1
    
        
        cfg_kwargs = {
            'output_dir': "./grpo_checkpoints",
            'learning_rate': learning_rate,
            'lr_scheduler_type': lr_scheduler_type,
            'warmup_steps': lr_scheduler_kwargs_dict["num_warmup_steps"],
            'lr_scheduler_kwargs': {k: v for k,v in lr_scheduler_kwargs_dict.items() if k != "num_warmup_steps"},
            'mask_truncated_completions': False,
            'temperature': 0.8,
            'top_p': 0.9,
            'use_liger_loss': (token_entropy_percentile_threshold == 0.0),
            'per_device_train_batch_size': 32,
            'gradient_accumulation_steps': 1,
            'max_prompt_length': 512,
            'max_completion_length': 512,
            'num_generations': 32,
            # 'optim': optim,
            # 'num_train_epochs': 3,
            # 'bf16': use_bf16,
            # 'remove_unused_columns': False,
            'logging_steps': 1,
            'save_steps': 100,
            'dataloader_num_workers': 0,
            'greater_is_better': True,
            # Ensure finite training when dataloader has no length
            'max_steps': 1,
            'scale_rewards': True, # DR. GRPO => False
            'max_grad_norm': 1.0,
            # 'loss_type': 'dr_grpo', 
            'beta': 0.05,
            'epsilon': 0.2,
            'disable_dropout': True,
            'report_to': "wandb",
            #vllm
            # 'use_vllm': True,
            # 'vllm_host': "localhost",
            # 'vllm_port': 8000,
            # "vllm_mode": "colocate", 
            # "vllm_server_timeout": 1200
        }

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
        # Hard cap lines to avoid pathological long bodies
        max_lines = 128
        result_lines = result.strip().splitlines()
        if len(result_lines) > max_lines:
            result_lines = result_lines[:max_lines]
        return "\n".join(result_lines)
    
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
        
        
    #     # Calculate reward statistics
    #     all_rewards = [reward for group_rewards in formatted_dataset['rewards'] for reward in group_rewards]
    #     if all_rewards:
    #         print(f"Reward statistics - Mean: {np.mean(all_rewards):.4f}, "
    #               f"Std: {np.std(all_rewards):.4f}, "
    #               f"Min: {np.min(all_rewards):.4f}, "
    #               f"Max: {np.max(all_rewards):.4f}")
        
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
        This method follows the VikhrModels approach of using pre-computed samples.
        """
        if not self.offline_dataset:
            print("No offline training data available for GRPO")
            return
        
        # Prepare dataset in offline format
        dataset = self.prepare_offline_dataset()
        if not dataset:
            return
        
        # try:
        print(f"Starting offline GRPO training episode {self.training_episodes + 1}...")
        
        # Ensure a fixed num_generations across all prompts: use the minimum available
        effective_num_generations = min(len(c) for c in dataset['completion'])
        if effective_num_generations < 1:
            print("No completions available for offline GRPO")
            return
        # Slice each prompt's completions/rewards to the same length
        for i in range(len(dataset['completion'])):
            dataset['completion'][i] = dataset['completion'][i][:effective_num_generations]
            dataset['rewards'][i] = dataset['rewards'][i][:effective_num_generations]
        

        print(f"Effective num_generations set to {getattr(self.grpo_config,'num_generations', None)}")
        print(f"GRPO config: per_device_train_batch_size={getattr(self.grpo_config,'per_device_train_batch_size', None)}, num_generations={getattr(self.grpo_config,'num_generations', None)}")
        
        # Create custom dataset for offline training
        train_dataset = Dataset.from_dict(dataset)
        
        # Prepare our LLM-SR samples for GRPO monkey-patching
        self.precomputed_samples = {}
        self.precomputed_rewards = {}

        for prompt, completions, rewards in zip(dataset['prompt'], dataset['completion'], dataset['rewards']):
            # Store normalized texts and rewards
            norm_completions = [str(c).strip() for c in completions]
            self.precomputed_samples[prompt] = norm_completions
            self.precomputed_rewards[prompt] = rewards
        
        print(f"GRPO setup: {len(self.precomputed_samples)} unique prompts with LLM-SR samples")
        
        # PRESERVE original model state before GRPO training
        original_generate = self.model.generate
        # original_generate = types.MethodType(self.model.generate.__func__, self.model)
        original_generation_config = getattr(self.model, 'generation_config', None)
        original_class_generate = self.model.__class__.generate
        
        sample_counter = 0  # Track which samples to return
        rng = np.random.default_rng()
        self._selected_rewards: List[float] = []
        
        
        # def patched_generate(input_ids, **kwargs):
        #     nonlocal sample_counter
        #     # Decode the prompt from input_ids
        #     prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        #     prompt_text = prompt_text.strip()
            
        #     # Find if decoded prompt is in our precomputed samples
        #     matching_prompt = None
        #     for stored_prompt in self.precomputed_samples.keys():
        #         if stored_prompt.strip() in prompt_text or prompt_text in stored_prompt.strip():
        #             matching_prompt = stored_prompt
        #             break
            
        #     if matching_prompt and matching_prompt in self.precomputed_samples:
        #         # Use our LLM-SR precomputed completions and rewards instead of generating again from input_ids
        #         completions = self.precomputed_samples[matching_prompt]

        #         num_generations = getattr(self.grpo_config, 'num_generations', 8)
        #         selected_completions = []
        #         selected_rewards = []
        #         for i in range(num_generations):
        #             idx = (sample_counter + i) % max(1, len(completions))
        #             selected_completions.append(completions[idx])
        #             selected_rewards.append(float(self.precomputed_rewards[matching_prompt][idx]))
                
        #         sample_counter += num_generations
                
        #         # Get output_ids of completions with max_comp_len
        #         base_prompt_ids = input_ids[0].to("cuda")
        #         max_comp_len = getattr(self.grpo_config, 'max_completion_length', 512)
        #         batch_outputs = []
        #         for completion in selected_completions:
        #             comp_ids = self.tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        #             comp_ids = comp_ids[:max_comp_len]
        #             out_ids = torch.cat([base_prompt_ids, comp_ids.to("cuda")], dim=0)
        #             batch_outputs.append(out_ids)
                
        #         # Pad to common length across batch with EOS for stacking
        #         eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        #         max_len = max(seq.size(0) for seq in batch_outputs)
        #         padded_outputs = []
        #         for seq in batch_outputs:
        #             if seq.size(0) < max_len:
        #                 pad_len = max_len - seq.size(0)
        #                 padding = torch.full((pad_len,), eos_id, dtype=seq.dtype).to("cuda")
        #                 seq = torch.cat([seq, padding], dim=0)
        #             padded_outputs.append(seq)

        #         result = torch.stack(padded_outputs, dim=0)
        #         # Store rewards for reward function to read back
        #         self._selected_rewards = selected_rewards
        #         print(f"Using LLM-SR samples: {len(selected_completions)} completions for prompt")
        #         return result
                

        def patched_generate(input_ids, **kwargs):
            # Decode the prompt from input_ids
            prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()

            # Find the matching stored prompt
            matching_prompt = None
            for stored_prompt in self.precomputed_samples.keys():
                if stored_prompt.strip() in prompt_text or prompt_text in stored_prompt.strip():
                    matching_prompt = stored_prompt
                    break

            if matching_prompt and matching_prompt in self.precomputed_samples:
                completions = self.precomputed_samples[matching_prompt]
                rewards = self.precomputed_rewards[matching_prompt]

                num_generations = getattr(self.grpo_config, 'num_generations', 8)

                # Instead of cycling deterministically, pick random subset with replacement
                chosen_idxs = rng.choice(len(completions), size=num_generations, replace=True)
                selected_completions = [completions[i] for i in chosen_idxs]
                selected_rewards = [float(rewards[i]) for i in chosen_idxs]

                # Save rewards for reward function
                self._selected_rewards = selected_rewards

                # Convert to tokenized outputs
                base_prompt_ids = input_ids[0].to("cuda")
                max_comp_len = getattr(self.grpo_config, 'max_completion_length', 512)

                batch_outputs = []
                eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                for completion in selected_completions:
                    comp_ids = self.tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
                    comp_ids = comp_ids[:max_comp_len]
                    out_ids = torch.cat([base_prompt_ids, comp_ids.to("cuda")], dim=0)
                    batch_outputs.append(out_ids)

                # Pad outputs to common length
                max_len = max(seq.size(0) for seq in batch_outputs)
                padded_outputs = []
                for seq in batch_outputs:
                    if seq.size(0) < max_len:
                        pad_len = max_len - seq.size(0)
                        padding = torch.full((pad_len,), eos_id, dtype=seq.dtype, device=seq.device)
                        seq = torch.cat([seq, padding], dim=0)
                    padded_outputs.append(seq)

                return torch.stack(padded_outputs, dim=0)

            # If no match, fall back to normal generation
            return original_generate(input_ids, **kwargs)
        
        # Create the proper reward function
        def llmsr_reward_function(prompts, completions, **kwargs):
            print(f"=== LLMSR REWARD FUNCTION CALLED ===")
            print(f"Prompts: {len(prompts)}, Completions: {len(completions)}")
            rewards = [float(r) for r in self._selected_rewards[:len(completions)]]
            # rewards = [max(0.01, min(1.0, r)) for r in rewards]
            print(f"LLM-SR rewards (direct aligned): {[f'{r:.4f}' for r in rewards]}")
            return rewards

        print("Generation Config", self.model.generation_config)
        
        try:
            # Apply the monkey patch
            self.model.generate = patched_generate
            # print("Monkey-patched model.generate to use LLM-SR samples")

            lora_cfg = LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                target_modules='all-linear',
                use_rslora="True",
            )

            import os
            os.environ["WANDB_PROJECT"] = f"llmsr-grpo-oscillator1-single-gpu"
            self.grpo_config.run_name = f"r{lora_cfg.r}-g{self.grpo_config.num_generations}-{int(time.time())}"

            self.grpo_trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=[llmsr_reward_function],
                args=self.grpo_config,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
                peft_config=lora_cfg,
            )
            print("Starting GRPO training...")
            self.grpo_trainer.train()
            print("GRPO training completed")

        
        finally:
            self.model.generate = original_generate
            self.model.__class__.generate = original_class_generate
            if original_generation_config is not None:
                self.model.generation_config = original_generation_config
            print("Restored original model.generate method after GRPO training")
            
            self.grpo_trainer = None
            self.model.eval()
            print("Model set to evaluation mode")
            print("Is generate patched?", "patched" in repr(self.model.generate))
            print("Generation Config", self.model.generation_config)
            # out1 = self.model.generate(self.tokenizer("test prompt", return_tensors="pt").input_ids.to('cuda'), 
            #                do_sample=True, temperature=1.0, top_p=0.9, max_new_tokens=50)
            # out2 = self.model.generate(self.tokenizer("test prompt", return_tensors="pt").input_ids.to('cuda'), 
            #                         do_sample=True, temperature=1.0, top_p=0.9, max_new_tokens=50)
            # print("Completion 1:", self.tokenizer.decode(out1[0]))
            # print("Completion 2:", self.tokenizer.decode(out2[0]))

            # breakpoint()
        
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
    
    def sample(self, **kwargs):
        """Sample with offline GRPO data collection."""
        while True:
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt
            breakpoint()

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
                
                original_generate = self.model.generate
                original_generation_config = getattr(self.model, 'generation_config', None)
                
                print("Triggering offline GRPO training after this iteration...")
                self._llm.train_with_offline_grpo()
                
                self.model.generate = original_generate
                if original_generation_config is not None:
                    self.model.generation_config = original_generation_config
                
                self.samples_since_training = 0
    

    def _calculate_score_from_tests(self, scores_per_test):
        """Calculate the aggregate score from test scores."""
        if not scores_per_test:
            return 0.0
        return np.mean(list(scores_per_test.values()))
    
    def _update_mse_history(self, mse: float) -> None:
        """Append MSE to rolling history with a bounded size."""
        if mse is None:
            return
        self.mse_history.append(float(mse))
        if len(self.mse_history) > self._reward_history_maxlen:
            self.mse_history.pop(0)
    
    def _reward_from_mse(self, mse: float) -> float:
        """Compute log-normalized reward in [0, 1] from MSE with robust fallbacks."""
        if mse is None:
            return 0.01
        if mse <= 0:
            return 1.0
        eps = 1e-12
        history = self.mse_history + [float(mse)] if self.mse_history else [float(mse)]
        if len(history) >= self._reward_min_history:
            p10, p90 = np.percentile(history, [10, 90])
            if p90 <= p10:
                p90 = p10 * 1.0001 + eps
            num = np.log10(p90 + eps) - np.log10(mse + eps)
            den = np.log10(p90 + eps) - np.log10(p10 + eps)
            reward = num / den
            reward = float(np.clip(reward, 0.0, 1.0))
        else:
            c = float(np.median(history)) + eps
            alpha = 0.5
            reward = 1.0 / (1.0 + (mse / c) ** alpha)
            reward = float(np.clip(reward, 0.0, 1.0))
        return reward
    
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
                    # Use exponential decay for MSE to reward mapping
                    reward = np.exp(-np.clip(abs(mse), 0, 10))  # Clip MSE to reasonable range
                else:
                    reward = 0.01
                # Ensure reward is in valid range [0.01, 1.0]
                reward = np.clip(float(reward), 0.01, 1.0)
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