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

# Add wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("WARNING: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False


class OfflineGRPOHuggingFaceLLM(HuggingFaceLLM):
    """
    HuggingFace LLM with offline GRPO training capabilities.
    Uses pre-collected LLM-SR samples as training dataset instead of generating new samples.
    """
    
    def __init__(self, samples_per_prompt: int, model_name: str = None, 
                 batch_inference: bool = True, trim: bool = True, 
                 learning_rate: float = 1e-6, wandb_project: str = "llmsr2l", 
                 wandb_run_name: str = None, use_wandb: bool = True, 
                 use_distributed: bool = True, max_memory: dict = None) -> None:
        """
        Initialize offline GRPO-enabled HuggingFace model.
        """
        # Store LoRA setup flag to handle it after model loading
        self._needs_lora_setup = True
        
        # Initialize parent class with distributed setting
        super().__init__(samples_per_prompt, model_name, batch_inference, trim, 
                         use_distributed, max_memory)
        
        # Handle MPS compatibility
        try:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print("########################")
            print(f"Number of GPUs available: {num_gpus}")
            if num_gpus > 0:
                for i in range(num_gpus):
                    mem = torch.cuda.get_device_properties(i).total_memory
                    mem_gb = mem / (1024 ** 3)
                    print(f"GPU {i}: {mem_gb:.2f} GB total memory")
            print("########################")
            if (not torch.cuda.is_available()) and torch.backends.mps.is_available():
                self.model = self.model.to(torch.float32).to("mps")
        except Exception:
            pass
        
        self._setup_grpo_trainer(learning_rate=learning_rate)
        
        self.offline_dataset = []
        self.training_episodes = 0
        
        # Initialize wandb tracking
        self.wandb_initialized = False
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name or f"grpo-{model_name or 'model'}-{int(time.time())}"
        self.episode_rewards = []  # Track rewards per episode
        self.episode_metrics = []  # Track detailed metrics per episode
        self.use_wandb = use_wandb
        
        if WANDB_AVAILABLE and self.use_wandb:
            self._init_wandb()
        
        print("Offline GRPO-enabled HuggingFace model initialized successfully")
    
    def get_generation_model(self):
        """Get the model for generation, handling DDP wrapping."""
        if hasattr(self.model, 'module'):
            # If model is wrapped (e.g., in DDP), get the base model for generation
            return self.model.module
        else:
            return self.model
    
    def _setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        print("Setting up LoRA for model fine-tuning...")
        
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=64,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0.1,
        )
        
        # Ensure model is not wrapped in DDP before applying LoRA
        base_model = self.model
        if hasattr(self.model, 'module'):
            # If model is wrapped (e.g., in DDP), get the base model
            base_model = self.model.module
        
        try:
            self.model = get_peft_model(base_model, lora_config)
            print("LoRA setup completed successfully")
        except Exception as e:
            print(f"Warning: Failed to setup LoRA: {e}")
            print("Continuing without LoRA fine-tuning")
    
    def _setup_grpo_trainer(self, learning_rate=1e-6):
        """Setup GRPO trainer configuration for offline training (version-compatible)."""
        # Use standard AdamW for CPU/MPS compatibility
        optim = "adamw_torch"
        use_bf16 = False
        
        # Build kwargs and filter by GRPOConfig signature for compatibility across TRL versions
        import inspect
        cfg_kwargs = {
            'output_dir': "./grpo_checkpoints",
            'learning_rate': learning_rate,
            'per_device_train_batch_size': 64,
            'gradient_accumulation_steps': 1,
            'max_prompt_length': 512,
            'max_completion_length': 1024,
            'num_generations': 64,
            'optim': optim,
            'num_train_epochs': 3,
            'bf16': use_bf16,
            'remove_unused_columns': False,
            'logging_steps': 1,
            'save_steps': 100,
            'dataloader_num_workers': 0,
            'report_to': [],
            'greater_is_better': True,
            # Ensure finite training when dataloader has no length
            'max_steps': 1,
            'scale_rewards': True,
            'max_grad_norm': 1.0,
            'loss_type': 'dr_grpo',
            'beta': 0.02,
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
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        if not WANDB_AVAILABLE:
            return
            
        wandb.init(
            project=self.wandb_project,
            name=self.wandb_run_name,
            config={
                "model_name": self.model_name,
                "learning_rate": getattr(self.grpo_config, 'learning_rate', 1e-6),
                "batch_size": getattr(self.grpo_config, 'per_device_train_batch_size', 8),
                "num_epochs": getattr(self.grpo_config, 'num_train_epochs', 3),
                "max_prompt_length": getattr(self.grpo_config, 'max_prompt_length', 512),
                "max_completion_length": getattr(self.grpo_config, 'max_completion_length', 512),
                "samples_per_prompt": self._samples_per_prompt,
            },
            tags=["llmsr2l", "grpo", "offline"],
            reinit=True
        )
        self.wandb_initialized = True
        print(f"Wandb initialized: {self.wandb_project}/{self.wandb_run_name}")
    
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
    
    def prepare_offline_dataset(self) -> Dict[str, List]:
        """
        Prepare the collected LLM-SR samples into offline GRPO format.
        Groups completions by prompt and organizes rewards accordingly.
        
        Returns:
            Formatted dataset with prompts, completions, and rewards
        """
        if not self.offline_dataset:
            print("No offline samples collected for training")
            return {}
        
        # Group samples by prompt
        prompt_groups = {}
        for sample in self.offline_dataset:
            prompt = sample['prompt']
            if prompt not in prompt_groups:
                prompt_groups[prompt] = {
                    'completions': [],
                    'rewards': []
                }
            prompt_groups[prompt]['completions'].append(sample['completion'])
            prompt_groups[prompt]['rewards'].append(sample['reward'])
        
        # Format for offline GRPO training
        # Note: TRL excludes 'prompt' and 'completion' from reward_kwargs, but NOT 'completions'
        # So we use 'completion' (singular) to avoid conflicts
        formatted_dataset = {
            'prompt': [],
            'completion': [],
            'rewards': []
        }
        
        for prompt, group_data in prompt_groups.items():
            # Normalize completions to stripped strings for robust matching later
            norm_completions = [str(c).strip() for c in group_data['completions']]
            formatted_dataset['prompt'].append(prompt)
            formatted_dataset['completion'].append(norm_completions)
            formatted_dataset['rewards'].append(group_data['rewards'])
        
        print(f"Prepared offline dataset: {len(formatted_dataset['prompt'])} unique prompts")
        print(f"Total samples: {len(self.offline_dataset)}")
        
        # PRINT DETAILED GRPO TRAINING INPUTS
        print("\n=== GRPO TRAINING INPUTS ===")
        for i, (prompt, completions, rewards) in enumerate(zip(formatted_dataset['prompt'], formatted_dataset['completion'], formatted_dataset['rewards'])):
            print(f"  Number of completions: {len(completions)}")
            failed_count = sum(1 for reward in rewards if abs(reward - 0.01) < 1e-6)
            print(f"  Failed samples (reward=0.01): {failed_count}/{len(completions)}")
            for j, (completion, reward) in enumerate(zip(completions, rewards)):
                print(f"    Completion {j+1}: '{completion.strip()}' → Reward: {reward:.9f}")
        print("=== END GRPO TRAINING INPUTS ===\n")
        
        # Calculate reward statistics
        all_rewards = [reward for group_rewards in formatted_dataset['rewards'] for reward in group_rewards]
        if all_rewards:
            print('All rewards: ', all_rewards)
            print(f"Reward statistics - Mean: {np.mean(all_rewards):.9f}, "
                  f"Std: {np.std(all_rewards):.9f}, "
                  f"Min: {np.min(all_rewards):.9f}, "
                  f"Max: {np.max(all_rewards):.9f}")
        
        return formatted_dataset
    
    def _log_episode_metrics(self, episode_metrics: Dict[str, Any]):
        """Log episode metrics to wandb."""
        if not self.wandb_initialized or not WANDB_AVAILABLE or not self.use_wandb:
            return
            
        try:
            # Add episode number to metrics
            episode_metrics['episode'] = self.training_episodes
            episode_metrics['timestamp'] = time.time()
            
            # Log to wandb
            wandb.log(episode_metrics)
            
            # Store locally for potential analysis
            self.episode_metrics.append(episode_metrics)
            
            print(f"Logged episode {self.training_episodes} metrics to wandb")
        except Exception as e:
            print(f"Failed to log metrics to wandb: {e}")
    
    
    def _extract_training_metrics(self) -> Dict[str, Any]:
        """Extract training metrics from the GRPO trainer for logging."""
        metrics = {
            'episode': self.training_episodes,
            'samples_used': len(self.offline_dataset),
            'unique_prompts': len(set(sample['prompt'] for sample in self.offline_dataset)),
        }
        
        # Extract metrics from trainer state if available
        if hasattr(self, 'grpo_trainer') and self.grpo_trainer is not None:
            try:
                # Get trainer state
                trainer_state = getattr(self.grpo_trainer, 'state', None)
                if trainer_state and hasattr(trainer_state, 'log_history') and trainer_state.log_history:
                    # Get the latest log entry
                    latest_log = trainer_state.log_history[-1]
                    
                    # Extract key metrics
                    metrics.update({
                        'grpo_loss': latest_log.get('loss', 0.0),
                        'grpo_reward_mean': latest_log.get('reward', 0.0),
                        'grpo_reward_std': latest_log.get('reward_std', 0.0),
                        'grpo_entropy': latest_log.get('entropy', 0.0),
                        'grpo_kl_divergence': latest_log.get('kl', 0.0),
                        'grpo_grad_norm': latest_log.get('grad_norm', 0.0),
                        'grpo_learning_rate': latest_log.get('learning_rate', 0.0),
                        'grpo_epoch': latest_log.get('epoch', 0.0),
                        'grpo_step': latest_log.get('step', 0),
                        'grpo_num_tokens': latest_log.get('num_tokens', 0.0),
                    })
                    
                    # Extract completion statistics
                    metrics.update({
                        'completions_mean_length': latest_log.get('completions/mean_length', 0.0),
                        'completions_max_length': latest_log.get('completions/max_length', 0.0),
                        'completions_min_length': latest_log.get('completions/min_length', 0.0),
                        'completions_clipped_ratio': latest_log.get('completions/clipped_ratio', 0.0),
                    })
                    
                    # Extract reward function specific metrics
                    for key, value in latest_log.items():
                        if key.startswith('rewards/') and key.endswith('/mean'):
                            reward_name = key.replace('rewards/', '').replace('/mean', '')
                            metrics[f'reward_{reward_name}_mean'] = value
                        elif key.startswith('rewards/') and key.endswith('/std'):
                            reward_name = key.replace('rewards/', '').replace('/std', '')
                            metrics[f'reward_{reward_name}_std'] = value
                    
            except Exception as e:
                print(f"Failed to extract trainer metrics: {e}")
        
        # Calculate offline dataset statistics
        if self.offline_dataset:
            rewards = [sample['reward'] for sample in self.offline_dataset]
            metrics.update({
                'offline_reward_mean': np.mean(rewards),
                'offline_reward_std': np.std(rewards),
                'offline_reward_min': np.min(rewards),
                'offline_reward_max': np.max(rewards),
                'offline_reward_median': np.median(rewards),
            })
            
            # Calculate GRPO advantages (normalized rewards within groups)
            grpo_advantages = self._calculate_grpo_advantages()
            if grpo_advantages:
                metrics.update({
                    'grpo_advantage_mean': np.mean(grpo_advantages),
                    'grpo_advantage_std': np.std(grpo_advantages),
                    'grpo_advantage_min': np.min(grpo_advantages),
                    'grpo_advantage_max': np.max(grpo_advantages),
                    'grpo_advantage_median': np.median(grpo_advantages),
                })
                print(f"GRPO advantages - Mean: {np.mean(grpo_advantages):.9f}, "
                      f"Std: {np.std(grpo_advantages):.9f}, "
                      f"Min: {np.min(grpo_advantages):.9f}, "
                      f"Max: {np.max(grpo_advantages):.9f}")
        
        return metrics
    
    def _calculate_grpo_advantages(self) -> List[float]:
        """
        Calculate GRPO advantages (normalized rewards within each prompt group).
        
        In GRPO, advantages are computed as rewards normalized within each group
        of completions for the same prompt. This helps the model learn relative
        preferences within each group.
        
        Returns:
            List of advantage values for all samples
        """
        if not self.offline_dataset:
            return []
        
        # Group samples by prompt
        prompt_groups = {}
        for sample in self.offline_dataset:
            prompt = sample['prompt']
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append(sample['reward'])
        
        advantages = []
        
        for prompt, rewards in prompt_groups.items():
            # Convert to numpy array for calculations
            rewards_array = np.array(rewards, dtype=np.float32)
            
            # Calculate advantage as reward minus mean reward in the group
            # This is the standard advantage calculation in GRPO
            mean_reward = np.mean(rewards_array)
            group_advantages = rewards_array - mean_reward
            
            # Alternative: normalize by standard deviation (z-score normalization)
            std_reward = np.std(rewards_array)
            if std_reward > 0:
                group_advantages = (rewards_array - mean_reward) / std_reward
            else:
                group_advantages = rewards_array - mean_reward
            
            advantages.extend(group_advantages.tolist())
        
        return advantages
    
    def _finalize_wandb(self):
        """Finalize wandb logging and close the run."""
        if self.wandb_initialized and WANDB_AVAILABLE and self.use_wandb:
            try:
                # Log final summary statistics
                if self.episode_metrics:
                    final_summary = {
                        'final_episode': self.training_episodes,
                        'total_episodes': len(self.episode_metrics),
                        'avg_grpo_loss': np.mean([m.get('grpo_loss', 0.0) for m in self.episode_metrics]),
                        'avg_grpo_reward': np.mean([m.get('grpo_reward_mean', 0.0) for m in self.episode_metrics]),
                        'best_grpo_reward': max([m.get('grpo_reward_mean', 0.0) for m in self.episode_metrics]),
                        'worst_grpo_reward': min([m.get('grpo_reward_mean', 0.0) for m in self.episode_metrics]),
                        'avg_grpo_advantage': np.mean([m.get('grpo_advantage_mean', 0.0) for m in self.episode_metrics]),
                        'best_grpo_advantage': max([m.get('grpo_advantage_max', 0.0) for m in self.episode_metrics]),
                        'worst_grpo_advantage': min([m.get('grpo_advantage_min', 0.0) for m in self.episode_metrics]),
                    }
                    wandb.log(final_summary)
                    print(f"Logged final summary to wandb: {final_summary}")
                
                wandb.finish()
                print("Wandb run finalized successfully")
            except Exception as e:
                print(f"Failed to finalize wandb: {e}")
    
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
        
        try:
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
            # Align GRPO expected num_generations and ensure non-zero effective batch sizing
            try:
                # Respect configured num_generations but do not exceed available completions
                requested_generations = int(getattr(self.grpo_config, 'num_generations', effective_num_generations) or effective_num_generations)
                final_num_generations = max(1, min(requested_generations, effective_num_generations))
                self.grpo_config.num_generations = final_num_generations
                # Ensure per_device_train_batch_size >= num_generations (use equal by default)
                if hasattr(self.grpo_config, 'per_device_train_batch_size'):
                    desired_pdbs = max(int(getattr(self.grpo_config, 'per_device_train_batch_size', final_num_generations)), final_num_generations)
                    self.grpo_config.per_device_train_batch_size = desired_pdbs
                # Extra guard for versions that derive internal batch_size via integer division
                ng = getattr(self.grpo_config, 'num_generations', final_num_generations)
                pdbs = getattr(self.grpo_config, 'per_device_train_batch_size', final_num_generations)
                if ng and (pdbs // ng) == 0:
                    self.grpo_config.per_device_train_batch_size = ng
                # Ensure gradient_accumulation_steps is valid
                if hasattr(self.grpo_config, 'gradient_accumulation_steps'):
                    self.grpo_config.gradient_accumulation_steps = max(int(getattr(self.grpo_config, 'gradient_accumulation_steps', 1)), 1)
            except Exception:
                pass
            print(f"Effective num_generations set to {getattr(self.grpo_config,'num_generations', None)}")
            print(f"GRPO config: per_device_train_batch_size={getattr(self.grpo_config,'per_device_train_batch_size', None)}, num_generations={getattr(self.grpo_config,'num_generations', None)}")
            
            # Create custom dataset for offline training
            train_dataset = Dataset.from_dict(dataset)
            
            # Prepare our LLM-SR samples for GRPO monkey-patching
            # Group completions by prompt for easy lookup
            self.precomputed_samples = {}
            self.precomputed_rewards = {}
            self.precomputed_ids = {}

            # Cache EOS/PAD once
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id

            for prompt, completions, rewards in zip(dataset['prompt'], dataset['completion'], dataset['rewards']):
                # Store normalized texts and rewards
                norm_completions = [str(c).strip() for c in completions]
                self.precomputed_samples[prompt] = norm_completions
                self.precomputed_rewards[prompt] = rewards

                # Pre-tokenize completion-only ids for exact matching PER prompt
                ids_list = []
                for comp in norm_completions:
                    enc = self.tokenizer(comp, return_tensors="pt", add_special_tokens=False)
                    ids = enc.input_ids[0].tolist()
                    # Trim trailing eos/pad
                    while ids and (ids[-1] == eos_id or (pad_id is not None and ids[-1] == pad_id)):
                        ids.pop()
                    ids_list.append(tuple(ids))
                self.precomputed_ids[prompt] = ids_list
            
            print(f"GRPO setup: {len(self.precomputed_samples)} unique prompts with LLM-SR samples")
            
            # PRESERVE original model state before GRPO training
            generation_model = self.get_generation_model() if hasattr(self, 'get_generation_model') else self.model
            original_generate = generation_model.generate
            original_generation_config = getattr(generation_model, 'generation_config', None)
            sample_counter = 0  # Track which samples to return
            # Will store rewards aligned with the last batch of generated completions
            self._selected_rewards: List[float] = []
            
            def patched_generate(input_ids, **kwargs):
                nonlocal sample_counter
                
                # Decode the prompt to find our pre-computed samples
                prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                prompt_text = prompt_text.strip()
                
                # Find matching prompt in our precomputed samples
                matching_prompt = None
                for stored_prompt in self.precomputed_samples.keys():
                    if stored_prompt.strip() in prompt_text or prompt_text in stored_prompt.strip():
                        matching_prompt = stored_prompt
                        break
                
                if matching_prompt and matching_prompt in self.precomputed_samples:
                    # Use our LLM-SR samples instead of generating
                    completions = self.precomputed_samples[matching_prompt]
                    
                    # GRPO expects multiple completions; always return requested by cycling
                    requested = kwargs.get('num_return_sequences', getattr(self.grpo_config, 'num_generations', 16))
                    num_generations = max(1, int(requested))
                    selected_completions = []
                    selected_rewards = []
                    
                    for i in range(num_generations):
                        idx = (sample_counter + i) % max(1, len(completions))
                        selected_completions.append(completions[idx])
                        # Align reward selection with completion cycling
                        selected_rewards.append(float(self.precomputed_rewards[matching_prompt][idx]))
                    
                    sample_counter += num_generations
                    
                    # Build completion-only ids and append to the given prompt ids
                    # Get device from the actual model that will be used
                    if hasattr(self.model, 'module'):
                        target_device = next(self.model.module.parameters()).device
                    else:
                        target_device = next(self.model.parameters()).device
                    
                    base_prompt_ids = input_ids[0].to(target_device)
                    max_comp_len = getattr(self.grpo_config, 'max_completion_length', 512)
                    batch_outputs = []
                    for completion in selected_completions:
                        cleaned_completion = self._clean_completion_text(completion)
                        comp_ids = self.tokenizer(cleaned_completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
                        comp_ids = comp_ids[:max_comp_len]
                        out_ids = torch.cat([base_prompt_ids, comp_ids.to(target_device)], dim=0)
                        batch_outputs.append(out_ids)
                    # Pad to common length across batch with EOS for stacking
                    eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                    max_len = max(seq.size(0) for seq in batch_outputs)
                    padded_outputs = []
                    for seq in batch_outputs:
                        if seq.size(0) < max_len:
                            pad_len = max_len - seq.size(0)
                            padding = torch.full((pad_len,), eos_id, device=target_device, dtype=seq.dtype)
                            seq = torch.cat([seq, padding], dim=0)
                        padded_outputs.append(seq)
                    result = torch.stack(padded_outputs, dim=0)
                    # Store rewards for reward function to read back
                    self._selected_rewards = selected_rewards
                    print(f"Using LLM-SR samples: {len(selected_completions)} completions for prompt")
                    return result
                    
                else:
                    # Strict offline mode: never generate new samples; pull from any stored prompt
                    if getattr(self, 'strict_offline', True) and len(self.precomputed_samples) > 0:
                        any_prompt = next(iter(self.precomputed_samples.keys()))
                        completions = self.precomputed_samples[any_prompt]
                        requested = kwargs.get('num_return_sequences', getattr(self.grpo_config, 'num_generations', 16))
                        num_generations = max(1, int(requested))
                        selected_completions = []
                        selected_rewards = []
                        for i in range(num_generations):
                            idx = (sample_counter + i) % max(1, len(completions))
                            selected_completions.append(completions[idx])
                            selected_rewards.append(float(self.precomputed_rewards[any_prompt][idx]))
                        sample_counter += num_generations
                        # Tokenize prompt+completion and return stacked ids
                        batch_outputs = []
                        for completion in selected_completions:
                            eos = self.tokenizer.eos_token or ""
                            full_text = (prompt_text + completion + (" " + eos if eos and not completion.strip().endswith(eos) else "")).strip()
                            tokenized = self.tokenizer(full_text, return_tensors="pt", 
                                                     max_length=kwargs.get('max_length', 512),
                                                     truncation=True, padding=False)
                            batch_outputs.append(tokenized.input_ids[0])
                        max_len = max(len(seq) for seq in batch_outputs)
                        padded_outputs = []
                        
                        # Get device from the actual model
                        if hasattr(self.model, 'module'):
                            target_device = next(self.model.module.parameters()).device
                        else:
                            target_device = next(self.model.parameters()).device
                            
                        for seq in batch_outputs:
                            if len(seq) < max_len:
                                pad_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                                padding = torch.full((max_len - len(seq),), pad_id, 
                                                   device=target_device, dtype=seq.dtype)
                                seq = torch.cat([seq.to(target_device), padding])
                            else:
                                seq = seq.to(target_device)
                            padded_outputs.append(seq)
                        result = torch.stack(padded_outputs).to(target_device)
                        # Store rewards for reward function to read back
                        self._selected_rewards = selected_rewards
                        print("Strict offline: returning completions from stored LLM-SR samples")
                        return result
                    else:
                        # Fallback to original generation if prompt not found and not strict offline
                        print(f"WARNING: Prompt not found in LLM-SR samples, using original generation")
                        return original_generate(input_ids, **kwargs)
            
            # Apply the monkey patch to the generation model
            generation_model.generate = patched_generate
            print("Monkey-patched model.generate to use LLM-SR samples")
            
            # Create the proper reward function
            def llmsr_reward_function(prompts, completions, **kwargs):
                """
                Real LLM-SR reward function using pre-computed rewards.
                Return rewards aligned exactly to the last generated completions batch.
                """
                print(f"=== LLMSR REWARD FUNCTION CALLED ===")
                print(f"Prompts: {len(prompts)}, Completions: {len(completions)}")
                print(f"Additional kwargs: {list(kwargs.keys())}")
                # If we have stored rewards aligned to generated completions, return them directly
                if hasattr(self, '_selected_rewards') and self._selected_rewards:
                    rewards = [float(r) for r in self._selected_rewards[:len(completions)]] 
                    print(f"REAL LLM-SR rewards (direct aligned): {[f'{r:.4f}' for r in rewards]}")
                    return rewards
                # Fallback: return baseline rewards if no aligned list is available
                print("WARNING: No aligned rewards found; returning baseline rewards")
                return [0.01 for _ in range(len(completions))]

            # Create GRPO trainer with proper model handling for distributed training
            if self.use_distributed and hasattr(self.model, 'module'):
                # For distributed training, pass the unwrapped model to GRPOTrainer
                # and let the trainer handle its own distributed setup
                trainer_model = self.model.module  # Unwrapped model
                
                # Store reference to DDP model for training
                self._ddp_model = self.model
                
                print("Using unwrapped model for GRPOTrainer initialization")
            else:
                trainer_model = self.model
                self._ddp_model = None

            # Create GRPO trainer with the unwrapped model
            self.grpo_trainer = GRPOTrainer(
                model=trainer_model,
                reward_funcs=[llmsr_reward_function],
                args=self.grpo_config,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
            )
            
            # If we have a DDP model, we need to replace the trainer's model with the DDP version
            # after initialization but before training
            if self._ddp_model is not None:
                print("Replacing trainer model with DDP version for distributed training")
                self.grpo_trainer.model = self._ddp_model
                self.grpo_trainer.model_wrapped = self._ddp_model
            
            print("Starting GRPO training...")
            self.grpo_trainer.train()
            print("GRPO training completed")
            
            # Extract training metrics for wandb logging
            episode_metrics = self._extract_training_metrics()
            
            # CRITICAL: Restore original model state after training
            generation_model.generate = original_generate
            if original_generation_config is not None:
                generation_model.generation_config = original_generation_config
            print("Restored original model.generate method after GRPO training")
            
            # Set model to eval mode
            self.model.eval()
            print("Model set to evaluation mode")
            
            # Save the trained model (use unwrapped model for saving)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(f"./grpo_checkpoints/offline_episode_{self.training_episodes}")
            
            # Log metrics to wandb
            self._log_episode_metrics(episode_metrics)
            
            # Clear offline dataset after training
            self.offline_dataset.clear()
            self.training_episodes += 1
            print(f"Offline training episode {self.training_episodes} completed")
            
        except Exception as e:
            print(f"Error during offline GRPO training: {e}")
            import traceback
            traceback.print_exc()
            
            # Restore original model state even on failure
            if 'generation_model' in locals():
                generation_model.generate = original_generate
            else:
                # Fallback if generation_model wasn't defined
                gen_model = self.get_generation_model() if hasattr(self, 'get_generation_model') else self.model
                if hasattr(gen_model, 'generate'):
                    gen_model.generate = original_generate
            self.model.eval()
            print("Model state restored after training failure")
    

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
            wandb_project: str = "llmsr2l",
            wandb_run_name: str = None,
            use_wandb: bool = True,
            use_distributed: bool = True,  # Add distributed training parameter
    ):
        super().__init__(database, evaluators, samples_per_prompt, config, max_sample_nums, 
                         llm_class, use_distributed)
        
        self.sample_scores = {}  
        self.samples_since_training = 0
        self.mse_history: list[float] = []
        self._reward_history_maxlen: int = 200
        self._reward_min_history: int = 4
        
        # Pass wandb configuration to LLM if it's an OfflineGRPOHuggingFaceLLM
        if isinstance(self._llm, OfflineGRPOHuggingFaceLLM):
            # Update wandb configuration if not already set
            if not self._llm.wandb_initialized:
                self._llm.wandb_project = wandb_project
                self._llm.wandb_run_name = wandb_run_name or f"grpo-{getattr(self._llm, 'model_name', 'model')}-{int(time.time())}"
                self._llm.use_wandb = use_wandb
                if WANDB_AVAILABLE and use_wandb:
                    self._llm._init_wandb()
        else:
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
                print("Triggering offline GRPO training after this iteration...")
                self._llm.train_with_offline_grpo()
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
                self._update_mse_history(mse)
                # reward = self._reward_from_mse(mse) ### Parshin: Scaled with History
                reward = 1 - mse
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
        
        # Finalize wandb logging
        if hasattr(self._llm, '_finalize_wandb'):
            self._llm._finalize_wandb()