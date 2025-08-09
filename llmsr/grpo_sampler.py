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

""" GRPO (Group Relative Policy Optimization) sampler classes. """
from __future__ import annotations

import numpy as np
import torch
import time
from typing import Sequence, Type
from .sampler import HuggingFaceLLM, Sampler, LLM
from llmsr import evaluator, buffer, config as config_lib

# GRPO imports
try:
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    GRPO_AVAILABLE = True
except ImportError:
    GRPO_AVAILABLE = False
    print("Warning: GRPO dependencies not available. Install trl, peft, and datasets to use GRPO training.")


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
            
            # Prepare our LLM-SR samples for GRPO monkey-patching
            # Group completions by prompt for easy lookup
            self.precomputed_samples = {}
            self.precomputed_rewards = {}
            
            for prompt, completion, reward in zip(flattened_prompts, flattened_completions, flattened_rewards):
                if prompt not in self.precomputed_samples:
                    self.precomputed_samples[prompt] = []
                    self.precomputed_rewards[prompt] = []
                self.precomputed_samples[prompt].append(completion)
                self.precomputed_rewards[prompt].append(reward)
            
            print(f"GRPO setup: {len(self.precomputed_samples)} unique prompts with LLM-SR samples")
            print(f"Total samples: {len(flattened_prompts)}")
            print(f"Reward range: {min(flattened_rewards):.4f} to {max(flattened_rewards):.4f}")
            
            # Create dataset with prompts only (GRPO expects this format)
            train_dataset = Dataset.from_dict({
                'prompt': list(self.precomputed_samples.keys())
            })
            
            # PRESERVE original model state before GRPO training
            original_generate = self.model.generate
            original_generation_config = self.model.generation_config if hasattr(self.model, 'generation_config') else None
            sample_counter = 0  # Track which samples to return
            
            def patched_generate(input_ids, **kwargs):
                nonlocal sample_counter
                
                # Decode the prompt to find our pre-computed samples
                prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                
                # Clean the prompt text to match our stored prompts
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
                    
                    # GRPO expects multiple completions, so cycle through our samples
                    num_generations = kwargs.get('num_return_sequences', self.grpo_config.num_generations)
                    selected_completions = []
                    
                    for i in range(num_generations):
                        idx = (sample_counter + i) % len(completions)
                        selected_completions.append(completions[idx])
                    
                    sample_counter += num_generations
                    
                    # Tokenize our completions and combine with input
                    batch_outputs = []
                    for completion in selected_completions:
                        # Combine prompt + completion
                        full_text = prompt_text + completion
                        tokenized = self.tokenizer(full_text, return_tensors="pt", 
                                                 max_length=kwargs.get('max_length', 512),
                                                 truncation=True, padding=False)
                        batch_outputs.append(tokenized.input_ids[0])
                    
                    # Stack into batch format and ensure correct device placement
                    import torch
                    max_len = max(len(seq) for seq in batch_outputs)
                    padded_outputs = []
                    
                    # Get the correct device from the model
                    target_device = self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
                    
                    for seq in batch_outputs:
                        if len(seq) < max_len:
                            # Create padding on the correct device
                            padding = torch.full((max_len - len(seq),), self.tokenizer.pad_token_id, 
                                               device=target_device, dtype=seq.dtype)
                            seq = torch.cat([seq.to(target_device), padding])
                        else:
                            seq = seq.to(target_device)
                        padded_outputs.append(seq)
                    
                    # Stack and ensure on correct device
                    result = torch.stack(padded_outputs).to(target_device)
                    print(f"Using LLM-SR samples: {len(selected_completions)} completions for prompt (device: {target_device})")
                    return result
                    
                else:
                    # Fallback to original generation if prompt not found
                    print(f"WARNING: Prompt not found in LLM-SR samples, using original generation")
                    return original_generate(input_ids, **kwargs)
            
            # Apply the monkey patch
            self.model.generate = patched_generate
            print("Monkey-patched model.generate to use LLM-SR samples")
            
            # Reward function that uses our pre-computed rewards
            def reward_function(completions, **kwargs):
                """
                Returns rewards for our LLM-SR samples.
                """
                rewards = []
                print(f"GRPO reward function called with {len(completions)} completions")
                
                # Try to match completions to our pre-computed rewards
                for completion in completions:
                    found_reward = 0.01  # Default minimal reward
                    
                    # Search through our precomputed samples
                    for prompt, stored_completions in self.precomputed_samples.items():
                        for i, stored_completion in enumerate(stored_completions):
                            if completion.strip() in stored_completion or stored_completion in completion.strip():
                                found_reward = self.precomputed_rewards[prompt][i]
                                break
                        if found_reward > 0.01:  # Found a match
                            break
                    
                    rewards.append(found_reward)
                    print(f"Completion reward: {found_reward:.4f}")
                
                print(f"GRPO rewards: {[f'{r:.4f}' for r in rewards]}")
                return rewards
            
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
            
            # CRITICAL: Restore original model state after training
            self.model.generate = original_generate
            if original_generation_config is not None:
                self.model.generation_config = original_generation_config
            print("Restored original model.generate method and generation config after GRPO training")
            
            # COMPREHENSIVE MODEL STATE RESTORATION
            print("=== DEBUGGING MODEL STATE AFTER GRPO ===")
            
            # 1. Set model to eval mode
            self.model.eval()
            print(f"Model training mode: {self.model.training}")
            
            # 2. Check LoRA state
            print(f"Model type: {type(self.model)}")
            if hasattr(self.model, 'peft_config'):
                print(f"PEFT config exists: {self.model.peft_config}")
            
            # 3. Check generation config state  
            if hasattr(self.model, 'generation_config'):
                print(f"Generation config exists: {self.model.generation_config is not None}")
            else:
                print("No generation config attribute found")
                
            # 4. Clear all caches
            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None
            
            # 5. Ensure model is on correct device
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print(f"Model device: {next(self.model.parameters()).device}")
            elif torch.backends.mps.is_available():
                self.model = self.model.to("mps")
                print(f"Model device: {next(self.model.parameters()).device}")
            
            # 6. Test generation capability immediately
            print("Testing model generation capability...")
            test_input = "def test():"
            test_tokens = self.tokenizer(test_input, return_tensors="pt")
            if torch.cuda.is_available():
                test_tokens = {k: v.cuda() for k, v in test_tokens.items()}
            
            with torch.no_grad():
                try:
                    test_output = self.model.generate(
                        **test_tokens,
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=1.0
                    )
                    test_generated = self.tokenizer.decode(test_output[0], skip_special_tokens=True)
                    print(f"Test generation successful: {test_generated[:100]}...")
                except Exception as test_e:
                    print(f"Test generation FAILED: {test_e}")
            
            print("=== END MODEL STATE DEBUG ===")
            print("Model restoration completed")
            
            self.model.save_pretrained("./grpo_checkpoints/latest")
            
        except Exception as e:
            print(f"Error during GRPO training: {e}")
            import traceback
            traceback.print_exc()
            
            # CRITICAL: Restore original generate method even if training fails
            self.model.generate = original_generate
            print("Restored original model.generate method after GRPO training failure")
            
            # Set model back to evaluation mode
            self.model.eval()
            print("Model set to evaluation mode after training failure")


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


