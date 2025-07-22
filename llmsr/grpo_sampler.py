"""
GRPO-enhanced sampler for LLM-SR equation discovery.
Integrates Group Relative Policy Optimization with symbolic regression.
"""

import torch
import numpy as np
import time
from typing import Collection, Dict, List, Any
from abc import ABC, abstractmethod
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from llmsr import evaluator, buffer, config as config_lib
from llmsr.sampler import LLM, Sampler


class GRPOEquationLLM(LLM):
    """
    GRPO-enhanced LLM for equation discovery that continuously learns
    from equation evaluation feedback.
    """
    
    def __init__(
        self, 
        model_path: str,
        samples_per_prompt: int,
        grpo_config: Dict[str, Any] = None,
        use_lora: bool = True,
        update_frequency: int = 50
    ):
        super().__init__(samples_per_prompt)
        
        self.model_path = model_path
        self.update_frequency = update_frequency
        self.training_buffer = []
        self.iteration_count = 0
        self.grpo_config = grpo_config or self._default_grpo_config()
        
        # Load model and tokenizer
        self._load_model_and_tokenizer(use_lora)
        
        # Initialize GRPO trainer
        self._initialize_grpo_trainer()
        
        logging.info(f"Initialized GRPO LLM with model: {model_path}")
    
    def _default_grpo_config(self) -> Dict[str, Any]:
        """Default GRPO configuration for equation discovery"""
        return {
            "learning_rate": 1e-5,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_prompt_length": 512,
            "max_completion_length": 256,
            "num_generations": 8,
            "optim": "adamw_8bit",
            "num_train_epochs": 1,
            "bf16": True,
            "remove_unused_columns": False,
            "logging_steps": 1,
            "kl_coeff": 0.1,
        }
    
    def _load_model_and_tokenizer(self, use_lora: bool = True):
        """Load the base model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Apply LoRA for efficient fine-tuning
        if use_lora:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                target_modules="all-linear",
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, lora_config)
            print(f"LoRA parameters: {self.model.print_trainable_parameters()}")
        
        # Create generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    
    def _initialize_grpo_trainer(self):
        """Initialize the GRPO trainer"""
        # Create training args
        training_args = GRPOConfig(
            output_dir="./grpo_equation_model",
            **self.grpo_config
        )
        
        # Initialize trainer (will be updated with actual dataset during training)
        self.grpo_trainer = None
        self.training_args = training_args
    
    def equation_reward_function(self, completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        """
        Reward function specifically designed for equation discovery.
        Combines multiple factors: correctness, parsimony, novelty.
        """
        rewards = []
        
        for completion, prompt in zip(completions, prompts):
            reward = 0.0
            
            try:
                # Extract equation from completion
                equation_code = self._extract_equation_body(completion)
                
                # Syntax reward: Can the equation be parsed/executed?
                syntax_reward = self._evaluate_syntax(equation_code)
                reward += syntax_reward
                
                # Parsimony reward: Prefer simpler equations
                complexity_penalty = self._compute_complexity_penalty(equation_code)
                reward -= complexity_penalty
                
                # Physical plausibility reward
                physics_reward = self._evaluate_physics_plausibility(equation_code)
                reward += physics_reward
                
                # Store for evaluation-based reward (will be updated later)
                evaluation_reward = kwargs.get('evaluation_scores', {}).get(completion, 0.0)
                reward += evaluation_reward
                
            except Exception as e:
                logging.warning(f"Error computing reward for equation: {e}")
                reward = -5.0  # Heavy penalty for invalid equations
            
            rewards.append(reward)
        
        return rewards
    
    def _extract_equation_body(self, completion: str) -> str:
        """Extract the equation body from the LLM completion"""
        lines = completion.strip().split('\n')
        equation_body = []
        in_function = False
        
        for line in lines:
            if 'def equation(' in line:
                in_function = True
                continue
            elif in_function:
                if line.strip() and not line.startswith('def ') and 'return' in line:
                    equation_body.append(line.strip())
                elif line.strip() == '' or line.startswith('def '):
                    break
                else:
                    equation_body.append(line.strip())
        
        return '\n'.join(equation_body) if equation_body else completion
    
    def _evaluate_syntax(self, equation_code: str) -> float:
        """Evaluate if the equation has valid Python syntax"""
        try:
            compile(equation_code, '<string>', 'exec')
            return 2.0  # Positive reward for valid syntax
        except SyntaxError:
            return -2.0  # Penalty for syntax errors
    
    def _compute_complexity_penalty(self, equation_code: str) -> float:
        """Compute complexity penalty based on equation structure"""
        # Count operators and function calls
        operators = ['+', '-', '*', '/', '**', 'np.', 'sin', 'cos', 'exp', 'log']
        complexity = sum(equation_code.count(op) for op in operators)
        return min(complexity * 0.1, 2.0)  # Cap penalty at 2.0
    
    def _evaluate_physics_plausibility(self, equation_code: str) -> float:
        """Basic physics plausibility check"""
        # Reward equations that use input variables appropriately
        reward = 0.0
        
        # Check if equation uses input variables
        if any(var in equation_code for var in ['x', 'v', 'params']):
            reward += 1.0
        
        # Bonus for physics-relevant functions
        physics_functions = ['sin', 'cos', 'exp', 'sqrt', 'abs']
        if any(func in equation_code for func in physics_functions):
            reward += 0.5
        
        return reward
    
    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """
        Generate equation samples and perform GRPO training if needed
        """
        # Generate samples
        samples = self._generate_samples(prompt)
        
        # Store for potential GRPO training
        self._store_training_data(prompt, samples)
        
        # Perform GRPO training if we have enough data
        if len(self.training_buffer) >= self.update_frequency:
            self._perform_grpo_training()
        
        self.iteration_count += 1
        return samples
    
    def _generate_samples(self, prompt: str) -> List[str]:
        """Generate equation samples using the current model"""
        samples = []
        
        # Add instruction for equation generation
        full_prompt = self._format_prompt(prompt)
        
        for _ in range(self._samples_per_prompt):
            try:
                # Generate with some randomness for diversity
                output = self.generator(
                    full_prompt,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=np.random.uniform(0.7, 1.0),
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                generated_text = output[0]['generated_text'].strip()
                samples.append(generated_text)
                
            except Exception as e:
                logging.warning(f"Generation error: {e}")
                # Fallback: simple equation
                samples.append("    return params[0] * x + params[1]")
        
        return samples
    
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for equation generation"""
        instruction = (
            "You are a helpful assistant tasked with discovering mathematical function structures "
            "for scientific systems. Complete the 'equation' function below, considering the "
            "physical meaning and relationships of inputs.\n\n"
        )
        return instruction + prompt
    
    def _store_training_data(self, prompt: str, samples: List[str]):
        """Store prompt-sample pairs for GRPO training"""
        for sample in samples:
            self.training_buffer.append({
                'prompt': prompt,
                'completion': sample,
                'timestamp': time.time()
            })
    
    def _perform_grpo_training(self):
        """Perform GRPO training on collected data"""
        if not self.training_buffer:
            return
        
        logging.info(f"Performing GRPO training on {len(self.training_buffer)} samples")
        
        try:
            # Create dataset from training buffer
            dataset_dict = {
                'prompt': [item['prompt'] for item in self.training_buffer],
                'completion': [item['completion'] for item in self.training_buffer]
            }
            training_dataset = Dataset.from_dict(dataset_dict)
            
            # Initialize GRPO trainer if not done
            if self.grpo_trainer is None:
                self.grpo_trainer = GRPOTrainer(
                    model=self.model,
                    reward_funcs=[self.equation_reward_function],
                    args=self.training_args,
                    train_dataset=training_dataset,
                )
            else:
                # Update dataset
                self.grpo_trainer.train_dataset = training_dataset
            
            # Perform training
            self.grpo_trainer.train()
            
            # Clear training buffer
            self.training_buffer = []
            
            logging.info("GRPO training completed successfully")
            
        except Exception as e:
            logging.error(f"GRPO training failed: {e}")
            # Clear buffer to prevent repeated failures
            self.training_buffer = []
    
    def update_rewards_with_evaluation(self, evaluation_results: Dict[str, float]):
        """Update stored training data with evaluation-based rewards"""
        # This can be called by the evaluator to provide feedback
        for item in self.training_buffer:
            completion = item['completion']
            if completion in evaluation_results:
                item['evaluation_reward'] = evaluation_results[completion]


class GRPOSampler(Sampler):
    """
    Enhanced sampler that uses GRPO for continuous learning
    """
    
    def __init__(
        self,
        database: buffer.ExperienceBuffer,
        evaluators,
        samples_per_prompt: int,
        config: config_lib.Config,
        model_path: str = "microsoft/DialoGPT-small",
        max_sample_nums: int = None,
        grpo_config: Dict[str, Any] = None,
    ):
        # Initialize GRPO LLM
        grpo_llm = GRPOEquationLLM(
            model_path=model_path,
            samples_per_prompt=samples_per_prompt,
            grpo_config=grpo_config
        )
        
        # Initialize parent class
        super().__init__(
            database=database,
            evaluators=evaluators,
            samples_per_prompt=samples_per_prompt,
            config=config,
            max_sample_nums=max_sample_nums,
            llm_class=type(grpo_llm)  # This is a bit hacky, but works
        )
        
        # Replace the LLM with our GRPO version
        self._llm = grpo_llm
    
    def sample(self, **kwargs):
        """Enhanced sampling with GRPO feedback integration"""
        evaluation_scores = {}
        
        while True:
            # Stop if max samples reached
            if self._max_sample_nums and self._get_global_sample_nums() >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt
            
            # Evaluate samples and collect rewards
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                
                chosen_evaluator = np.random.choice(self._evaluators)
                
                # Store evaluation start time
                eval_start_time = time.time()
                
                # Perform evaluation
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )
                
                # Extract evaluation score for GRPO feedback
                # This is a simplified way to get the score - in practice you might
                # need to modify the evaluator to return the score directly
                evaluation_scores[sample] = self._extract_evaluation_score(chosen_evaluator)
        
        # Update GRPO LLM with evaluation feedback
        if evaluation_scores:
            self._llm.update_rewards_with_evaluation(evaluation_scores)
    
    def _extract_evaluation_score(self, evaluator) -> float:
        """Extract the evaluation score from the evaluator"""
        # This is a placeholder - you might need to modify the evaluator
        # to expose the evaluation score directly
        try:
            # Attempt to get the last evaluation score
            # This assumes the evaluator stores the last result
            return getattr(evaluator, '_last_score', 0.0)
        except:
            return 0.0 