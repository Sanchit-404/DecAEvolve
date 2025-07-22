
import os
from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd

from llmsr import pipeline
from llmsr import config
from llmsr import sampler
from llmsr import evaluator


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)

# GRPO-specific arguments
parser.add_argument('--use_grpo', action='store_true', help='Enable GRPO training')
parser.add_argument('--grpo_model', type=str, default="HuggingFaceTB/SmolLM-135M-Instruct", 
                   help='Model path for GRPO training')
parser.add_argument('--grpo_lr', type=float, default=1e-5, help='GRPO learning rate')
parser.add_argument('--grpo_update_freq', type=int, default=50, help='GRPO update frequency')
parser.add_argument('--grpo_batch_size', type=int, default=4, help='GRPO batch size')
args = parser.parse_args()




if __name__ == '__main__':
    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    
    # Create GRPO config
    grpo_config = config.GRPOConfig(
        model_path=args.grpo_model,
        use_grpo=args.use_grpo,
        learning_rate=args.grpo_lr,
        per_device_train_batch_size=args.grpo_batch_size,
        update_frequency=args.grpo_update_freq,
    )
    
    config = config.Config(
        use_api=args.use_api, 
        api_model=args.api_model,
        grpo_config=grpo_config
    )
    global_max_sample_num = 10000 

    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    
    # Load dataset
    problem_name = args.problem_name
    df = pd.read_csv('./data/'+problem_name+'/train.csv')
    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    if 'torch' in args.spec_path:
        X = torch.Tensor(X)
        y = torch.Tensor(y)
    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict} 
    
    
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        # log_dir = 'logs/m1jobs-mixtral-v10',
        log_dir=args.log_path,
    )
