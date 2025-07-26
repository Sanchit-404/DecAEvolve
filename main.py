
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
parser.add_argument('--hf_model', type=str, default="Qwen/Qwen3-1.5B-Instruct")
parser.add_argument('--use_grpo', type=bool, default=False)
parser.add_argument('--grpo_batch_size', type=int, default=4)
parser.add_argument('--grpo_learning_rate', type=float, default=2e-5)
args = parser.parse_args()




if __name__ == '__main__':
    # Load config and parameters
    # Choose LLM class based on GRPO flag
    if args.use_grpo:
        llm_class = sampler.GRPOHuggingFaceLLM
        print("Using GRPO-enabled HuggingFace model for training")
    else:
        llm_class = sampler.HuggingFaceLLM
        print("Using standard HuggingFace model")
        
    class_config = config.ClassConfig(llm_class=llm_class, sandbox_class=evaluator.LocalSandbox)
    config = config.Config(use_api = args.use_api, 
                           api_model = args.api_model,
                           hf_model = args.hf_model,
                           use_grpo = args.use_grpo,
                           grpo_batch_size = args.grpo_batch_size,
                           grpo_learning_rate = args.grpo_learning_rate)
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
