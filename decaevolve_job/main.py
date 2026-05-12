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
parser.add_argument('--api_model', type=str, default="gpt-4o-mini")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator2")
parser.add_argument('--problem_name', type=str, default="oscillator2")
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--n_prompts', type=int, default=10)
parser.add_argument(
    '--max_sample_nums',
    type=int,
    default=10000,
    help='Stop after this many evaluated LLM samples (Sampler._global_samples_nums).',
)
parser.add_argument('--hf_model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument('--grpo_learning_rate', type=float, default=1e-6)
parser.add_argument('--use_offline_grpo', type=bool, default=False)
parser.add_argument('--use_atomsr', type=bool, default=False)
parser.add_argument('--use_wandb', type=bool, default=True)
parser.add_argument(
    '--peft_checkpoint',
    type=str,
    default='',
    help='Directory with PEFT adapter (e.g. grpo_checkpoints/.../checkpoint-300). '
    'Loads on top of --hf_model. Omit for base-model-only LLM-SR.',
)
args = parser.parse_args()
if __name__ == '__main__':
    # Repository root: parent of this folder (decaevolve_job/). Override with DECAEVOLVE_REPO_ROOT.
    _job_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = (os.environ.get("DECAEVOLVE_REPO_ROOT") or "").strip()
    if not _repo_root:
        _repo_root = os.path.abspath(os.path.join(_job_dir, ".."))

    # Load config and parameters
    _pc = str(getattr(args, "peft_checkpoint", "") or "").strip()
    # Offline GRPO applies its own LoRA path; do not stack a frozen PEFT dir here.
    if args.use_offline_grpo:
        sampler.HuggingFaceLLM.peft_checkpoint = None
    else:
        sampler.HuggingFaceLLM.peft_checkpoint = _pc if _pc else None

    # Choose LLM class based on GRPO flags
    sandbox_mod = evaluator
    if args.use_atomsr:
        from llmsr import evaluator2

        sandbox_mod = evaluator2

    if args.use_offline_grpo:
        from llmsr.offline_grpo_sampler import OfflineGRPOHuggingFaceLLM
        llm_class = OfflineGRPOHuggingFaceLLM
        print("Using Offline GRPO-enabled HuggingFace model for training")
    else:
        llm_class = sampler.HuggingFaceLLM
        print("Using standard HuggingFace model")

    llm_class.problem_name = args.problem_name
    llm_class.n_prompts = args.n_prompts
    class_config = config.ClassConfig(llm_class=llm_class, sandbox_class=sandbox_mod.LocalSandbox)
    config = config.Config(use_api = args.use_api, 
                           api_model = args.api_model,
                           hf_model = args.hf_model,
                           grpo_learning_rate = args.grpo_learning_rate,
                           use_offline_grpo = args.use_offline_grpo,
                           use_atomsr = args.use_atomsr,
                           n_prompts = args.n_prompts)
    global_max_sample_num = args.max_sample_nums
    # Load prompt specification (paths relative to repo root unless absolute)
    _spec_path = args.spec_path
    if not os.path.isabs(_spec_path):
        _spec_path = os.path.join(_repo_root, _spec_path)
    with open(_spec_path, encoding="utf-8") as f:
        specification = f.read()

    # Load dataset
    problem_name = args.problem_name
    _train_csv = os.path.join(_repo_root, "data", problem_name, "train.csv")
    df = pd.read_csv(_train_csv)
    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    if 'torch' in _spec_path:
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
        use_wandb=args.use_wandb,
    )