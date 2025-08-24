

CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --use_offline_grpo True \
               --grpo_learning_rate 1e-6 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/oscillator1_v16" \
               --use_wandb True


# python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
#                --use_offline_grpo True \
#                --grpo_learning_rate 2e-5 \
#                --hf_model "Qwen/Qwen3-0.6B" \
#                --log_path "./logs/oscillator1_v16"


# export CUDA_LAUNCH_BLOCKING=1
# python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
#                --use_offline_grpo True \
#                --grpo_learning_rate 1e-6 \
#                --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
#                --log_path "./logs/oscillator1_grpo_Qwen2.5-.5B-Instruct_v1"

