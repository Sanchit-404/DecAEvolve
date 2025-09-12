
CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_offline_grpo True \
               --grpo_learning_rate 1e-6 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               # --use_wandb True
            #    --log_path "./logs/oscillator1_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp50" \
            #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
            #    --use_wandb True


# CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
#                --use_offline_grpo True \
#                --grpo_learning_rate 1e-6 \
#                --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
#                --use_wandb True
#             #    --log_path "./logs/oscillator1_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp50" \
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --use_wandb True