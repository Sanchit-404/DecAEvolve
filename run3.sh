######## LLMSR WITHOUT GRPO ########

# CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
#                --problem_name oscillator1 \
#                --hf_model "Qwen/Qwen2.5-1.5B-Instruct" \
#                --log_path "./logs/oscillator1_llmsr_Qwen2.5-1.5B-Instruct_v2" \
#                > ./logs/oscillator1_llmsr_Qwen2.5-1.5B-Instruct_v2/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True



# CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
#                --problem_name oscillator2 \
#                --hf_model "Qwen/Qwen2.5-1.5B-Instruct" \
#                --log_path "./logs/oscillator2_llmsr_Qwen2.5-1.5B-Instruct" \
#                > ./logs/oscillator2_llmsr_Qwen2.5-1.5B-Instruct/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True



# CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
#                --problem_name bactgrow \
#                --hf_model "Qwen/Qwen2.5-1.5B-Instruct" \
#                --log_path "./logs/bactgrow_llmsr_Qwen2.5-1.5B-Instruct" \
#                > ./logs/bactgrow_llmsr_Qwen2.5-1.5B-Instruct/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True



# CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
#                --problem_name stressstrain \
#                --hf_model "Qwen/Qwen2.5-1.5B-Instruct" \
#                --log_path "./logs/stressstrain_llmsr_Qwen2.5-1.5B-Instruct_v2" \
#                > ./logs/stressstrain_llmsr_Qwen2.5-1.5B-Instruct_v2/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


################ LLMSR + Decomposition ########################

# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
#                --problem_name oscillator1 \
#                --use_atomsr True \
#                --hf_model "Qwen/Qwen2.5-1.5B-Instruct" \
#                --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct" \
#                > ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
#                --problem_name bactgrow \
#                --use_atomsr True \
#                --hf_model "Qwen/Qwen2.5-1.5B-Instruct" \
#                --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-1.5B-Instruct" \
#                > ./logs/bactgrow_llmsr_dec_Qwen2.5-1.5B-Instruct/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True



# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
#                --problem_name oscillator2 \
#                --use_atomsr True \
#                --hf_model "Qwen/Qwen2.5-1.5B-Instruct" \
#                --log_path "./logs/oscillator2_llmsr_dec_Qwen2.5-1.5B-Instruct" \
#                > ./logs/oscillator2_llmsr_dec_Qwen2.5-1.5B-Instruct/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-1.5B-Instruct" \
               --log_path "./logs/stressstrain_llmsr_dec_Qwen2.5-1.5B-Instruct" \
               > ./logs/stressstrain_llmsr_dec_Qwen2.5-1.5B-Instruct/main.txt 2>&1 
            #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
            #    --log_path "./logs/oscillator1_v16"
            #    --use_wandb True