################ LLMSR WITHOUT GRPO ########################


################### LLMSR+GRPO+Dec 
##### CLP 300
CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v1b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v1b/main.txt 2>&1 



CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v2b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v2b/main.txt 2>&1 



CUDA_VISIBLE_DEVICES=4 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v3b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v3b/main.txt 2>&1 



CUDA_VISIBLE_DEVICES=5 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v4b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v4b/main.txt 2>&1 



CUDA_VISIBLE_DEVICES=6 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v5b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v5b/main.txt 2>&1 





########################## LLMSE+GRPO 
###### CKP 300
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v1" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v1/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v1b" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v1b/main.txt 2>&1 




CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v2" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v2/main.txt 2>&1 



CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v3" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v3/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=3 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v4b" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v4b/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-300" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v5" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp300_v5/main.txt 2>&1 


########### CKP150
CUDA_VISIBLE_DEVICES=3 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-150" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v1" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v1/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=3 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-150" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v2" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v2/main.txt 2>&1 



CUDA_VISIBLE_DEVICES=4 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-150" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v3" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v3/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=4 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-150" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v4" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v4/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=5 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-150" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v5" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp150_v5/main.txt 2>&1 




########### CKP100
CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-100" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp100_v1" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp100_v1/main.txt 2>&1 

CUDA_VISIBLE_DEVICES=3 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-100" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp100_v2" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp100_v2/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=4 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/llm-sr2l/grpo_checkpoints/bactgrow-V2-adaptive-Llama-3.2-1B-Instruct-r8-ga4-g64-lr1e-05/nprompt2000/checkpoint-100" \
               --log_path "./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp100_v3" \
               > ./logs/bactgrow_llmsr_grpo_Llama-3.2-1B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp100_v3/main.txt 2>&1 




############## Pure LLMSR ##########
CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v1" \
               > ./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v1/main.txt 2>&1 
               


CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v2" \
               > ./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v2/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v3" \
               > ./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v3/main.txt 2>&1 
               


CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v4" \
               > ./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v4/main.txt 2>&1 



CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v5" \
               > ./logs/bactgrow_llmsr_Llama-3.2-1B-Instruct_v5/main.txt 2>&1 




############ LLMSR+Dec 
CUDA_VISIBLE_DEVICES=3 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v1b" \
               > ./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v1b/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=3 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v2b" \
               > ./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v2b/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=3 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v3b" \
               > ./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v3b/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=3 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v4b" \
               > ./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v4b/main.txt 2>&1 


CUDA_VISIBLE_DEVICES=3 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "/home/jovyan/nly934-storage/models/Llama-3.2-1B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v5b" \
               > ./logs/bactgrow_llmsr_dec_Llama-3.2-1B-Instruct_v5b/main.txt 2>&1





################ LLMSR ########################

# CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
#                --problem_name oscillator1 \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/oscillator1-Qwen2.5-0.5B-Instruct-r8-g64-1755898769/checkpoint-50" \
#                --log_path "./logs/oscillator1_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp50" \
#                > ./logs/oscillator1_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp50/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True

# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
#                --problem_name oscillator1 \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/oscillator1-Qwen2.5-1.5B-Instruct-r8-g64-1756158958/checkpoint-100" \
#                --log_path "./logs/oscillator1_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100" \
#                > ./logs/oscillator1_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
#                --problem_name oscillator2 \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/oscillator2-Qwen2.5-0.5B-Instruct-r8-g64-1756174705/checkpoint-75" \
#                --log_path "./logs/oscillator2_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp75" \
#                > ./logs/oscillator2_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp75/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
#                --problem_name oscillator2 \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/oscillator2-Qwen2.5-1.5B-Instruct-r8-g64-1756174399/checkpoint-120" \
#                --log_path "./logs/oscillator2_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp120" \
#                > ./logs/oscillator2_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp120/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
#                --problem_name bactgrow \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/bactgrow-Qwen2.5-0.5B-Instruct-r8-g64-1756181910/checkpoint-100" \
#                --log_path "./logs/bactgrow_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp100" \
#                > ./logs/bactgrow_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp100/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
#                --problem_name bactgrow \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/bactgrow-Qwen2.5-1.5B-Instruct-r8-g64-1756174399/checkpoint-100" \
#                --log_path "./logs/bactgrow_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100" \
#                > ./logs/bactgrow_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
#                --problem_name stressstrain \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/stressstrain-Qwen2.5-0.5B-Instruct-r8-g64-1756185192/checkpoint-125" \
#                --log_path "./logs/stressstrain_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp125" \
#                > ./logs/stressstrain_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp125/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
#                --problem_name stressstrain \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/stressstrain-Qwen2.5-1.5B-Instruct-r8-g64-1756184995/checkpoint-120" \
#                --log_path "./logs/stressstrain_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp120" \
#                > ./logs/stressstrain_llmsr_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp120/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True



######################## LLMSR + Decomposition ########################

# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
#                --problem_name oscillator1 \
#                --use_atomsr True \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/oscillator1-Qwen2.5-0.5B-Instruct-r8-g64-1755898769/checkpoint-50" \
#                --log_path "./logs/oscillator1_llmsr_dec_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp50_v2" \
#                > ./logs/oscillator1_llmsr_dec_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp50_v2/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True

# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
#                --problem_name oscillator1 \
#                --use_atomsr True \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/oscillator1-Qwen2.5-1.5B-Instruct-r8-g64-1756158958/checkpoint-100" \
#                --log_path "./logs/oscillator1_llmsr_dec_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100" \
#                > ./logs/oscillator1_llmsr_dec_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
#                --problem_name oscillator2 \
#                --use_atomsr True \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/oscillator2-Qwen2.5-0.5B-Instruct-r8-g64-1756174705/checkpoint-75" \
#                --log_path "./logs/oscillator2_llmsr_dec_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp75_v2" \
#                > ./logs/oscillator2_llmsr_dec_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp75_v2/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
#                --problem_name oscillator2 \
#                --use_atomsr True \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/oscillator2-Qwen2.5-1.5B-Instruct-r8-g64-1756174399/checkpoint-120" \
#                --log_path "./logs/oscillator2_llmsr_dec_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp120" \
#                > ./logs/oscillator2_llmsr_dec_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp120/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
#                --problem_name bactgrow \
#                --use_atomsr True \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/bactgrow-Qwen2.5-0.5B-Instruct-r8-g64-1756181910/checkpoint-100" \
#                --log_path "./logs/bactgrow_llmsr_dec_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp100" \
#                > ./logs/bactgrow_llmsr_dec_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp100/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
#                --problem_name bactgrow \
#                --use_atomsr True \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/bactgrow-Qwen2.5-1.5B-Instruct-r8-g64-1756174399/checkpoint-100" \
#                --log_path "./logs/bactgrow_llmsr_dec_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100" \
#                > ./logs/bactgrow_llmsr_dec_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp100/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
#                --problem_name stressstrain \
#                --use_atomsr True \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/stressstrain-Qwen2.5-0.5B-Instruct-r8-g64-1756185192/checkpoint-125" \
#                --log_path "./logs/stressstrain_llmsr_dec_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp125_v2" \
#                > ./logs/stressstrain_llmsr_dec_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp125_v2/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True


# CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
#                --problem_name stressstrain \
#                --use_atomsr True \
#                --hf_model "/home/grads/parshinshojaee/llm-sr2l/grpo_checkpoints/stressstrain-Qwen2.5-1.5B-Instruct-r8-g64-1756184995/checkpoint-120" \
#                --log_path "./logs/stressstrain_llmsr_dec_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp120" \
#                > ./logs/stressstrain_llmsr_dec_grpo_Qwen2.5-1.5B-Instruct_r8_g64_ckp120/main.txt 2>&1 
#             #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
#             #    --log_path "./logs/oscillator1_v16"
#             #    --use_wandb True



########################################################
################ LLMSR WITH GRPO ########################

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --use_offline_grpo True \
               --grpo_learning_rate 1e-6 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               # --log_path "./logs/oscillator1_llmsr_adaptive_grpo_Qwen2.5-0.5B-Instruct_r16_g64_ckp50" \
               # --log_path "./logs/oscillator1_llmsr_grpo_Qwen2.5-0.5B-Instruct_r8_g64_ckp50" \
            #    --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
            #    --use_wandb True


# --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \


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
