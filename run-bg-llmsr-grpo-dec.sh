############## 3B
mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v1c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v1c" \
                --vllm_model_name "bact_3b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v1c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v2c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v2c" \
                --vllm_model_name "bact_3b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v2c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v3c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v3c" \
                --vllm_model_name "bact_3b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v3c/main.txt 2>&1


mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v4c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v4c" \
                --vllm_model_name "bact_3b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v4c/main.txt 2>&1


mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v5c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v5c" \
                --vllm_model_name "bact_3b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-3B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v5c/main.txt 2>&1


###### 7B
mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v1c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v1c" \
                --vllm_model_name "bact_7b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v1c/main.txt 2>&1 


mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v2c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v2c" \
                --vllm_model_name "bact_7b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v2c/main.txt 2>&1


mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v3c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v3c" \
                --vllm_model_name "bact_7b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v3c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v4c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v4c" \
               --vllm_model_name "bact_7b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v4c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v5c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --log_path "./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v5c" \
                --vllm_model_name "bact_7b" \
               > ./logs/bactgrow_llmsr_grpo_dec_Qwen2.5-7B-Instruct_nprompt2000_r8_ga4_ng64_lr1e-05_ckp350_v5c/main.txt 2>&1
