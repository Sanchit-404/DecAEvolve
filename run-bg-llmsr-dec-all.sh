
########## 3B
#Osc1
mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v1c
CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v1c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v1c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v2c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v2c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v2c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v3c
CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v3c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v3c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v4c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v4c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v4c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v5c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v5c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-3B-Instruct_v5c/main.txt 2>&1



########## 7B
#Osc1
mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v1c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v1c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v1c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v2c
CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v2c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v2c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v3c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v3c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v3c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v4c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v4c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v4c/main.txt 2>&1

mkdir -p ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v5c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v5c" \
               > ./logs/bactgrow_llmsr_dec_Qwen2.5-7B-Instruct_v5c/main.txt 2>&1