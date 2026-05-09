########## 1.5B
#Osc1
mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v1c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_1_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v1c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v1c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v2c
CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_1_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v2c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v2c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v3c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_1_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v3c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v3c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v4c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_1_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v4c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v4c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v5c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_1_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v5c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-1.5B-Instruct_v5c/main.txt 2>&1



########## 0.5B
#Osc1
mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v1c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_0_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v1c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v1c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v2c
CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_0_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v2c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v2c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v3c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_0_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v3c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v3c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v4c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_0_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v4c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v4c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v5c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_atomsr True \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_0_5b" \
               --log_path "./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v5c" \
               > ./logs/oscillator1_llmsr_dec_Qwen2.5-0.5B-Instruct_v5c/main.txt 2>&1