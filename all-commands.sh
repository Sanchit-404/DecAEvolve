#Osc1
mkdir -p ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v1c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v1c" \
               > ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v1c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v2c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v2c" \
               > ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v2c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v3c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v3c" \
               > ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v3c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v4c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v4c" \
               > ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v4c/main.txt 2>&1

mkdir -p ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v5c
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v5c" \
               > ./logs/oscillator1_llmsr_Qwen2.5-3B-Instruct_v5c/main.txt 2>&1


#Osc2
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v1b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v1b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v2b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v2b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v3b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v3b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v4b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v4b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v5b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-3B-Instruct_v5b/main.txt 2>&1


#BactGrow
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v1b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v1b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v2b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v2b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v3b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v3b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v4b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v4b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v5b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-3B-Instruct_v5b/main.txt 2>&1


#StressStrain
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v1b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v1b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v2b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v2b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v3b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v3b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v4b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v4b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_3b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v5b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-3B-Instruct_v5b/main.txt 2>&1



##################### Inference Pure LLMSR
########## 7B
#Osc1
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v1b" \
               > ./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v1b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v2b" \
               > ./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v2b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v3b" \
               > ./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v3b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v4b" \
               > ./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v4b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v5b" \
               > ./logs/oscillator1_llmsr_Qwen2.5-7B-Instruct_v5b/main.txt 2>&1


#Osc2
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v1b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v1b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v2b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v2b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v3b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v3b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v4b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v4b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator2_numpy.txt \
               --problem_name oscillator2 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v5b" \
               > ./logs/oscillator2_llmsr_Qwen2.5-7B-Instruct_v5b/main.txt 2>&1


#BactGrow
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v1b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v1b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v2b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v2b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v3b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v3b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v4b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v4b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_bactgrow_numpy.txt \
               --problem_name bactgrow \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v5b" \
               > ./logs/bactgrow_llmsr_Qwen2.5-7B-Instruct_v5b/main.txt 2>&1


#StressStrain
CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v1b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v1b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v2b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v2b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v3b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v3b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v4b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v4b/main.txt 2>&1

CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_stressstrain_numpy.txt \
               --problem_name stressstrain \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \
               --vllm_model_name "qwen_7b" \
               --log_path "./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v5b" \
               > ./logs/stressstrain_llmsr_Qwen2.5-7B-Instruct_v5b/main.txt 2>&1

# --- agent process log (2026-04-02) ---
# scancel 1551 (root compare_run_oscillator2_offline_grpo.sbatch); queue clear.
# Fixed external/decaevolve-llama/decaevolve-llama/llmsr/offline_grpo_sampler.py:
#   f-string SyntaxErrors (output_dir, run_name), vLLM from VLLM_SERVER_PORT + 127.0.0.1.
# New Slurm: compare_run_oscillator2_offline_grpo_external.sbatch -> external/.../main.py
# sbatch compare_run_oscillator2_offline_grpo_external.sbatch  -> job 1552
# Logs: logs/slurm/compare-osc2-offline-grpo-ext-{1552}.out|.err, vllm log ext-vllm-1552.log
#
# 2026-04-03: cp llmsr/evaluator2.py -> external/.../llmsr/ (external had none; use_atomsr needs it).
# compare_run_oscillator2_offline_grpo_external_atomsr.sbatch (--use_atomsr True)
# sbatch compare_run_oscillator2_offline_grpo_external_atomsr.sbatch -> job 1553
# Logs: compare-osc2-offline-grpo-ext-atomsr-1553.out|.err, ext-atomsr-vllm-1553.log
#
# 2026-04-03: external offline_grpo_sampler.py — replace c1_aiml_aem wandb with import wandb.
# compare *.sbatch — WANDB_API_KEY from SLURM_SUBMIT_DIR/.wandb_api_key or ~/.wandb_api_key.
# sbatch compare_run_oscillator2_offline_grpo_external_atomsr.sbatch -> job 1554
#
# 2026-04-03: W&B UsageError fix — _grpo_report_to() -> [] if WANDB_API_KEY empty (ext + root offline_grpo_sampler).
# sbatch: WANDB_KEY_FILE + extra key file paths; warn if unset.
# sbatch compare_run_oscillator2_offline_grpo_external_atomsr.sbatch -> job 1555
# .wandb_api_key in repo root (gitignored); sbatch compare_run_oscillator2_offline_grpo_external_atomsr -> job 1556
# 1556 NCCL error vs 1555: same vllm_group_port=51216 on shared node -> fixed with SLURM_JOB_ID-based port base.
# Audit: root+ext offline_grpo_sampler — vllm 127.0.0.1, group port base+clamp, VLLM_GROUP_PORT_BASE override.
# data/crk4, crk7, crk10, crk12 from HF nnheui/llm-srbench lsr_bench_data.hdf5 chem_react; CSV cols t,A,dA_dt (last=y for main.py).
# 2026-04-05: Refreshed data/crk4 and added data/crk15, data/crk28 from same HDF5 (lsr_synth/chem_react/CRK{4,15,28} -> train.csv, test_id.csv, test_ood.csv).
# compare_run_oscillator2_offline_grpo_external_atomsr_qwen3b.sbatch — oscillator2 + Qwen2.5-3B-Instruct (override HF_MODEL=... if needed)
# sampler.py / ext sampler — LLMSR_COMPLETIONS_HOST/PORT; kazem — VLLM_SERVER_HOST/PORT.
# sbatch compare_run_oscillator2_offline_grpo_external_atomsr -> job 1557
#
# 2026-04-03: external offline_grpo_sampler train_with_offline_grpo — align with root PEFT lifecycle:
#   peft_config=None when model already PeftModel (avoid stacked LoRA each episode);
#   disable_adapters with ValueError "No adapter loaded" catch; delete stale "ref" adapter if beta!=0;
#   enable_adapters() in finally so local model.generate() after GRPO uses trained LoRA (vLLM is separate sync path).
# sbatch compare_run_oscillator2_offline_grpo_external_atomsr_qwen3b.sbatch -> job 1559 (Qwen2.5-3B-Instruct + PEFT fix)
#
# ========== REVERT CHEATSHEET (external decaevolve-llama) ==========
# Use `git diff <file>` or `git checkout -- <file>` from a known-good commit if you committed before these edits.
#
# 2026-04-03: external offline_grpo — call model.enable_adapters() after GRPOTrainer(...) and before .train()
#   so LoRA is active during loss (disable_adapters() before trainer left adapters off for episode 2+).
#
# (A) Offline GRPO / PEFT lifecycle — file:
#     external/decaevolve-llama/decaevolve-llama/llmsr/offline_grpo_sampler.py
#     Undo: Remove _unwrap_to_peft_model, _model_already_has_peft_training; drop PeftModel + is_peft_model import block.
#     In train_with_offline_grpo: replace the disable_adapters try/except + train_peft_cfg + ref-adapter delete with:
#       if hasattr(self.model, "peft_config"):
#           self.model.disable_adapters()
#     GRPOTrainer(..., peft_config=lora_cfg)  # always lora_cfg, not train_peft_cfg
#     In finally: remove enable_adapters() block after model.eval().
#
# (B) use_atomsr → full evaluator2.Evaluator — file:
#     external/decaevolve-llama/decaevolve-llama/llmsr/pipeline.py
#     Undo: Delete the `if config.use_atomsr:` / _evaluator_cls branch; in the evaluators loop use only:
#       evaluators.append(evaluator.Evaluator(..., sandbox_class=class_config.sandbox_class))
#     (main.py still switches sandbox to evaluator2.LocalSandbox when use_atomsr; that was already true before (B).)
#
# 2026-04-03: pipeline.main — when config.use_atomsr, use evaluator2.Evaluator + main.py’s evaluator2.LocalSandbox
#             (term/atom hooks in evaluator2; AtomSymbolicAnalyzer only if atom_integration import works).
# sbatch compare_run_oscillator2_offline_grpo_external_atomsr_qwen3b.sbatch -> job 1560 (evaluator2.Evaluator + PEFT fix)
#
# 2026-04-03: vllm_importance_sampling_correction default OFF (env GRPO_VLLM_IS_CORRECTION=1 to enable TRL default behavior).
#   ext + root offline_grpo_sampler _setup and episode cfg. Root: enable_adapters() after GRPOTrainer before train (match ext).
# sbatch compare_run_oscillator2_offline_grpo_external_atomsr_qwen3b.sbatch -> job 1561 (IS off + adapter-on-before-train + root sync)
# ext offline_grpo: LLMSR_GRPO_REWARD_SIGMA (default 1) — exp(-mse/sigma); sigma>1 raises logged rewards for same MSE (collection + GRPO reward_fn).
#
# CRK chem + Qwen2.5-7B + atomsr offline GRPO: compare_run_crk_offline_grpo_external_atomsr_qwen7b.sbatch
#   specs/specification_crk_chem_numpy.txt; data/crk4, crk7, crk10/train.csv (cols t,A,dA_dt).
#   CRK_TAG=crk4|crk7|crk10  N_PROMPTS=50 (default in script). Override: N_PROMPTS=64 CRK_TAG=crk7 sbatch ...
# compare_run_oscillator2_offline_grpo_external_atomsr_qwen3b.sbatch — --n_prompts ${N_PROMPTS:-50}
# sbatch CRK 7B: CRK_TAG=crk4 -> 1562, crk7 -> 1563, crk10 -> 1564
#
# 2026-04-04: Reverted TCPStore patch + launch_trl_vllm_serve.py + python -c launcher; back to plain `trl vllm-serve` (1561-era vLLM start).
# scancel 1567 1568; stagger CRK 7B: CRK_TAG=crk4 -> 1569, CRK_TAG=crk7 -> 1570 with sbatch --dependency=afterok:1569
# 2026-04-04: scancel 1564 1569 1570 (CRK cmp-crk-7b); spec L-BFGS-B + PARAM_BOUND + robust loss (specification_crk_chem_numpy.txt).
# Resubmit stagger: CRK_TAG=crk4 -> 1571, CRK_TAG=crk7 -> 1572 (afterok:1571). Left cmp-osc2-3b 1561 running.
# 2026-04-04: scancel 1571 1572; compare_run_crk_offline_grpo_external_atomsr_qwen7b.sbatch exports LLMSR_GRPO_REWARD_SIGMA=0.001 (override at submit time).
# Resubmit stagger: CRK_TAG=crk4 -> 1573, CRK_TAG=crk7 -> 1574 (afterok:1573).
# scancel 1574; CRK_TAG=crk7 sbatch (no dependency) -> 1575 — run parallel with 1573.
# 2026-04-04: data/crk17 from HF nnheui/llm-srbench lsr_bench_data.hdf5 lsr_synth/chem_react/CRK17 -> train.csv, test_id.csv, test_ood.csv
# ext offline_grpo_sampler: LLMSR_DATALOADER_NUM_WORKERS (default 4); CRK sbatch VLLM_GPU_MEMORY_UTILIZATION default 0.5, exports workers env.
# CRK_TAG=crk17 sbatch compare_run_crk_offline_grpo_external_atomsr_qwen7b.sbatch -> 1576
# CRK sbatch: SLURM_RUN_TAG (default 7b) -> log_path atomsr_${SLURM_RUN_TAG} and vLLM log filename.
# Meta-Llama-3.1-8B-Instruct + SLURM_RUN_TAG=llama31-8b: crk4 -> 1577; scancel 1578 1579 (had afterany deps).
# crk7/crk17 resubmit no dependency -> 1580, 1581. Slurm logs: compare-crk-llama31-8b-{crk4,crk7,crk17}-%j.out/.err
# 1580: CANCELLED + vLLM port 8003 in use on shared node; user reverted SLURM_JOB_ID-based port scan (original 8000.. scan).
# Retry Llama 3.1 8B (no port fix): CRK_TAG=crk4|crk7|crk17 -> 1597, 1598, 1599 (no Slurm dependency).
# Qwen2.5-3B-Instruct CRK: HF_MODEL=Qwen/Qwen2.5-3B-Instruct SLURM_RUN_TAG=qwen25-3b -> crk4:1600 crk7:1601 crk17:1602
# Llama 1597 “hang” vs Qwen 1600: external+root llmsr/sampler.py loaded HF weights twice (from_pretrained x2); fixed single load + device_map auto.
# Llama 1603 “hang” after Set evaluators: HF generate(batch, n=50 seqs) on 8B — mitigated by chunked batch generate (LLMSR_MAX_PARALLEL_GENERATIONS, default 8) in external llmsr/sampler.py _do_request.
# trl vllm-serve does not serve OpenAI POST /v1/chat/completions (404) — do NOT auto-route sampling on VLLM_SERVER_PORT. Opt-in only: LLMSR_SAMPLE_VIA_OPENAI_VLLM=1 + LLMSR_COMPLETIONS_PORT (or VLLM_SERVER_PORT) for a plain vLLM OpenAI server.
# scancel 1597 (llama crk4); resubmit Meta-Llama-3.1-8B crk4|crk7|crk17 -> 1603, 1604, 1605 (no Slurm deps).
# scancel 1603; vLLM chat sampling attempt -> 1606–1608; superseded by chunked HF + no auto vLLM chat (see above).
# 2026-04-02: scancel 1603 (already gone); Llama 3.1 8B crk4 + chunked HF sampler -> job 1609 (logs: compare-crk-llama31-8b-crk4-%j.out/.err).
# Job 1607 (.err): ValueError tokenizer padding with no pad_token on Llama 3.1 8B — external llmsr/sampler.py: if "llama" in model_name.lower() and pad_token is None, set pad_token = eos_token.
# 2026-04-05: Llama 3.1 8B crk7 resubmit after Llama-only pad_token fix -> job 1610 (logs: compare-crk-llama31-8b-crk7-%j.out/.err).
# external offline_grpo_sampler: GRPO num_warmup_steps 200 -> 50 (warmup_stable_decay).
# Llama 3.1 8B + warmup 50: CRK_TAG=crk4|crk15|crk28 -> jobs 1611, 1612, 1613 (logs compare-crk-llama31-8b-{tag}-%j.out/.err).
# scancel 1611 1612 1613; resubmit Llama crk4|crk15|crk28 -> 1614, 1615, 1616. Put rotated HF token in ~/.hf_token (chmod 600) so sbatch loads HF_TOKEN (gated Hub / PEFT).
# scancel 1614 1615 1616; resubmit Llama crk4|crk15|crk28 -> 1617, 1618, 1619 (restart with ~/.hf_token).
# Stagger: Llama crk4 only -> 1620 (run crk15/crk28 later with same pattern, different CRK_TAG).
# CRK sbatch: optional HF_TOKEN / HUGGING_FACE_HUB_TOKEN, or HF_TOKEN_FILE / ~/.hf_token / repo .hf_token (gitignored) for gated Llama Hub config/PEFT.
# Slurm policy (user): use batch jobs (sbatch) only — do not use srun (including --overlap) for diagnostics or ad-hoc node commands.
# 2026-04-05: kill stray VLLM on frieren (gpu-audit PIDs, not 1610) -> job 1624; stagger Llama crk4/crk15/crk28 afterok:1624 + 600s gaps -> job 1625 (logs: kill-stray-vllm-*.out, stagger-llama-crk-*.out).
# Qwen2.5-3B CRK: crk7|crk10|crk15 -> 1629,1630,1631 (logs compare-crk-qwen25-3b-${tag}-%j). Qwen2.5-7B same CRKs -> 1632,1633,1634 (compare-crk-qwen25-7b-${tag}-%j).
# Post-GRPO LLM-SR (no offline GRPO loop): external main.py --peft_checkpoint <checkpoint-dir> + Qwen/Qwen2.5-3B-Instruct; env LLMSR_PEFT_CHECKPOINT also works.
# ./run-crk4-llmsr-peft-qwen3b-nprompt10-ckpt300.sh | ./run-crk7-llmsr-peft-qwen3b-nprompt50-ckpt300.sh | ./run-crk10-llmsr-peft-qwen3b-nprompt50-ckpt250.sh
# crk4 PEFT smoke (1000 samples): external main.py --max_sample_nums 1000 --peft_checkpoint .../nprompt10/checkpoint-300; WANDB_MODE=disabled optional
# If nohup shows "Model loaded successfully on cpu" then RuntimeError CUDA: inputs must follow model device — llmsr/sampler.py + external llmsr/sampler.py use next(model.parameters()).device for inputs (not hardcoded cuda).
# GPU PEFT LLM-SR n=1000: sbatch run_crk4_peft_llmsr_qwen3b_n1000.sbatch | run_crk7_peft_llmsr_qwen3b_n1000.sbatch | run_crk10_peft_llmsr_qwen3b_n1000.sbatch -> logs/slurm/<tag>-peft-llmsr-qwen3b-n1000-%j.out/.err
# Offline GRPO Qwen2.5-7B: CRK_TAG=crk4 sbatch compare_run_crk_offline_grpo_external_atomsr_qwen7b.sbatch -> job 1638 example; logs/slurm/compare-crk-offline-grpo-ext-atomsr-7b-%j.out/.err, log_path logs/compare_crk4_offline_grpo_ext_atomsr_7b_<JOBID>
# PEFT eval (no GRPO): sbatch run_crk7_peft_llmsr_qwen7b_ckpt225_n1000.sbatch — Qwen2.5-7B + checkpoint-225 on crk7, max_sample_nums 1000, logs/slurm/crk7-peft-llmsr-qwen7b-ck225-n1000-%j.out
# 2026-05-07: PEFT eval submissions for missing CRKs (Qwen2.5-7B, no GRPO loop), using checkpoint-500 from each finished training:
#   CRK_TAG=crk3  PEFT_CHECKPOINT=grpo_checkpoints/crk3-V2-adaptive-Qwen/.../nprompt50/checkpoint-500  sbatch run_crk7_peft_llmsr_qwen7b_ckpt225_n1000.sbatch -> 2398
#   CRK_TAG=crk19 PEFT_CHECKPOINT=grpo_checkpoints/crk19-V2-adaptive-Qwen/.../nprompt50/checkpoint-500 sbatch run_crk7_peft_llmsr_qwen7b_ckpt225_n1000.sbatch -> 2399
#   CRK_TAG=crk20 PEFT_CHECKPOINT=grpo_checkpoints/crk20-V2-adaptive-Qwen/.../nprompt50/checkpoint-500 sbatch run_crk7_peft_llmsr_qwen7b_ckpt225_n1000.sbatch -> 2400