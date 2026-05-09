#!/usr/bin/env bash
# Post-GRPO LLM-SR on crk4: base Qwen/Qwen2.5-3B-Instruct + LoRA from GRPO checkpoint (nprompt10, step 300).
# Uses external main.py with standard HuggingFaceLLM (no offline GRPO loop). Matches CRK atomsr + chem spec.
# Data: ./data/crk4/train.csv only (same as training main). Does not load test_id/test_ood here.
#
# Run from repo root (or sbatch with this script as command). Activate conda env llmsr if needed.
set -euo pipefail
REPO=/home/grads/sanchit23/downloaded_decaevolve-1215
cd "$REPO"
PEFT="${REPO}/grpo_checkpoints/crk4-V2-adaptive-Qwen/Qwen2.5-3B-Instruct-r8-ga4-g64-lr1e-06/nprompt10/checkpoint-300"
LOGDIR="${REPO}/logs/crk4_llmsr_peft_qwen3b_nprompt10_ckpt300"
mkdir -p "$LOGDIR"
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u external/decaevolve-llama/decaevolve-llama/main.py \
  --spec_path ./specs/specification_crk_chem_numpy.txt \
  --problem_name crk4 \
  --hf_model "Qwen/Qwen2.5-3B-Instruct" \
  --n_prompts 10 \
  --use_atomsr True \
  --peft_checkpoint "$PEFT" \
  --log_path "$LOGDIR" \
  > "${LOGDIR}/main.stdout" 2> "${LOGDIR}/main.stderr"
