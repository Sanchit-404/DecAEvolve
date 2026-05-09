#!/usr/bin/env bash
# Post-GRPO LLM-SR on crk7: Qwen2.5-3B + LoRA checkpoint-300 (nprompt50).
# See run-crk4-llmsr-peft-qwen3b-nprompt10-ckpt300.sh for notes.
set -euo pipefail
REPO=/home/grads/sanchit23/downloaded_decaevolve-1215
cd "$REPO"
PEFT="${REPO}/grpo_checkpoints/crk7-V2-adaptive-Qwen/Qwen2.5-3B-Instruct-r8-ga4-g64-lr1e-06/nprompt50/checkpoint-300"
LOGDIR="${REPO}/logs/crk7_llmsr_peft_qwen3b_nprompt50_ckpt300"
mkdir -p "$LOGDIR"
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u external/decaevolve-llama/decaevolve-llama/main.py \
  --spec_path ./specs/specification_crk_chem_numpy.txt \
  --problem_name crk7 \
  --hf_model "Qwen/Qwen2.5-3B-Instruct" \
  --n_prompts 50 \
  --use_atomsr True \
  --peft_checkpoint "$PEFT" \
  --log_path "$LOGDIR" \
  > "${LOGDIR}/main.stdout" 2> "${LOGDIR}/main.stderr"
