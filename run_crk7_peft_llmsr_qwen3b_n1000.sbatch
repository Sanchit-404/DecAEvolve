#!/bin/bash
#SBATCH --job-name=crk7-peft-llmsr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/slurm/crk7-peft-llmsr-qwen3b-n1000-%j.out
#SBATCH --error=./logs/slurm/crk7-peft-llmsr-qwen3b-n1000-%j.err

# Post-GRPO LLM-SR: crk7, nprompt50 GRPO checkpoint-300. Submit from repo root.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p ./logs/slurm

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"

CRK_TAG="${CRK_TAG:-crk7}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
N_PROMPTS="${N_PROMPTS:-50}"
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-1000}"
PEFT_CHECKPOINT="${PEFT_CHECKPOINT:-${SLURM_SUBMIT_DIR}/grpo_checkpoints/crk7-V2-adaptive-Qwen/Qwen2.5-3B-Instruct-r8-ga4-g64-lr1e-06/nprompt50/checkpoint-300}"

if [ ! -d "${PEFT_CHECKPOINT}" ]; then
  echo "[crk7-peft-llmsr] ERROR: PEFT dir missing: ${PEFT_CHECKPOINT}" >&2
  exit 1
fi
if [ ! -f "./data/${CRK_TAG}/train.csv" ]; then
  echo "[crk7-peft-llmsr] ERROR: missing ./data/${CRK_TAG}/train.csv" >&2
  exit 1
fi

if [ -f "/home/grads/sanchit23/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "/home/grads/sanchit23/miniconda3/etc/profile.d/conda.sh"
  conda activate llmsr
fi

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  for _HF in \
    "${HF_TOKEN_FILE:-}" \
    "${SLURM_SUBMIT_DIR:-.}/.hf_token" \
    "${SLURM_SUBMIT_DIR:-.}/hf_token.txt" \
    "${HOME:-}/.hf_token" \
    "${HOME:-}/hf_token.txt"; do
    [ -z "${_HF}" ] && continue
    [ -f "${_HF}" ] || continue
    export HF_TOKEN="$(tr -d ' \t\n\r' < "${_HF}")"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    break
  done
fi

LOG_PATH="./logs/slurm/${CRK_TAG}_peft_llmsr_qwen3b_n${MAX_SAMPLE_NUMS}_job${SLURM_JOB_ID:-manual}"
mkdir -p "${LOG_PATH}"

echo "[crk7-peft-llmsr] CRK_TAG=${CRK_TAG} HF_MODEL=${HF_MODEL} N_PROMPTS=${N_PROMPTS} MAX_SAMPLE_NUMS=${MAX_SAMPLE_NUMS}"
echo "[crk7-peft-llmsr] PEFT_CHECKPOINT=${PEFT_CHECKPOINT}"
echo "[crk7-peft-llmsr] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

python -u external/decaevolve-llama/decaevolve-llama/main.py \
  --spec_path ./specs/specification_crk_chem_numpy.txt \
  --problem_name "${CRK_TAG}" \
  --hf_model "${HF_MODEL}" \
  --n_prompts "${N_PROMPTS}" \
  --max_sample_nums "${MAX_SAMPLE_NUMS}" \
  --use_atomsr True \
  --peft_checkpoint "${PEFT_CHECKPOINT}" \
  --log_path "${LOG_PATH}"

echo "[crk7-peft-llmsr] Done. log_path=${LOG_PATH}"
