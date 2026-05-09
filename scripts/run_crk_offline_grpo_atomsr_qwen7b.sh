#!/usr/bin/env bash
# CRK + offline GRPO + AtomSR + Qwen2.5-7B (or override HF_MODEL). Needs 2 GPUs.
# Works on bare metal, interactive clusters, or Slurm (when submitted via compare_run_crk_offline_grpo_external_atomsr_qwen7b.sbatch).
#
# Usage (from repo root):
#   CRK_TAG=crk4 bash scripts/run_crk_offline_grpo_atomsr_qwen7b.sh
#   CUDA_VISIBLE_DEVICES=0,1 CRK_TAG=crk12 HF_MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/run_crk_offline_grpo_atomsr_qwen7b.sh
#
# Optional env:
#   RUN_ID=...           Log suffix when not under Slurm (default: timestamp).
#   SLURM_RUN_TAG=7b     W&B / checkpoint tag (default 7b).
#   N_PROMPTS=50
#   LLMSR_SKIP_CONDA=1   Do not try to source conda or activate llmsr.
#   CONDA_ROOT=...       Path containing etc/profile.d/conda.sh (default: search HOME).
#   CONDA_ENV_NAME=llmsr
#   VLLM_PORT=8001       Fixed port; otherwise picks a free port in 8000–8099.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
CRK_TAG="${CRK_TAG:-crk4}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SLURM_RUN_TAG="${SLURM_RUN_TAG:-7b}"
N_PROMPTS="${N_PROMPTS:-50}"
export LLMSR_GRPO_REWARD_SIGMA="${LLMSR_GRPO_REWARD_SIGMA:-0.001}"
export LLMSR_DATALOADER_NUM_WORKERS="${LLMSR_DATALOADER_NUM_WORKERS:-4}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  JOB_LOG_ID="${SLURM_JOB_ID}"
  mkdir -p "${REPO_ROOT}/logs/slurm"
  VLLM_LOG_DIR="${REPO_ROOT}/logs/slurm"
else
  JOB_LOG_ID="${RUN_ID}"
  VLLM_LOG_DIR="${REPO_ROOT}/logs/local/${RUN_ID}"
  mkdir -p "${VLLM_LOG_DIR}"
fi

echo "[crk-7b] LLMSR_GRPO_REWARD_SIGMA=${LLMSR_GRPO_REWARD_SIGMA}"
echo "[crk-7b] LLMSR_DATALOADER_NUM_WORKERS=${LLMSR_DATALOADER_NUM_WORKERS}"

if [ ! -f "${REPO_ROOT}/external/decaevolve-llama/decaevolve-llama/main.py" ]; then
  echo "[crk-7b] ERROR: missing external/decaevolve-llama/decaevolve-llama/main.py — see README (external snapshot)." >&2
  exit 1
fi

if [ ! -f "./data/${CRK_TAG}/train.csv" ]; then
  echo "[crk-7b] ERROR: missing ./data/${CRK_TAG}/train.csv" >&2
  exit 1
fi

if [ -z "${LLMSR_SKIP_CONDA:-}" ]; then
  _conda_sh=""
  for _c in \
    "${CONDA_ROOT:+${CONDA_ROOT}/etc/profile.d/conda.sh}" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    [ -z "${_c}" ] && continue
    [ -f "${_c}" ] || continue
    _conda_sh="${_c}"
    break
  done
  if [ -n "${_conda_sh}" ]; then
    # shellcheck source=/dev/null
    source "${_conda_sh}"
    conda activate "${CONDA_ENV_NAME:-llmsr}" 2>/dev/null || true
  fi
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  for _WKEY in \
    "${WANDB_KEY_FILE:-}" \
    "${REPO_ROOT}/.wandb_api_key" \
    "${REPO_ROOT}/wandb_api_key.txt" \
    "${HOME:-}/.wandb_api_key" \
    "${HOME:-}/wandb_api_key.txt"; do
    [ -z "${_WKEY}" ] && continue
    [ -f "${_WKEY}" ] || continue
    export WANDB_API_KEY="$(tr -d ' \t\n\r' < "${_WKEY}")"
    break
  done
fi
[ -z "${WANDB_API_KEY:-}" ] && echo "[crk-7b] WARN: WANDB_API_KEY unset — GRPO may skip W&B." >&2

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  for _HF in \
    "${HF_TOKEN_FILE:-}" \
    "${REPO_ROOT}/.hf_token" \
    "${REPO_ROOT}/hf_token.txt" \
    "${HOME:-}/.hf_token" \
    "${HOME:-}/hf_token.txt"; do
    [ -z "${_HF}" ] && continue
    [ -f "${_HF}" ] || continue
    export HF_TOKEN="$(tr -d ' \t\n\r' < "${_HF}")"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    break
  done
fi
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "[crk-7b] WARN: HF_TOKEN unset — gated Hub models may warn or fail." >&2
else
  echo "[crk-7b] Hugging Face token loaded for Hub (if present)."
fi

VLLM_PID=""
kill_vllm_tree() {
  local pid="$1"
  [ -z "${pid}" ] && return 0
  local c
  for c in $(pgrep -P "${pid}" 2>/dev/null || true); do
    kill_vllm_tree "${c}"
  done
  kill -TERM "${pid}" >/dev/null 2>&1 || true
}

cleanup() {
  if [ -n "${VLLM_PID}" ]; then
    kill_vllm_tree "${VLLM_PID}"
    sleep 1
    kill -KILL "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra GPU_ARR <<< "${GPU_LIST}"
if [ "${#GPU_ARR[@]}" -lt 2 ]; then
  echo "[crk-7b] ERROR: need 2 GPUs in CUDA_VISIBLE_DEVICES (got: ${GPU_LIST})" >&2
  exit 1
fi

VLLM_GPU=""
BEST_FREE=-1
for g in "${GPU_ARR[@]}"; do
  fm="$(nvidia-smi -i "${g}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1)"
  if [ -n "${fm}" ] && [ "${fm}" -gt "${BEST_FREE}" ] 2>/dev/null; then
    BEST_FREE="${fm}"
    VLLM_GPU="${g}"
  fi
done
[ -z "${VLLM_GPU}" ] && VLLM_GPU="${GPU_ARR[0]}"
TRAIN_GPU=""
for g in "${GPU_ARR[@]}"; do
  if [ "${g}" != "${VLLM_GPU}" ]; then
    TRAIN_GPU="${g}"
    break
  fi
done
[ -z "${TRAIN_GPU}" ] && TRAIN_GPU="${GPU_ARR[1]}"

echo "[crk-7b] CRK_TAG=${CRK_TAG}  MODEL=${HF_MODEL}  RUN_TAG=${SLURM_RUN_TAG}  N_PROMPTS=${N_PROMPTS}"
echo "[crk-7b] CUDA_VISIBLE_DEVICES=${GPU_LIST}  vLLM_GPU=${VLLM_GPU}  TRAIN_GPU=${TRAIN_GPU}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
FREE_MB="$(nvidia-smi -i "${VLLM_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1)"
TOTAL_MB="$(nvidia-smi -i "${VLLM_GPU}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1)"
CAP_UTIL="$(awk -v f="${FREE_MB:-0}" -v t="${TOTAL_MB:-1}" 'BEGIN { if (t+0 <= 0) { print 0.5; exit } ; c = 0.90 * (f+0)/(t+0); if (c < 0.08) c = 0.08; if (c > 0.88) c = 0.88; printf "%.3f", c }')"
REQ_UTIL="${VLLM_GPU_MEMORY_UTILIZATION}"
FINAL_UTIL="$(awk -v a="${REQ_UTIL}" -v b="${CAP_UTIL}" 'BEGIN { if (a+0 < b+0) print a; else print b }')"
echo "[crk-7b] GPU ${VLLM_GPU} mem: ${FREE_MB:-?}/${TOTAL_MB:-?} MiB free -> vLLM gpu-memory-utilization ${FINAL_UTIL}"

if [ -z "${VLLM_PORT:-}" ]; then
  VLLM_PORT="$(python3 -c 'import socket
for port in range(8000, 8100):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        continue
    else:
        s.close()
        print(port)
        break
else:
    raise SystemExit("no free port in 8000-8099")
')"
fi
export VLLM_SERVER_PORT="${VLLM_PORT}"
export VLLM_LOCAL_ENDPOINT="http://127.0.0.1:${VLLM_PORT}"
echo "[crk-7b] VLLM_PORT=${VLLM_PORT}"

VLLM_LOG="${VLLM_LOG_DIR}/compare-crk-offline-grpo-ext-${SLURM_RUN_TAG}-vllm-${CRK_TAG}-${JOB_LOG_ID}.log"
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" trl vllm-serve \
  --model "${HF_MODEL}" \
  --tensor_parallel_size 1 \
  --gpu-memory-utilization "${FINAL_UTIL}" \
  --max-model-len 3072 \
  --port "${VLLM_PORT}" \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "[crk-7b] vLLM logs -> ${VLLM_LOG}"
echo "[crk-7b] Waiting for http://127.0.0.1:${VLLM_PORT}/health/ ..."
while true; do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[crk-7b] ERROR: vLLM exited before ready." >&2
    exit 1
  fi
  if curl -4 -fsS "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
    echo "[crk-7b] vLLM ready."
    break
  fi
  sleep 2
done

echo "[crk-7b] Starting external main.py (${CRK_TAG}, atomsr, ${HF_MODEL}, n_prompts=${N_PROMPTS}) ..."

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python -u external/decaevolve-llama/decaevolve-llama/main.py \
  --spec_path ./specs/specification_crk_chem_numpy.txt \
  --problem_name "${CRK_TAG}" \
  --use_offline_grpo True \
  --use_atomsr True \
  --grpo_learning_rate 1e-6 \
  --hf_model "${HF_MODEL}" \
  --n_prompts "${N_PROMPTS}" \
  --use_wandb True \
  --log_path "./logs/compare_${CRK_TAG}_offline_grpo_ext_atomsr_${SLURM_RUN_TAG}_${JOB_LOG_ID}"

echo "[crk-7b] Done."
