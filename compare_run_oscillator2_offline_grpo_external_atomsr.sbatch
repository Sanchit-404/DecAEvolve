#!/bin/bash
#SBATCH --job-name=cmp-osc2-grpo-at
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=./logs/slurm/compare-osc2-offline-grpo-ext-atomsr-%j.out
#SBATCH --error=./logs/slurm/compare-osc2-offline-grpo-ext-atomsr-%j.err

# Same as compare_run_oscillator2_offline_grpo_external.sbatch but --use_atomsr True (evaluator2 sandbox).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p ./logs/slurm

export PYTHONUNBUFFERED=1
# vLLM HTTP: VLLM_SERVER_PORT (set below). TRL NCCL group port: derived from SLURM_JOB_ID in Python, or set VLLM_GROUP_PORT_BASE.

if [ -f "/home/grads/sanchit23/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "/home/grads/sanchit23/miniconda3/etc/profile.d/conda.sh"
  conda activate llmsr
fi

# W&B: use existing env, or first file found (trim whitespace). Set WANDB_KEY_FILE to a custom path.
if [ -z "${WANDB_API_KEY:-}" ]; then
  for _WKEY in \
    "${WANDB_KEY_FILE:-}" \
    "${SLURM_SUBMIT_DIR:-.}/.wandb_api_key" \
    "${SLURM_SUBMIT_DIR:-.}/wandb_api_key.txt" \
    "${HOME:-}/.wandb_api_key" \
    "${HOME:-}/wandb_api_key.txt"; do
    [ -z "${_WKEY}" ] && continue
    [ -f "${_WKEY}" ] || continue
    export WANDB_API_KEY="$(tr -d ' \t\n\r' < "${_WKEY}")"
    break
  done
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[compare-ext-at] WARN: WANDB_API_KEY unset — GRPO runs without W&B (add repo .wandb_api_key or export WANDB_KEY_FILE)." >&2
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
  echo "[compare-ext-at] ERROR: need 2 GPUs (got: ${GPU_LIST})" >&2
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

echo "[compare-ext-at] CUDA_VISIBLE_DEVICES=${GPU_LIST}  vLLM_GPU=${VLLM_GPU}  TRAIN_GPU=${TRAIN_GPU}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
FREE_MB="$(nvidia-smi -i "${VLLM_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1)"
TOTAL_MB="$(nvidia-smi -i "${VLLM_GPU}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1)"
CAP_UTIL="$(awk -v f="${FREE_MB:-0}" -v t="${TOTAL_MB:-1}" 'BEGIN { if (t+0 <= 0) { print 0.45; exit } ; c = 0.92 * (f+0)/(t+0); if (c < 0.08) c = 0.08; if (c > 0.90) c = 0.90; printf "%.3f", c }')"
REQ_UTIL="${VLLM_GPU_MEMORY_UTILIZATION}"
FINAL_UTIL="$(awk -v a="${REQ_UTIL}" -v b="${CAP_UTIL}" 'BEGIN { if (a+0 < b+0) print a; else print b }')"
echo "[compare-ext-at] GPU ${VLLM_GPU} mem: ${FREE_MB:-?}/${TOTAL_MB:-?} MiB free -> vLLM gpu-memory-utilization ${FINAL_UTIL}"

echo "[compare-ext-at] Starting vLLM on GPU ${VLLM_GPU}..."

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
echo "[compare-ext-at] VLLM_PORT=${VLLM_PORT}  VLLM_LOCAL_ENDPOINT=${VLLM_LOCAL_ENDPOINT}"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" trl vllm-serve \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --tensor_parallel_size 1 \
  --gpu-memory-utilization "${FINAL_UTIL}" \
  --max-model-len 3072 \
  --port "${VLLM_PORT}" \
  > "./logs/slurm/compare-osc2-offline-grpo-ext-atomsr-vllm-${SLURM_JOB_ID:-0}.log" 2>&1 &
VLLM_PID=$!

echo "[compare-ext-at] Waiting for http://127.0.0.1:${VLLM_PORT}/health/ ..."
while true; do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[compare-ext-at] ERROR: vLLM exited before ready." >&2
    exit 1
  fi
  if curl -4 -fsS "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
    echo "[compare-ext-at] vLLM ready."
    break
  fi
  sleep 2
done

echo "[compare-ext-at] Starting external main.py (use_atomsr True) ..."

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python -u external/decaevolve-llama/decaevolve-llama/main.py \
  --spec_path ./specs/specification_oscillator2_numpy.txt \
  --problem_name oscillator2 \
  --use_offline_grpo True \
  --use_atomsr True \
  --grpo_learning_rate 1e-6 \
  --hf_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --n_prompts "${N_PROMPTS:-50}" \
  --use_wandb True \
  --log_path "./logs/compare_osc2_offline_grpo_ext_atomsr_${SLURM_JOB_ID:-manual}"

echo "[compare-ext-at] Done."
