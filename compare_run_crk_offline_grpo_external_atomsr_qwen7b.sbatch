#!/bin/bash
#SBATCH --job-name=cmp-crk-7b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --output=./logs/slurm/compare-crk-offline-grpo-ext-atomsr-7b-%j.out
#SBATCH --error=./logs/slurm/compare-crk-offline-grpo-ext-atomsr-7b-%j.err

# CRK (crk4 / crk7 / crk10 / crk12 / crk17 / …) + offline GRPO + atomsr + Qwen2.5-7B.
# Submit:  CRK_TAG=crk4 sbatch compare_run_crk_offline_grpo_external_atomsr_qwen7b.sbatch
# Non-Slurm: CRK_TAG=crk4 bash scripts/run_crk_offline_grpo_atomsr_qwen7b.sh
# Reward: exp(-mse/sigma). Default sigma=0.001 in script; override LLMSR_GRPO_REWARD_SIGMA=...
# Requires: ./data/${CRK_TAG}/train.csv  and  ./specs/specification_crk_chem_numpy.txt
# Training logic: scripts/run_crk_offline_grpo_atomsr_qwen7b.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p ./logs/slurm
export SLURM_JOB_ID="${SLURM_JOB_ID:-0}"
exec bash "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/run_crk_offline_grpo_atomsr_qwen7b.sh"
