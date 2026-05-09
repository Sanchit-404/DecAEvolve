# decaevolve / LLM-SR (GRPO + AtomSR)

Offline GRPO training and symbolic regression experiments (CRK chemistry, oscillators, etc.). This repo expects a **second tree** under `external/decaevolve-llama/` (patched `main.py` and `llmsr`).

## GitHub / sharing checklist

| Topic | Status |
|--------|--------|
| **Secrets** | `.hf_token`, `.wandb_api_key` are gitignored — never commit them. |
| **External code** | `external/decaevolve-llama/` is gitignored by default. Ship a zip or document how to unpack it (see below). Without it, Slurm/local CRK scripts **will not run**. |
| **Large artifacts** | `logs/`, `wandb/`, `grpo_checkpoints/`, `data/` are not all ignored — **review** before `git add` (datasets can be large). Prefer documenting HF export instead of committing every CSV. |
| **Lockfile** | `requirements-lock.txt` pins the last known-good `pip` set (Linux, CUDA 12, `llmsr` conda env). Regenerate with `bash scripts/refresh_requirements_lock.sh` after upgrading packages. |
| **Platform** | Training is aimed at **Linux + NVIDIA GPU** (2 GPUs for CRK 7B + vLLM). macOS/CPU-only is not supported for the full GRPO+vLLM path. |

## Environment (frozen)

```bash
conda env create -f environment.yml
conda activate llmsr
pip install -r requirements-lock.txt
```

For a looser install (may drift): `pip install -r requirements.txt` then install `vllm` / `trl` per your CUDA stack.

## External `decaevolve-llama` snapshot

1. Obtain the patched snapshot (e.g. `decaevolve-llama.zip` from your team, or your own export).
2. Unpack so that this file exists:

   `external/decaevolve-llama/decaevolve-llama/main.py`

3. Do not commit the unpacked tree if you keep it gitignored; contributors repeat step 1–2.

## CRK datasets (`data/crk{N}/`)

CSV layout: `train.csv`, `test_id.csv`, `test_ood.csv` with columns `t,A,dA_dt`.

Export from Hugging Face (dataset is **gated** — need access + `HF_TOKEN`):

```bash
export HF_TOKEN=...   # or use ~/.hf_token
pip install h5py pandas huggingface_hub
python scripts/export_crk_dataset_from_hf.py 22
```

## Run CRK offline GRPO + AtomSR + Qwen2.5-7B **without Slurm**

Requires **2 GPUs** (vLLM on one, training on the other). From repo root:

```bash
conda activate llmsr
export CUDA_VISIBLE_DEVICES=0,1
CRK_TAG=crk4 bash scripts/run_crk_offline_grpo_atomsr_qwen7b.sh
```

Optional: `HF_MODEL=...`, `N_PROMPTS=...`, `LLMSR_SKIP_CONDA=1` if your shell already has the env active.

Logs: `logs/local/<timestamp>/` for vLLM; training `log_path` under `logs/compare_*`.

## Slurm

```bash
CRK_TAG=crk4 sbatch compare_run_crk_offline_grpo_external_atomsr_qwen7b.sbatch
```

## Legacy quick start (root `main.py`)

```bash
pip install -r requirements.txt
# pip install vllm  # match your CUDA
CUDA_VISIBLE_DEVICES=0 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
  --problem_name oscillator1 --use_offline_grpo True --grpo_learning_rate 1e-6 \
  --hf_model "Qwen/Qwen2.5-0.5B-Instruct"
```
