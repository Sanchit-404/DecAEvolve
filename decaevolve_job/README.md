# decaevolve_job — runtime tree for Slurm / local GRPO jobs

This directory is a **self-contained** copy of the code path used by
`scripts/run_crk_offline_grpo_atomsr_qwen7b.sh` (offline GRPO + AtomSR + HF models).

Contents:

- `main.py` — CLI entry (paths below)
- `llmsr/` — pipeline, offline GRPO sampler, evaluators, etc.
- `llm_engine/` — auxiliary engine module
- `accelerate_*.yaml`, `run_grpo.sh`, `run_inf.sh` — optional helpers

## Paths

Training reads:

- **Spec:** `--spec_path` — if not absolute, resolved under the **repository root** (parent of `decaevolve_job/`).
- **Data:** `data/<problem_name>/train.csv` under that same repository root.

Override the root with:

```bash
export DECAEVOLVE_REPO_ROOT=/path/to/repo
```

## Example (from repository root)

```bash
conda activate llmsr
export CUDA_VISIBLE_DEVICES=0,1
python -u decaevolve_job/main.py \
  --spec_path specs/specification_crk_chem_numpy.txt \
  --problem_name crk4 \
  --use_offline_grpo True \
  --use_atomsr True \
  --hf_model Qwen/Qwen2.5-7B-Instruct \
  --n_prompts 50 \
  --use_wandb True \
  --log_path ./logs/my_run
```

Use the wrapper script for vLLM + second GPU: `bash scripts/run_crk_offline_grpo_atomsr_qwen7b.sh`.

## What is *not* inside this folder

`decaevolve_job/` holds **Python sources and small configs** only. A real job still needs, from the **repository root** (or `DECAEVOLVE_REPO_ROOT`):

- `data/<problem_name>/train.csv`
- `specs/...` (e.g. `specs/specification_crk_chem_numpy.txt`)

Plus the **environment** (conda `llmsr`, CUDA, `trl vllm-serve`, PyTorch, etc.), **Hugging Face** model weights/cache, optional **W&B** / **HF** tokens, and the **wrapper script** under `scripts/` that starts vLLM and `main.py`. Training and evaluation logic for CRK+AtomSR live under `llmsr/` here; there is **no** `newtonbench` dependency in this tree. Optional `atom_integration` (for `AtomSymbolicAnalyzer`) is not shipped in this copy; AtomSR falls back when the import is missing.

## Slurm check (uses `decaevolve_job/main.py` by default)

From repo root (optional tiny model / early LLM-SR cap — GRPO inner `max_steps` may still be large):

```bash
export DECAEVOLVE_MAIN_PY="$PWD/decaevolve_job/main.py"
export MAX_SAMPLE_NUMS=30 N_PROMPTS=4 HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct SLURM_RUN_TAG=smoke
CRK_TAG=crk4 sbatch compare_run_crk_offline_grpo_external_atomsr_qwen7b.sbatch
```
