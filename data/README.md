# Data directory

CRK and other `train.csv` trees are **not committed** (size + licensing). Generate locally:

```bash
export HF_TOKEN=...   # Hugging Face; dataset nnheui/llm-srbench is gated
pip install h5py pandas huggingface_hub
python scripts/export_crk_dataset_from_hf.py 22
```

Place other benchmarks (oscillator, etc.) under `data/<problem_name>/train.csv` as expected by `main.py`.
