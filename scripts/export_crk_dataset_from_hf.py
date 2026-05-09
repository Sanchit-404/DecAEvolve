#!/usr/bin/env python3
"""Export CRK chemistry splits from nnheui/llm-srbench (lsr_bench_data.hdf5) to data/crk{N}/ CSVs.

Requires: huggingface_hub, h5py, pandas, numpy, and HF_TOKEN for the gated dataset.

Example:
  export HF_TOKEN=$(cat ~/.hf_token)
  python scripts/export_crk_dataset_from_hf.py 22
  python scripts/export_crk_dataset_from_hf.py 12 --out-dir data/crk12
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("crk_num", type=int, help="Benchmark id, e.g. 22 -> CRK22 / data/crk22")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: data/crk{N})",
    )
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    path = hf_hub_download(
        repo_id="nnheui/llm-srbench",
        filename="lsr_bench_data.hdf5",
        repo_type="dataset",
        token=token,
    )
    group = f"lsr_synth/chem_react/CRK{args.crk_num}"
    out = args.out_dir or (args.repo_root / "data" / f"crk{args.crk_num}")
    out.mkdir(parents=True, exist_ok=True)
    cols = ["t", "A", "dA_dt"]

    with h5py.File(path, "r") as f:
        if group not in f:
            raise SystemExit(f"Missing {group} in HDF5 — check lsr_synth/chem_react in the file.")
        g = f[group]
        for split, name in [("train", "train.csv"), ("test", "test_id.csv"), ("ood_test", "test_ood.csv")]:
            arr = np.array(g[split])
            df = pd.DataFrame(arr, columns=cols)
            dest = out / name
            df.to_csv(dest, index=False)
            print(dest, df.shape)


if __name__ == "__main__":
    main()
