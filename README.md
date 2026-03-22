# MolDesign

A reproducible molecular generation/optimization workflow based on VAE latent search + PSO + QSPR predictors.

## What this repository contains

- VAE model for SMILES latent encoding/decoding: `vae.py`
- Property predictors (SeqQSPR + DPCNN): `dpcnn.py`
- PSO generation scripts:
  - Uni-objective chunked runs: `OptUni.py`
  - Multi-objective chunked runs: `OptTri.py`
- Backfill and evaluation for missing/corrupted seeds: `backfill_pso_and_evaluate.py`
- Property/statistics evaluation: `model_stats.py`
- MOSES metrics (Validity/Uniqueness/Novelty/IntDiv...): `mosesBenchmark.py`

## Environment (single env, Python 3.7)

This project is intended to run in **one conda environment** with:

- Python `3.7`
- RDKit `2022.03.5` (conda-forge)
- TensorFlow `2.10.1` (2022 mainstream)
- MOSES benchmarking package

Recommended setup:

```bash
conda env create -f environment.yml
conda activate moldesign37
```

If you prefer pip-only installation:

```bash
python -m pip install -r requirements.txt
```

> Note: on Windows, RDKit is more stable through conda (`environment.yml`).

## Typical workflow

### 1) Train/refresh property models (optional if pretrained already exists)

```bash
python dpcnn.py --properties LOGP TPSA SA --trials 3000
```

### 2) Run PSO generation by chunks (R1-R5 for 500 runs total)

```bash
python OptUni.py --total-runs 500 --chunk-size 100
python OptTri.py --total-runs 500 --chunk-size 100
```

Add `--with-stats` if you want immediate `model_stats` outputs after each chunk.

### 3) Backfill missing/corrupted seed files and evaluate

```bash
python backfill_pso_and_evaluate.py --rounds 5 --chunk-size 100
```

Detection only:

```bash
python backfill_pso_and_evaluate.py --detect-only --rounds 5 --chunk-size 100
```

### 4) Compute MOSES metrics for R1-R5 outputs

```bash
python mosesBenchmark.py --rounds 5
```

## Output locations

- PSO results: `results/pso/<OBJECTIVE...>/<PROP>/<target>-Seed*.npz`
- Chunk evaluation summary: `results/pso/*-R*/evaluation_summary.csv`
- Combined summary: `results/pso/UNI-OBJECTIVE/evaluation_summary.csv`, `results/pso/MULTI-OBJECTIVE/evaluation_summary.csv`
- MOSES round metrics: `results/pso/*-R*/moses_metrics.csv`
- MOSES pooled + mean/std:
  - `results/pso/UNI-OBJECTIVE/moses_metrics_pooled.csv`
  - `results/pso/UNI-OBJECTIVE/moses_metrics_mean_std.csv`
  - `results/pso/MULTI-OBJECTIVE/moses_metrics_pooled.csv`
  - `results/pso/MULTI-OBJECTIVE/moses_metrics_mean_std.csv`

## Reproducibility notes

- Scripts rely on project-root relative paths; they call `set_working_directory()` from `config.py`.
- Default chunk setting in this repo is `5 rounds x 100 seeds`.
- If OpenMP conflict appears in MOSES runs, `mosesBenchmark.py` already sets:
  - `KMP_DUPLICATE_LIB_OK=TRUE`
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`

## Publish to GitHub (skip NPZ and files >25MB)

This repository includes `publish_github.ps1` for one-click sync.

What it does:

- Initializes git repo if needed
- Ensures `origin` matches your repository URL
- Stages changes
- Automatically skips:
  - all `.npz` files
  - any staged file larger than 25MB (configurable)
- Commits only when there are actual changes
- Pushes to `origin/main`

PowerShell examples:

```powershell
# Dry run: commit only, do not push
powershell -ExecutionPolicy Bypass -File .\publish_github.ps1 -NoPush -CommitMessage "chore: local sync"

# Normal one-click upload
powershell -ExecutionPolicy Bypass -File .\publish_github.ps1 -CommitMessage "chore: update code and evaluation csv"
```

Because git tracks content by commit state, rerunning this script does not re-upload unchanged files.
