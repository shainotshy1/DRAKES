# Inference & Evaluation Pipeline

End-to-end workflow for running sequence generation, merging multi-worker outputs, and scoring results.

```
batch_gen_inference.sh
        │
        ▼  (sbatch × NUM_WORKERS)
gen_inference_finetune.sh  →  gen_inference_finetune.py
        │
        ▼  CSVs: <run>_w0.csv, <run>_w1.csv, …
auto_merge_results.py
        │
        ▼  merged: <run>.csv
sbatch eval_all.sh  →  log_likelihood / ddg / scrmsd
```

---

## 1. `scripts/gen_inference_finetune.sh`

SLURM job script that runs a **single worker** of sequence generation via `gen_inference_finetune.py`.

### What it does

- Activates the `mf2` env
- Executes `gen_inference_finetune.py`
- Writes result CSVs under `OUTPUT_FOLDER`

### Worker sharding

Proteins are partitioned across workers:

| Env var        | Role                                      |
|----------------|-------------------------------------------|
| `WORKER_ID`    | 0-indexed shard index (default `0`)       |
| `NUM_WORKERS`  | Total shards (default `1`)                |

When `NUM_WORKERS > 1`, outputs are named with a `_w{WORKER_ID}` suffix, e.g.:

```
pretrained_train_ddg_bon_N=1_w3.csv
```

### Key knobs (edit in the script)

| Variable              | Meaning |
|-----------------------|---------|
| `BASE_PATH`           | Data + model root |
| `MODEL`               | `pretrained` or `drakes` |
| `DATASET`             | `train` / `validation` / `test` / `single` |
| `ALIGN_TYPE`          | `bon` or `beam` |
| `ALIGN_N`             | Samples for best-of-N / beam |
| `ORACLE_MODE`         | Reward oracle: `ddg`, `protgpt`, or `scrmsd` |
| `FEEDBACK_METHOD`     | `spectral` \| `lasso` \| `max-mask` \| `exclusion` \| `hill-climb` \| `gradient` |
| `SPEC_FEEDBACK_ITS`   | Feedback iterations (`0` = off) |
| `BATCH_REPEAT`        | Repeats per protein batch |
| `OUTPUT_FOLDER`       | Where CSVs are written |

### Run one worker locally / via SLURM

```bash
# Single worker (defaults WORKER_ID=0, NUM_WORKERS=1)
sbatch scripts/gen_inference_finetune.sh
```
---

## 2. `scripts/batch_gen_inference.sh`

Launcher that submits **many** `gen_inference_finetune.sh` jobs with consistent `WORKER_ID` / `NUM_WORKERS`.

### Settings

| Variable       | Default | Meaning |
|----------------|---------|---------|
| `NUM_WORKERS`  | `20`    | How many shards to create |
| `NUM_COMPUTE`  | `-1`    | Parallelism grouping (see below) |

### Modes

**Independent jobs** (`NUM_COMPUTE=-1`, default)  
Submits `NUM_WORKERS` independent `sbatch` jobs that can all run at once (subject to cluster limits).

**Grouped / chained** (`NUM_COMPUTE=N > 0`)  
Assigns workers to `N` groups with SLURM `--dependency=afterany:...` so at most `N` jobs run concurrently (one chain per group). Useful when GPU quota is limited.

### Usage

```bash
cd /u/sdickman/DRAKES/drakes_protein/fmif
bash scripts/batch_gen_inference.sh
```

Edit `NUM_WORKERS` / `NUM_COMPUTE` at the top of the script before submitting. Generation settings still come from `gen_inference_finetune.sh`.

---

## 3. `post_processing_scripts/auto_merge_results.py`

Merges per-worker CSV shards into one file per experiment.

### Naming convention

Shards look like:

```
<preamble>_w0.csv
<preamble>_w1.csv
…
```

The merger strips a trailing `_w*` token and concatenates all members of a group into:

```
<preamble>.csv
```

Groups with only one file are left unchanged. An existing merged file whose name equals the group name is skipped when collecting rows (so re-running does not double-count it as a shard).

### Usage

```bash
python post_processing_scripts/auto_merge_results.py /path/to/eval_results/spex_logs/proteins/...
```

Run this **after** all workers finish and **before** (or alongside) evaluation if you want metrics on the combined CSV. `eval_all.sh` will also score individual `_w*` files if they are still present.

---

## 4. `post_processing_scripts/eval_all.sh`

SLURM job script that runs the standard metric suite on every CSV in a directory (or a single CSV path). Submit with `sbatch` (the script has `#SBATCH` headers for GPU allocation).

### Metrics (in order)

1. **Log-likelihood** — `log_likelihood.py` (ProtGPT2; column `loglikelihood`)
2. **ΔΔG** — `ddg.py` (stability oracle)
3. **scRMSD** — `scrmsd.py` (structure consistency)

Each metric script skips files that already have the target column, so re-runs are safe.

### Usage

```bash
sbatch post_processing_scripts/eval_all.sh /path/to/results_dir [GPU]
```

`DIR` can be a folder of CSVs or a single `.csv` file. Default GPU is `0`.

---

## Typical workflow

```bash
# 1. Configure generation knobs in scripts/gen_inference_finetune.sh
#    (MODEL, DATASET, OUTPUT_FOLDER, FEEDBACK_METHOD, …)

# 2. Launch sharded inference
bash scripts/batch_gen_inference.sh

# 3. Wait for SLURM jobs to finish, then merge shards
python post_processing_scripts/auto_merge_results.py "$OUTPUT_FOLDER"

# 4. Score sequences (via SLURM)
sbatch post_processing_scripts/eval_all.sh "$OUTPUT_FOLDER"
```

### Notes

- Run generation from `fmif/` (or ensure `gen_inference_finetune.py` is on `PYTHONPATH` / cwd), since the shell script calls `python gen_inference_finetune.py` with a relative path.
- `eval_all.sh` hardcodes `SCRIPT_DIR` to the absolute post-processing path; adjust if you move the repo.
- Trajectory dumps (`full_traj_*.pkl`) are per-worker when `SAVE_FULL_TRAJ_DATASET=True`; they are separate from the CSV merge step.
