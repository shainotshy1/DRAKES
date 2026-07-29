# Noisy (Soft Value Function) Oracle Training

`train_noisy_oracle.py` trains a **soft value function** `f(x_t, t)` over the
*noisy* (partially-masked) sequences produced along the FMIF denoising
trajectory. It is adapted from `protein_oracle/train_oracle.py`, but instead of
regressing the ddG of clean sequences it fits the Monte-Carlo regression
objective (Algorithm 2, "Value Function Estimation using Monte Carlo
Regression"):

```
f_hat = argmin_f  sum_t sum_s [ exp( r(x_0^{(s)}) / alpha )
                                - exp( f(x_t^{(s)}, t) / alpha ) ]^2
```

- `x_t^{(s)}` — the partially-masked sequence at diffusion step `t` for trajectory `s`.
- `r(x_0^{(s)})` — the reward at the fully-denoised sequence (from the reward oracle).
- `alpha` — temperature of the exponential / soft value function.

---

## What the script does

1. **Model init from the alignment oracle.** The model is a `ProteinMPNNOracle`
   initialized from the **alignment** reward oracle
   (`protein_oracle/outputs/reward_oracle_ft.pt`), **not** the evaluation oracle
   (`reward_oracle_eval.pt`). The architecture is reused unchanged so the
   alignment weights load with no missing/unexpected keys.

2. **`t` is implicit.** The noisy sequence `x_t` is fed directly into the
   oracle; still-masked positions use the model's mask token (embedding index
   `21`). The step dependence of `f(x_t, t)` is therefore carried by the masking
   level of `x_t`, which keeps the architecture identical to the alignment
   oracle (a prerequisite for the weight initialization above).

3. **Sample selection + trajectory lookup.** Training proteins are selected with
   the same technique as `train_oracle.py` — the curated training dict
   `proteindpo_data/processed_data/dpo_train_dict_curated.pkl`. The script then
   looks up the full-trajectory records in
   `align_oracle_data/full_traj_train_pretrained.pkl` whose protein name matches
   those training proteins, and uses those (noisy-sequence, step, reward)
   records as the training samples.

   Names are normalized to reconcile the two sources (trajectory names look like
   `EA|run2_0325_0005`; dpo-dict names look like `EA:run2_0325_0005.pdb`). This
   matches **310 of 311** curated proteins, ~**1.51M** training records.

4. **Objective.** For each record, loss = `MSE( exp(r(x_0)/alpha), exp(f(x_t)/alpha) )`,
   using the stored `final_align_reward` as `r(x_0)` by default.

---

## Prerequisites

- **Environment:** `mf2` (micromamba).
- **GPU:** required. The script asserts `torch.cuda.is_available()` (same as
  `train_oracle.py`), so run it on a GPU node, not the login node.
- **Data / checkpoints** (under `--base_path`, default
  `/u/sdickman/DRAKES/data_and_model`):
  - `proteindpo_data/AlphaFold_model_PDBs/` — backbone structures.
  - `proteindpo_data/processed_data/dpo_train_dict_curated.pkl` (+ `dpo_valid_dict.pkl`, `dpo_test_dict.pkl`).
  - `protein_oracle/outputs/reward_oracle_ft.pt` — alignment oracle used for init.
- **Trajectory data** (under `--traj_data_dir`, default
  `/u/sdickman/DRAKES/drakes_protein/fmif/align_oracle_data`):
  - `full_traj_train_pretrained.pkl` — the aggregated training trajectories.
    (Produced by concatenating the per-worker `full_traj_train_pretrained_*.pkl`
    dumps; see `prepare_oracle_data.ipynb`.)

---

## Running

Run from the `fmif/` directory so the `protein_oracle` package and local
modules resolve.

```bash
cd /u/sdickman/DRAKES/drakes_protein/fmif
eval "$(micromamba shell hook --shell bash)"
micromamba activate mf2

python train_noisy_oracle.py \
    --timestamp 20260727_0810 \
    --alpha 1.0 \
    --num_epochs 100 \
    --batch_size 128
```

`--timestamp` is **required** and tags the output run directory.

### Quick debug run

Trains on ~20 batches per split:

```bash
python train_noisy_oracle.py --timestamp debug --debug True
```

---

## Key arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `--timestamp` | *(required)* | Tag for the output run directory |
| `--base_path` | `/u/sdickman/DRAKES/data_and_model` | Root for data + checkpoints |
| `--traj_data_dir` | `.../fmif/align_oracle_data` | Directory holding the trajectory pkls |
| `--train_traj_pkl` | `full_traj_train_pretrained.pkl` | Training trajectory dataset (filename in `traj_data_dir`) |
| `--valid_traj_pkl` | `""` | Validation pkl; empty ⇒ reuse the training pkl (placeholder) |
| `--test_traj_pkl` | `""` | Test pkl; empty ⇒ reuse the training pkl (placeholder) |
| `--reward_key` | `final_align_reward` | Which stored reward to regress (`final_align_reward` or `final_eval_reward`) |
| `--align_oracle_ckpt` | `protein_oracle/outputs/reward_oracle_ft.pt` | Checkpoint used to initialize the model (relative to `--base_path` unless absolute) |
| `--initialize_with_align_oracle` | `True` | Load the alignment-oracle weights before training |
| `--alpha` | `1.0` | Temperature in the `exp()` objective |
| `--exp_clamp` | `30.0` | Clamp on `r/alpha`, `f/alpha` for numerical stability |
| `--num_epochs` | `100` | Epochs |
| `--batch_size` | `128` | Batch size |
| `--num_data_workers` | `4` | DataLoader workers |
| `--lr` / `--wd` | `1e-4` / `1e-4` | Optimizer learning rate / weight decay |
| `--mixed_precision` | `True` | Train with AMP |
| `--previous_checkpoint` | `""` | Resume from a checkpoint |
| `--save_model_every_n_epochs` | `10` | Periodic checkpoint interval |

Model-shape args (`--hidden_dim`, `--num_encoder_layers`, `--num_decoder_layers`,
`--num_neighbors`, `--dropout`, `--backbone_noise`) mirror `train_oracle.py` and
should match the alignment oracle you initialize from (defaults already do).

---

## Validation / testing (temporary)

Dedicated validation/test trajectory pkls do not exist yet. With
`--valid_traj_pkl` / `--test_traj_pkl` left empty (the default), both splits
**reuse the training records** as a placeholder, so validation/test metrics are
not yet meaningful.

Once you generate the real pkls, point the args at them; each split is then
filtered by its own selection set (`dpo_valid_dict` / `dpo_test_dict`):

```bash
python train_noisy_oracle.py \
    --timestamp 20260727_0810 \
    --valid_traj_pkl full_traj_valid_pretrained.pkl \
    --test_traj_pkl  full_traj_test_pretrained.pkl
```

---

## Outputs

Written under:

```
<base_path>/protein_oracle/noisy_oracle_outputs/<timestamp>/<runid>/
    log.txt                        # config + per-epoch train/valid/test loss & grouped Pearson
    training_curves.csv            # one row per epoch: losses + all Pearson variants
    training_curves.png            # loss / pooled / by-t / by-protein curves for all splits
    model_weights/
        epoch_last.pt              # latest checkpoint (overwritten each epoch)
        epoch{E}_step{S}.pt        # periodic checkpoints
```

In `--debug` mode the base directory is `noisy_oracle_outputs_debug/` and wandb
is disabled. Non-debug runs log to the `protein_noisy_oracle` wandb project.

Each checkpoint stores `model_state_dict`, `optimizer_state_dict`, `epoch`,
`step`, `noise_level`, and the `alpha` used for training.

`training_curves.csv`/`.png` are rewritten at the end of every epoch, so an
interrupted run still leaves curves for the epochs it completed. The figure is
also uploaded to wandb as `training_curves` on non-debug runs.
