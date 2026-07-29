"""Train a *noisy* (soft value function) protein oracle.

This script is based on ``protein_oracle/train_oracle.py`` but instead of
regressing the ddG of clean sequences, it learns a soft value function
``f(x_t, t)`` over the *noisy* (partially-masked) sequences produced along the
FMIF denoising trajectory, following the Monte-Carlo regression objective of
Algorithm 2 ("Value Function Estimation using Monte Carlo Regression"):

    f_hat = argmin_f  sum_t sum_s [ exp(r(x_0^{(s)}) / alpha)
                                    - exp(f(x_t^{(s)}, t) / alpha) ]^2

Key differences from ``train_oracle.py``:
  * The model (a ``ProteinMPNNOracle``) is initialized from the weights of the
    *alignment* reward oracle (``reward_oracle_ft.pt``), NOT the evaluation
    oracle (``reward_oracle_eval.pt``).
  * A ``--timestamp`` is required and is used to tag the output run directory.
  * Training samples are the trajectory records stored in
    ``align_oracle_data/full_traj_train_pretrained.pkl``. We select proteins
    using the same technique as ``train_oracle.py`` (the curated training dict
    ``dpo_train_dict_curated.pkl``) and then look up the full-trajectory records
    whose protein name matches those training proteins.
  * ``t`` (the diffusion step) enters ``f(x_t, t)`` implicitly through the
    masking level of ``x_t``: the ``ProteinMPNNOracle`` architecture is reused
    unchanged so that the alignment-oracle weights can initialize it directly,
    and the mask token is embedded through the model's amino-acid embedding
    (vocab size 22, mask index 21).

Validation/testing temporarily reuse the training trajectory pkl until the
corresponding validation/test trajectory pkls are generated.
"""

import argparse
import csv
import random
import string
import datetime
from datetime import date
import pickle
import os

import numpy as np
from scipy.stats import pearsonr

from protein_oracle.data_utils import ALPHABET, MASK_TOKEN_INDEX
from protein_oracle.utils import str2bool

runid = ''.join(random.choice(string.ascii_letters) for i in range(10)) + '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

# Mapping from the trajectory sequence characters to model token indices.
# The saved trajectory sequences use ``mu.ALPHABET + '-'`` where '-' marks a
# still-masked (not-yet-denoised) position (see fmif/fm_utils.py). We map the
# 21 amino-acid characters to indices 0..20 and '-' to the mask token index.
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
CHAR_TO_IDX['-'] = MASK_TOKEN_INDEX


def safe_pearson(x, y):
    """Pearson r that returns nan on degenerate input instead of warning.

    Uses the same exact-uniqueness condition scipy uses internally, so a constant
    input is detected reliably (``np.std > 0`` can be fooled by floating-point
    round-off). Non-finite entries are dropped first.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float('nan')
    corr, _ = pearsonr(x, y)
    return float(corr)


def grouped_pearson(true_vals, pred_vals, keys):
    """Mean Pearson r within groups defined by ``keys``.

    Degenerate groups (constant true or predicted values) and non-finite
    correlations are dropped before averaging, so a single bad group can never
    poison the mean. Returns ``(mean_r, n_groups_used)``.
    """
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    keys = np.asarray(keys)
    corrs = []
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        corr = safe_pearson(true_vals[idx], pred_vals[idx])
        if np.isfinite(corr):
            corrs.append(corr)
    return (float(np.mean(corrs)) if corrs else float('nan')), len(corrs)


def _normalize_protein_name(name):
    """Normalize a protein/WT name so that trajectory names and dpo-dict names
    can be matched.

    Trajectory ``protein_name`` values look like ``EA|run2_0325_0005`` (pipe,
    no extension), while the dpo-dict WT names look like ``EA:run2_0325_0005.pdb``
    (idx 1, colon, ``.pdb``) or ``EA|run2_0325_0005.pdb`` (idx 7, pipe, ``.pdb``).
    """
    name = name.strip('_')
    if name.endswith('.pdb'):
        name = name[:-4]
    return name.replace(':', '|')


def main(args):
    import time
    import warnings
    import collections
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    from protein_oracle.utils import set_seed
    from protein_oracle.data_utils import ProteinStructureDataset
    from protein_oracle.model_utils import ProteinMPNNOracle
    from tqdm import tqdm
    warnings.filterwarnings("ignore", category=UserWarning)

    scaler = torch.cuda.amp.GradScaler()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # ------------------------------------------------------------------ output
    path_for_outputs = os.path.join(args.base_path, 'protein_oracle/noisy_oracle_outputs') if not args.debug \
        else os.path.join(args.base_path, 'protein_oracle/noisy_oracle_outputs_debug')
    base_folder = time.strftime(path_for_outputs, time.localtime())
    # Tag the run with the provided timestamp so it can be located later.
    base_folder = os.path.join(base_folder, args.timestamp, runid)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    if base_folder[-1] != '/':
        base_folder += '/'
    for subfolder in ['model_weights']:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    assert torch.cuda.is_available(), "CUDA is not available"
    set_seed(args.seed, use_cuda=True)

    with open(logfile, 'a') as f:
        f.write(f"Date: {date.today().strftime('%B %d, %Y')}\n")
        f.write(f"Run ID: {runid}\n")
        f.write(f"Timestamp: {args.timestamp}\n")
        f.write(f"Arguments: {args}\n")

    # ------------------------------------------------------------- structures
    pdb_path = os.path.join(args.base_path, 'proteindpo_data/AlphaFold_model_PDBs')
    max_len = 75  # sequences range from 31 to 74
    dataset = ProteinStructureDataset(pdb_path, max_len)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    for batch in loader:
        pdb_structures = batch[0]
        pdb_filenames = batch[1]
        pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
        break

    # ------------------------------------------------- dpo dicts (selection + structure keys)
    dpo_dict_path = os.path.join(args.base_path, 'proteindpo_data/processed_data')
    dpo_train_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict_curated.pkl'), 'rb'))
    dpo_valid_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict.pkl'), 'rb'))
    dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict.pkl'), 'rb'))

    # Map normalized protein name -> structure key (the colon/".pdb" name used
    # as key into pdb_idx_dict, stored at index 1 of each dpo-dict value).
    name_to_pdbkey = {}
    for dd in (dpo_train_dict, dpo_valid_dict, dpo_test_dict):
        for v in dd.values():
            name_to_pdbkey[_normalize_protein_name(v[7])] = v[1]

    # "Same technique as train_oracle.py" for selecting training proteins: use
    # the curated training dict. The resulting set of (normalized) WT/protein
    # names is what we look up inside the full trajectory dataset.
    train_name_set = set(_normalize_protein_name(v[7]) for v in dpo_train_dict.values())
    valid_name_set = set(_normalize_protein_name(v[7]) for v in dpo_valid_dict.values())
    test_name_set = set(_normalize_protein_name(v[7]) for v in dpo_test_dict.values())

    with open(logfile, 'a') as f:
        f.write(f"Selection proteins -> train: {len(train_name_set)}, valid: {len(valid_name_set)}, test: {len(test_name_set)}\n")

    # ------------------------------------------------------------- dataset
    class NoisyTrajDataset(Dataset):
        """Yields one (noisy sequence x_t, timestep t, reward r(x_0)) sample per
        trajectory record, paired with its protein's backbone structure."""

        def __init__(self, records):
            # records: list of (struct_idx, sequence_str, t, reward, name)
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            struct_idx, seq, t, reward, name = self.records[idx]
            return {
                'protein_name': name,
                'sequence': seq,
                't': t,
                'reward': reward,
                'structure': pdb_structures[struct_idx],
            }

    def build_records(traj_pkl_path, name_set, split_label):
        """Load a full-trajectory pkl and keep only records whose protein name
        matches ``name_set`` (and for which we know the backbone structure)."""
        with open(traj_pkl_path, 'rb') as fh:
            data = pickle.load(fh)
        records = []
        skipped_no_struct = set()
        for r in data:
            nm = _normalize_protein_name(r['protein_name'])
            if nm not in name_set:
                continue
            pdbkey = name_to_pdbkey.get(nm)
            if pdbkey is None or pdbkey not in pdb_idx_dict:
                skipped_no_struct.add(nm)
                continue
            records.append((
                pdb_idx_dict[pdbkey],
                r['sequence'],
                int(r['t']),
                float(r[args.reward_key]),
                nm,
            ))
        del data
        with open(logfile, 'a') as f:
            f.write(f"[{split_label}] {traj_pkl_path}: {len(records)} records over "
                    f"{len(set(x[4] for x in records))} proteins"
                    + (f" (skipped {len(skipped_no_struct)} w/o structure)" if skipped_no_struct else "")
                    + "\n")
        return records

    traj_dir = args.traj_data_dir
    train_traj_pkl = os.path.join(traj_dir, args.train_traj_pkl)
    train_records = build_records(train_traj_pkl, train_name_set, 'train')

    # Validation / testing temporarily reuse the training trajectory pkl. When
    # dedicated valid/test trajectory pkls are generated, pass them via
    # --valid_traj_pkl / --test_traj_pkl (with the matching selection sets).
    if args.valid_traj_pkl:
        valid_records = build_records(os.path.join(traj_dir, args.valid_traj_pkl), valid_name_set, 'valid')
    else:
        with open(logfile, 'a') as f:
            f.write("[valid] reusing train trajectory records (placeholder)\n")
        valid_records = train_records
    if args.test_traj_pkl:
        test_records = build_records(os.path.join(traj_dir, args.test_traj_pkl), test_name_set, 'test')
    else:
        with open(logfile, 'a') as f:
            f.write("[test] reusing train trajectory records (placeholder)\n")
        test_records = train_records

    def log_split_diagnostics(records, split_label):
        """Report how much *independent* signal a split actually contains.

        Each trajectory shares one r(x_0) by construction, but if every protein
        also has only a single distinct r(x_0) then a Pearson correlation
        computed *within* a protein is mathematically undefined (constant input),
        so we warn and rely on the across-protein metrics instead.
        """
        rewards_by_prot = collections.defaultdict(set)
        seqs_by_prot = collections.defaultdict(set)
        for _, seq, _, reward, nm in records:
            rewards_by_prot[nm].add(reward)
            seqs_by_prot[nm].add(seq)
        n_prot = len(rewards_by_prot)
        multi = sum(1 for v in rewards_by_prot.values() if len(v) > 1)
        distinct_rewards = len(set(r for v in rewards_by_prot.values() for r in v))
        mean_seqs = (sum(len(v) for v in seqs_by_prot.values()) / n_prot) if n_prot else 0
        lines = [f"[{split_label}] proteins={n_prot}, distinct r(x_0) over split={distinct_rewards}, "
                 f"proteins with >1 distinct r(x_0)={multi}, mean distinct sequences/protein={mean_seqs:.1f}"]
        if n_prot and multi == 0:
            lines.append(f"[{split_label}] WARNING: every protein has exactly ONE distinct r(x_0) "
                         f"(the generation repeats are duplicates), so Pearson grouped WITHIN a protein "
                         f"is undefined. Use the pooled / by-timestep correlations instead.")
        with open(logfile, 'a') as f:
            for ln in lines:
                f.write(ln + "\n")
        for ln in lines:
            print(ln)

    log_split_diagnostics(train_records, 'train')
    if valid_records is not train_records:
        log_split_diagnostics(valid_records, 'valid')
    if test_records is not train_records:
        log_split_diagnostics(test_records, 'test')

    train_dataset = NoisyTrajDataset(train_records)
    valid_dataset = NoisyTrajDataset(valid_records)
    test_dataset = NoisyTrajDataset(test_records)

    # The eval loaders are shuffled as well. Records in the pkl are grouped by
    # protein, so with shuffle=False and a capped number of eval batches (--debug
    # or --max_eval_batches) evaluation would only ever see the first protein --
    # a single group with a constant reward, which makes every grouped metric
    # degenerate. Dedicated generators are re-seeded before each evaluation pass
    # so the eval subset is identical across epochs (comparable metrics) while
    # still spanning many proteins and timesteps.
    valid_gen = torch.Generator()
    test_gen = torch.Generator()

    loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_data_workers)
    loader_valid = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_data_workers, generator=valid_gen)
    loader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_data_workers, generator=test_gen)

    # ------------------------------------------------------------- featurize
    def featurize_noisy(batch, targ_len=75):
        """Turn a batch of trajectory records into model inputs. ``S`` carries
        the noisy sequence x_t (masked positions -> mask token index)."""
        seqs = batch['sequence']
        B = len(seqs)
        L_max = min(max(len(s) for s in seqs), targ_len)
        X = batch['structure'][:, :L_max, :, :].to(dtype=torch.float32, device=device)
        S = np.zeros([B, L_max], dtype=np.int64)
        mask = np.zeros([B, L_max], dtype=np.float32)
        residue_idx = -100 * np.ones([B, L_max], dtype=np.int64)
        for i, seq in enumerate(seqs):
            seq = seq[:targ_len]
            S[i, :len(seq)] = np.asarray([CHAR_TO_IDX[c] for c in seq], dtype=np.int64)
            mask[i, :len(seq)] = 1.0
            residue_idx[i, :len(seq)] = np.arange(len(seq))
        S = torch.from_numpy(S).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
        residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
        chain_M = mask.clone()
        chain_encoding_all = mask.clone()
        reward = batch['reward'].to(dtype=torch.float32, device=device)
        return X, S, mask, chain_M, residue_idx, chain_encoding_all, reward

    # ------------------------------------------------------------- model
    model = ProteinMPNNOracle(node_features=args.hidden_dim,
                              edge_features=args.hidden_dim,
                              hidden_dim=args.hidden_dim,
                              num_encoder_layers=args.num_encoder_layers,
                              num_decoder_layers=args.num_encoder_layers,
                              k_neighbors=args.num_neighbors,
                              dropout=args.dropout,
                              augment_eps=args.backbone_noise)
    model.to(device)

    # Initialize with the ALIGNMENT reward oracle weights (not the eval oracle).
    if args.initialize_with_align_oracle:
        align_ckpt = args.align_oracle_ckpt
        if not os.path.isabs(align_ckpt):
            align_ckpt = os.path.join(args.base_path, align_ckpt)
        align_state = torch.load(align_ckpt, map_location=device)['model_state_dict']
        missing, unexpected = model.load_state_dict(align_state, strict=False)
        with open(logfile, 'a') as f:
            f.write(f"Initialized from alignment oracle: {align_ckpt}\n")
            if missing:
                f.write(f"  missing keys: {missing}\n")
            if unexpected:
                f.write(f"  unexpected keys: {unexpected}\n")

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step']
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if PATH:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    alpha = args.alpha
    # Clamp the exponent for numerical stability (esp. with small alpha / mixed precision).
    exp_clamp = args.exp_clamp

    def value_loss(pred, reward):
        """Algorithm 2 objective: MSE between exp(r(x_0)/alpha) and exp(f(x_t)/alpha)."""
        pred = pred.float()
        reward = reward.float()
        target = torch.exp(torch.clamp(reward / alpha, max=exp_clamp))
        pred_exp = torch.exp(torch.clamp(pred / alpha, max=exp_clamp))
        return F.mse_loss(pred_exp, target)

    # ------------------------------------------------------------- curves
    curves_csv = base_folder + 'training_curves.csv'
    curves_png = base_folder + 'training_curves.png'
    history = []

    CURVE_PANELS = [
        ('loss', ['train_loss', 'valid_loss', 'test_loss']),
        ('pearson (pooled)', ['train_pearson', 'valid_pearson', 'test_pearson']),
        ('pearson (by t)', ['train_pearson_by_t', 'valid_pearson_by_t', 'test_pearson_by_t']),
        ('pearson (by protein)', ['train_pearson_by_protein', 'valid_pearson_by_protein',
                                  'test_pearson_by_protein']),
    ]

    def save_curves():
        """Write the per-epoch metric history to csv and render it as a figure.

        Called at the end of every epoch so an interrupted run still leaves
        usable curves behind.
        """
        if not history:
            return
        with open(curves_csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = [h['epoch'] for h in history]
        fig, axes = plt.subplots(1, len(CURVE_PANELS), figsize=(5 * len(CURVE_PANELS), 4))
        for ax, (title, keys) in zip(np.atleast_1d(axes), CURVE_PANELS):
            for k in keys:
                ax.plot(epochs, [h[k] for h in history], marker='o', markersize=3,
                        label=k.split('_')[0])
            ax.set_xlabel('epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend()
        losses = [h[k] for h in history for k in CURVE_PANELS[0][1] if np.isfinite(h[k])]
        if losses and min(losses) > 0:
            np.atleast_1d(axes)[0].set_yscale('log')
        fig.tight_layout()
        fig.savefig(curves_png, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------- train loop
    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_loss = 0.
        train_r_true = []
        train_r_pred = []
        train_names = []
        train_ts = []
        n_train_batches = 0
        for _, batch in tqdm(enumerate(loader_train)):
            if args.debug and _ > 100:
                break
            X, S, mask, chain_M, residue_idx, chain_encoding_all, reward = featurize_noisy(batch)

            optimizer.zero_grad()
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    r_pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                loss = value_loss(r_pred, reward)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                r_pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                loss = value_loss(r_pred, reward)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            total_step += 1
            n_train_batches += 1
            train_r_true.extend(reward.cpu().data.numpy())
            train_r_pred.extend(r_pred.detach().float().cpu().data.numpy())
            train_names.extend(batch['protein_name'])
            train_ts.extend(batch['t'].cpu().numpy())

        # Cap on eval batches: --debug keeps the old 21-batch behaviour, and
        # --max_eval_batches (0 = all) bounds evaluation cost on the full split.
        if args.debug:
            max_eval_batches = 21
        else:
            max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

        model.eval()
        with torch.no_grad():
            # Re-seed so every epoch evaluates on the same shuffled subset.
            valid_gen.manual_seed(args.seed)
            test_gen.manual_seed(args.seed)

            validation_loss = 0.
            valid_r_true, valid_r_pred, valid_names, valid_ts = [], [], [], []
            n_valid_batches = 0
            for _, batch in enumerate(loader_valid):
                if max_eval_batches is not None and _ >= max_eval_batches:
                    break
                X, S, mask, chain_M, residue_idx, chain_encoding_all, reward = featurize_noisy(batch)
                r_pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                validation_loss += value_loss(r_pred, reward).item()
                n_valid_batches += 1
                valid_r_true.extend(reward.cpu().data.numpy())
                valid_r_pred.extend(r_pred.float().cpu().data.numpy())
                valid_names.extend(batch['protein_name'])
                valid_ts.extend(batch['t'].cpu().numpy())

            test_loss = 0.
            test_r_true, test_r_pred, test_names, test_ts = [], [], [], []
            n_test_batches = 0
            for _, batch in enumerate(loader_test):
                if max_eval_batches is not None and _ >= max_eval_batches:
                    break
                X, S, mask, chain_M, residue_idx, chain_encoding_all, reward = featurize_noisy(batch)
                r_pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                test_loss += value_loss(r_pred, reward).item()
                n_test_batches += 1
                test_r_true.extend(reward.cpu().data.numpy())
                test_r_pred.extend(r_pred.float().cpu().data.numpy())
                test_names.extend(batch['protein_name'])
                test_ts.extend(batch['t'].cpu().numpy())

        train_loss = train_loss / max(n_train_batches, 1)
        validation_loss = validation_loss / max(n_valid_batches, 1)
        test_loss = test_loss / max(n_test_batches, 1)

        # Primary metric: pooled across the whole split. Well-defined even when
        # each protein has a single r(x_0), because r varies *across* proteins.
        train_pearson = safe_pearson(train_r_true, train_r_pred)
        validation_pearson = safe_pearson(valid_r_true, valid_r_pred)
        test_pearson = safe_pearson(test_r_true, test_r_pred)

        # Per-timestep: at a fixed t, correlate f(x_t) against r(x_0) across
        # proteins, then average over t. This is the metric that actually makes
        # sense for a value function (it shows how the value estimate sharpens
        # as t advances) and is unaffected by the duplicate-repeat issue.
        train_pearson_t, train_nt = grouped_pearson(train_r_true, train_r_pred, train_ts)
        validation_pearson_t, valid_nt = grouped_pearson(valid_r_true, valid_r_pred, valid_ts)
        test_pearson_t, test_nt = grouped_pearson(test_r_true, test_r_pred, test_ts)

        # Within-protein (the old metric), kept as a diagnostic. It is undefined
        # whenever a protein has only one distinct r(x_0); n_groups=0 makes that
        # visible instead of silently reporting nan.
        train_pearson_p, train_np = grouped_pearson(train_r_true, train_r_pred, train_names)
        validation_pearson_p, valid_np = grouped_pearson(valid_r_true, valid_r_pred, valid_names)
        test_pearson_p, test_np = grouped_pearson(test_r_true, test_r_pred, test_names)

        train_loss_ = np.format_float_positional(np.float32(train_loss), unique=False, precision=3)
        validation_loss_ = np.format_float_positional(np.float32(validation_loss), unique=False, precision=3)
        test_loss_ = np.format_float_positional(np.float32(test_loss), unique=False, precision=3)
        train_pearson_ = np.format_float_positional(np.float32(train_pearson), unique=False, precision=3)
        validation_pearson_ = np.format_float_positional(np.float32(validation_pearson), unique=False, precision=3)
        test_pearson_ = np.format_float_positional(np.float32(test_pearson), unique=False, precision=3)

        def _fmt(v):
            return np.format_float_positional(np.float32(v), unique=False, precision=3)

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
        msg = (f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss_}, '
               f'valid: {validation_loss_}, test: {test_loss_}, train_pearson: {train_pearson_}, '
               f'valid_pearson: {validation_pearson_}, test_pearson: {test_pearson_}')
        # Grouped diagnostics: "(n)" is the number of groups that contributed, so
        # a nan is always attributable to "no non-degenerate groups".
        msg2 = (f'    by_t     -> train: {_fmt(train_pearson_t)} (n={train_nt}), '
                f'valid: {_fmt(validation_pearson_t)} (n={valid_nt}), '
                f'test: {_fmt(test_pearson_t)} (n={test_nt})\n'
                f'    by_prot  -> train: {_fmt(train_pearson_p)} (n={train_np}), '
                f'valid: {_fmt(validation_pearson_p)} (n={valid_np}), '
                f'test: {_fmt(test_pearson_p)} (n={test_np})')
        with open(logfile, 'a') as f:
            f.write(msg + '\n')
            f.write(msg2 + '\n')
        print(msg)
        print(msg2)

        history.append({
            'epoch': e + 1, 'step': total_step,
            'train_loss': train_loss, 'valid_loss': validation_loss, 'test_loss': test_loss,
            'train_pearson': train_pearson, 'valid_pearson': validation_pearson,
            'test_pearson': test_pearson,
            'train_pearson_by_t': train_pearson_t, 'valid_pearson_by_t': validation_pearson_t,
            'test_pearson_by_t': test_pearson_t,
            'train_pearson_by_protein': train_pearson_p,
            'valid_pearson_by_protein': validation_pearson_p,
            'test_pearson_by_protein': test_pearson_p,
        })
        save_curves()

        checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'
        torch.save({
            'epoch': e + 1,
            'step': total_step,
            'num_edges': args.num_neighbors,
            'noise_level': args.backbone_noise,
            'alpha': alpha,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_filename_last)

        if (e + 1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = base_folder + 'model_weights/epoch{}_step{}.pt'.format(e + 1, total_step)
            torch.save({
                'epoch': e + 1,
                'step': total_step,
                'num_edges': args.num_neighbors,
                'noise_level': args.backbone_noise,
                'alpha': alpha,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_filename)

    save_curves()
    with open(logfile, 'a') as f:
        f.write(f"Saved training curves: {curves_csv}, {curves_png}\n")
    print(f"Saved training curves: {curves_csv}, {curves_png}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--timestamp", type=str, required=True, help="Timestamp tag for this run (used in the output directory)")
    argparser.add_argument("--base_path", type=str, default="/u/sdickman/DRAKES/data_and_model", help="base path for data and model")
    argparser.add_argument("--traj_data_dir", type=str, default="/u/sdickman/DRAKES/drakes_protein/fmif/align_oracle_data", help="directory containing the full-trajectory pkl files")
    argparser.add_argument("--train_traj_pkl", type=str, default="full_traj_train_pretrained.pkl", help="filename (within traj_data_dir) of the training trajectory dataset")
    argparser.add_argument("--valid_traj_pkl", type=str, default="full_traj_validation_pretrained.pkl", help="validation trajectory pkl; empty => reuse the training pkl (temporary placeholder)")
    argparser.add_argument("--test_traj_pkl", type=str, default="full_traj_test_pretrained.pkl", help="test trajectory pkl; empty => reuse the training pkl (temporary placeholder)")
    argparser.add_argument("--reward_key", type=str, default="final_align_reward", choices=["final_align_reward", "final_eval_reward"], help="which stored reward r(x_0) to regress against")

    argparser.add_argument("--align_oracle_ckpt", type=str, default="protein_oracle/outputs/reward_oracle_ft.pt", help="alignment oracle checkpoint used to initialize the model (relative to base_path unless absolute)")
    argparser.add_argument("--initialize_with_align_oracle", type=str2bool, default=True, help="initialize model weights with the alignment oracle")

    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--batch_size", type=int, default=128, help="number of sequences for one batch")
    argparser.add_argument("--num_data_workers", type=int, default=4, help="DataLoader worker processes")
    argparser.add_argument("--max_eval_batches", type=int, default=0, help="cap on validation/test batches per epoch (0 = use the whole split); eval loaders are shuffled so a cap still spans many proteins")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers")
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=str2bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--mixed_precision", type=str2bool, default=True, help="train with mixed precision")
    argparser.add_argument("--alpha", type=float, default=1.0, help="temperature alpha in the exp() soft-value objective")
    argparser.add_argument("--exp_clamp", type=float, default=30.0, help="clamp on the exponent (r/alpha, f/alpha) for numerical stability")
    argparser.add_argument("--lr", type=float, default=1e-4)
    argparser.add_argument("--wd", type=float, default=1e-4)
    argparser.add_argument("--seed", type=int, default=0)

    args = argparser.parse_args()
    print(args)
    main(args)
