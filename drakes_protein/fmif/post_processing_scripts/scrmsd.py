import argparse
from tqdm import tqdm
import argparse
import os.path
from protein_oracle.utils import str2bool
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from biotite.sequence.io import fasta
import shutil
from protein_oracle.data_utils import ALPHABET
import pandas as pd
import numpy as np
import torch
import os
import shutil
import warnings
import os.path
from tqdm import tqdm
from multiflow.models import folding_model
from types import SimpleNamespace
import pyrosetta
import csv
pyrosetta.init(extra_options="-out:level 100")
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta import *
import esm
import biotite.structure.io as bsio
from utils import process_seq_data_directory, get_drakes_test_data

def calc_rcsmd(pose, true_pose):
    return pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, pose)

def relax_pose(pose):
    scorefxn = pyrosetta.create_score_function("ref2015_cart")
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(pose))
    packer.apply(pose)
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.apply(pose)

def calc_new_pose(seq, model, cached_poses, base_path):
    sc_output_dir_base = os.path.join(base_path, 'sc_eval')
    if seq in cached_poses:
        return cached_poses[seq]
    sc_output_dir = os.path.join(sc_output_dir_base, 'folded')
    os.makedirs(sc_output_dir, exist_ok=True)
    os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)
    codesign_fasta = fasta.FastaFile()
    codesign_fasta['codesign_seq_1'] = seq
    codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
    codesign_fasta.write(codesign_fasta_path)

    folded_dir = os.path.join(sc_output_dir, 'folded')
    if os.path.exists(folded_dir):
        shutil.rmtree(folded_dir)
    os.makedirs(folded_dir, exist_ok=False)

    _ = model.fold_fasta(codesign_fasta_path, folded_dir)
    gen_folded_pdb_path = os.path.join(folded_dir, 'folded_codesign_seq_1.pdb')
    pose = pyrosetta.pose_from_file(gen_folded_pdb_path)
    relax_pose(pose)
    cached_poses[seq] = pose
    return pose

def get_true_pose(protein_name, pdb_paths, cached_poses):
    if protein_name in cached_poses:
        return cached_poses[protein_name]
    true_pose = pyrosetta.pose_from_file(pdb_paths[protein_name])
    relax_pose(true_pose)
    cached_poses[protein_name] = true_pose
    return true_pose

def extract_scrmsd_distr(df, seq_label, true_seq_label, protein_label, base_path, model, pdb_paths):
    assert seq_label in df.columns, f"'{seq_label}' must be a label in the data frame"
    sequences = df[seq_label]
    true_sequences = df[true_seq_label]
    true_protein_fns = df[protein_label]
    assert sequences.shape == true_sequences.shape == true_protein_fns.shape, "Must have same number of sequences and true sequences"
    cached_poses = {}
    cached_scrmsd = {}
    values = []
    for i in tqdm(range(sequences.size)):
        seq = sequences[i]
        true_sequence = true_sequences[i]
        true_protein_fn = true_protein_fns[i]
        if seq == true_sequence:
            values.append(0)
        elif (seq, true_sequence) in cached_scrmsd:
            values.append(cached_scrmsd[(seq, true_sequence)])
        else:
            pose = calc_new_pose(seq, model, cached_poses, base_path)
            true_pose = get_true_pose(true_protein_fn, pdb_paths, cached_poses)
            values.append(calc_rcsmd(pose, true_pose))
    values = np.array(values)
    return values

def extract_scrmsd_directory(dir_name, seq_label, true_seq_label, protein_label, device_id):
    base_path, pdb_path, loader_test = get_drakes_test_data()
    pdb_paths = {}
    for batch in loader_test:
        pdb_paths[batch['protein_name'][0]] = os.path.join(pdb_path, batch['WT_name'][0])

    folding_cfg = {
        'seq_per_sample': 1,
        'folding_model': 'esmf',
        'own_device': False,
        'pmpnn_path': './ProteinMPNN/',
        'pt_hub_dir': os.path.join(base_path, '.cache/torch/'),
        'colabfold_path': os.path.join(base_path, 'colabfold-conda/bin/colabfold_batch') # for AF2
    }
    folding_cfg = SimpleNamespace(**folding_cfg)
    model = folding_model.FoldingModel(folding_cfg, device_id=device_id)

    process_seq_data_directory(dir_name, 'scrmsd', lambda df : extract_scrmsd_distr(df, seq_label, true_seq_label, protein_label, base_path, model, pdb_paths))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ScRMSD Computation")
    parser.add_argument("dir", help="Directory containing distribution csv files")
    parser.add_argument("--gpu", help="GPU device index", default=0)
    args = parser.parse_args()

    device_id = int(args.gpu)
    extract_scrmsd_directory(args.dir, 'seq', 'true_seq', 'protein_name', device_id)