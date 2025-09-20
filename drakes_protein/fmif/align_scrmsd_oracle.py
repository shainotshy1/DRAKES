import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)

from biotite.sequence.io import fasta
import shutil
from protein_oracle.data_utils import ALPHABET
import torch
import os
import shutil
import warnings
from multiflow.models import folding_model
from types import SimpleNamespace
import pyrosetta # Assumes it is already inititialized!
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta import * # type: ignore

from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset
from torch.utils.data import DataLoader, ConcatDataset
import pickle

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

def relax_pose(pose):
    scorefxn = pyrosetta.create_score_function("ref2015_cart")
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(pose))
    packer.apply(pose)
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.apply(pose)

def calc_new_pose(seq, model, base_path, test_name):
    sc_output_dir_base = os.path.join(base_path, f'sc_{test_name}')
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
    # relax_pose(pose) Uncommenting to improve runtime since unrelaxed is still reasonable in alignment
    return pose

def get_true_pose(protein_name, pdb_paths):
    true_pose = pyrosetta.pose_from_file(pdb_paths[protein_name])
    relax_pose(true_pose)
    return true_pose

def calc_rcsmd(pose, true_pose):
    return pyrosetta.rosetta.core.scoring.bb_rmsd(true_pose, pose)

def get_drakes_test_data():
    base_path = "/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
    pdb_path = os.path.join(base_path, 'proteindpo_data/AlphaFold_model_PDBs')
    max_len = 75  # Define the maximum length of proteins
    dataset = ProteinStructureDataset(pdb_path, max_len) # max_len set to 75 (sequences range from 31 to 74)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    for batch in loader:
        pdb_structures = batch[0]
        pdb_filenames = batch[1]
        pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
        break
    dpo_dict_path = os.path.join(base_path, 'proteindpo_data/processed_data')
    
    dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
    dpo_valid_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict_wt.pkl'), 'rb'))
    dpo_train_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict_wt.pkl'), 'rb'))

    dpo_test_dataset = ProteinDPODataset(dpo_test_dict, pdb_idx_dict, pdb_structures) #type: ignore
    dpo_valid_dataset = ProteinDPODataset(dpo_valid_dict, pdb_idx_dict, pdb_structures) #type: ignore
    dpo_train_dataset = ProteinDPODataset(dpo_train_dict, pdb_idx_dict, pdb_structures) #type: ignore

    combined_dataset = ConcatDataset([dpo_test_dataset, dpo_valid_dataset, dpo_train_dataset])

    loader_test = DataLoader(combined_dataset, batch_size=1, shuffle=False)
    return base_path, pdb_path, loader_test

def build_scRMSD_oracle(protein_name, mask_for_loss, test_name, device_id=0):
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
    
    true_pose = get_true_pose(protein_name, pdb_paths)

    logging.info(f"Built scRMSD Reward Oracle for {protein_name[:protein_name.index('.pdb')]}!")

    def reward_oracle(ssps):
        res = torch.zeros((ssps.shape[0]), device=ssps.device)
        for i, ssp in enumerate(ssps):
            ssp_str = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[0][_ix] == 1])
            pose = calc_new_pose(ssp_str, model, base_path, test_name)
            res[i] = -1 * calc_rcsmd(pose, true_pose) # Flip sign since we want to MINIMIZE scRMSD
        return res
        
    return reward_oracle

