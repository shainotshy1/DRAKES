import os
import pandas as pd
from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
from torch.utils.data import DataLoader, ConcatDataset
import pickle

if __name__ == '__main__':
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
    
    seqs = [batch for batch in loader_test]
    seqs = sorted(seqs, key=lambda s : len(s['aa_seq_wt'][0]))
    for s in seqs:
        print(s['aa_seq_wt'][0], " --- ", len(s['aa_seq_wt'][0]), s['protein_name'][0][:-4])