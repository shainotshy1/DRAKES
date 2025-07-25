import os
import pandas as pd
from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
from torch.utils.data import DataLoader, ConcatDataset
import pickle

def process_seq_data_csv(fn, target_col, process_fn):
    print(f"Processing {fn}...")
    try:
        df = pd.read_csv(fn, nrows=0)  # Read only the header
        if target_col in df.columns: # Don't re-compute if already done
            print(f"{fn} already processed - Skipping...")
        else:
            df = pd.read_csv(fn)
            value = process_fn(df)
            df[target_col] = value
            df.to_csv(fn, index=False)
    except Exception as e: 
        print(f"ERROR: {e}")
        pass

def process_seq_data_directory(dir_name, target_col, process_fn):
    if dir_name.lower().endswith(".csv"):
        process_seq_data_csv(dir_name, target_col, process_fn)
        return
    
    for fn in os.listdir(dir_name):
        file_path = os.path.join(dir_name, fn)
        if fn.lower().endswith(".csv"):
            process_seq_data_csv(file_path, target_col, process_fn)

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