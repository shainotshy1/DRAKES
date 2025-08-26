import os
import pickle
import argparse
from tqdm import tqdm
import logging
import torch
import pandas as pd

from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
from torch.utils.data import DataLoader

from execute_align_utils import generate_execution_func # type: ignore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_drakes_data(base_path, dataset_name: str, val_batch_size: int, batch_repeat: int, target_protein = None):
    assert dataset_name != "single" or target_protein is not None, "If dataset is 'single', target_protein must be specified"

    pdb_path = os.path.join(base_path, 'proteindpo_data/AlphaFold_model_PDBs')
    logging.info(f"Reading protein set ({pdb_path})")

    max_len = 75  # Define the maximum length of proteins
    dataset = ProteinStructureDataset(pdb_path, max_len) # max_len set to 75 (sequences range from 31 to 74)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    pdb_idx_dict = {}
    pdb_structures = None
    for batch in loader:
        pdb_structures = batch[0]
        pdb_filenames = batch[1]
        pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
        break

    dpo_dict_path = os.path.join(base_path, 'proteindpo_data/processed_data')

    if dataset_name == "single":
        dpo_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
        dpo_dict.update(pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict_wt.pkl'), 'rb')))
        dpo_dict.update(pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict_wt.pkl'), 'rb')))
        
        target_protein_key = target_protein + ".pdb" # type: ignore
        assert target_protein_key in dpo_dict, f"Target protein {target_protein} not found in DPO dataset"
        
        single_dict = {target_protein_key: dpo_dict[target_protein_key]}

        dpo_dataset = ProteinDPODataset(single_dict, pdb_idx_dict, pdb_structures) #type: ignore
        loader_valid = DataLoader(dpo_dataset, batch_size=1, shuffle=False)
        logging.info(f"Executing on protein {target_protein} (Batch repeats: {batch_repeat}, Proteins per batch: {val_batch_size})")
    else:
        if dataset_name == "validation":
            dpo_pkl_file = 'dpo_valid_dict_wt.pkl'
        elif dataset_name == "test":
            dpo_pkl_file = 'dpo_test_dict_wt.pkl'
        elif dataset_name == "train":
            dpo_pkl_file ='dpo_train_dict_wt.pkl'
        else:
            raise ValueError()
        
        dpo_dict_path_v = os.path.join(base_path, 'proteindpo_data/processed_data')
        dpo_dict_full_path = os.path.join(dpo_dict_path_v, dpo_pkl_file)
        logging.info(f"Reading validation set ({dpo_dict_full_path})")

        dpo_dict = pickle.load(open(dpo_dict_full_path, 'rb'))

        dpo_dataset = ProteinDPODataset(dpo_dict, pdb_idx_dict, pdb_structures)
        loader_valid = DataLoader(dpo_dataset, batch_size=val_batch_size, shuffle=False)
        logging.info(f"Executing on {dataset_name} set (Batch repeats: {batch_repeat}, Proteins per batch: {val_batch_size}, Total proteins: {len(dpo_dataset)})")

    return loader_valid

def execute_on_dataset(func, base_path, dataset_name="validation", target_protein=None, batch_repeat=1, val_batch_size=1):
    assert dataset_name in ['validation', 'test', 'train', 'single'], f"Encountered dataset value '{dataset_name}' which is not in ['validation', 'test', 'single']"

    loader_valid = get_drakes_data(base_path, dataset_name, val_batch_size, batch_repeat, target_protein=target_protein)
    
    for batch in loader_valid:
        for _ in tqdm(range(batch_repeat)):
            func(batch)

def generate_output_fn(args):

    out_name = f"{args.model}"
    
    if args.dataset == "single":
        out_name += f"_{args.target_protein}"
    else:
        out_name += f"_{args.dataset}"
    
    out_name += f"_{args.oracle_mode}"
    
    if args.oracle_mode == "balanced":
        out_name += f"_alpha={args.oracle_alpha}"

    out_name += f"_{args.align_type}_N={args.align_n}"
    if args.align_type == "linear":
        out_name += f"_lambda={args.lasso_lambda}"

    full_out_path = f"{args.output_folder}/{out_name}.csv"
    return full_out_path

def main():
    logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.INFO)

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    argparser.add_argument("--base_path", required=True, type=str, help="Base path for data and model") 
    argparser.add_argument("--model", required=True, choices=['pretrained', 'drakes'], help="Model must be one of ['pretrained', 'drakes']")
    argparser.add_argument("--dataset", required=True, choices=['validation', 'test', 'train', 'single'], help="Dataset must be on of ['validation', 'test', 'train']")
    argparser.add_argument("--output_folder", type=str, required=True, help="Output folder for protein generations")
    argparser.add_argument("--align_type", choices=['bon', 'beam', 'spectral', 'linear'], required=True)
    argparser.add_argument("--oracle_mode", choices=['ddg', 'protgpt', 'balanced'], required=True)
    argparser.add_argument("--align_n", type=int, required=True, help="Number of samples parameter for alignment techniques")

    # Optional arguments
    argparser.add_argument("--batch_repeat", type=int, default=1, help="Number of times to repeat execution of each protein batch")
    argparser.add_argument("--batch_size", type=int, default=1, help="Number of times to generate each protein in a batch")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device to be used in execution script")
    argparser.add_argument("--oracle_alpha", type=float, help="Alpha parameter for balanced oracle mode, 1.0 = only ddg, 0.0 = only protgpt")
    argparser.add_argument("--lasso_lambda", default=0.0, type=float, help="Lambda parameter for lasso regularization, only used if align_type is 'linear'")
    argparser.add_argument("--target_protein", type=str, help="Target protein structure name for inverse folding, mandatory if dataset is 'single'")

    args = argparser.parse_args()

    assert args.dataset != 'single' or args.target_protein is not None, "If dataset is 'single', --target-protein must be specified"
    assert args.oracle_mode != 'balanced' or args.oracle_alpha is not None, "If oracle_mode is 'balanced', --oracle-alpha must be specified"

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu", args.gpu)
    logging.info(f"Seting device {device}")

    results = []
    execution_func = generate_execution_func(results,       \
                                            device,         \
                                            args.model,     \
                                            args.base_path, \
                                            repeat_num=args.batch_size, \
                                            align_type=args.align_type, \
                                            oracle_mode=args.oracle_mode, \
                                            oracle_alpha=args.oracle_alpha, \
                                            lasso_lambda=args.lasso_lambda, \
                                            N=args.align_n,)
    
    execute_on_dataset(execution_func,                  \
                    args.base_path,                     \
                    dataset_name=args.dataset,          \
                    target_protein=args.target_protein, \
                    batch_repeat=args.batch_repeat,     \
                    val_batch_size=1)
    
    output_fn = generate_output_fn(args)
    results_merge = pd.concat(results)
    
    logging.info(f"Saving to file {output_fn}")
    results_merge.to_csv(output_fn, index=False)

if __name__ == "__main__":
    main()