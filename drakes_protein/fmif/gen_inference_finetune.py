import os
import pickle
import argparse
from tqdm import tqdm
import logging
import torch
import pandas as pd
import numpy as np

from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset
from protein_oracle.utils import set_seed
from torch.utils.data import DataLoader, Subset

from execute_align_utils import generate_execution_func # type: ignore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_drakes_data(base_path, dataset_name: str, val_batch_size: int, batch_repeat: int, target_protein = None, num_workers=1, worker_id=0):
    assert dataset_name != "single" or target_protein is not None, "If dataset is 'single', target_protein must be specified"
    assert type(num_workers) is int and type(worker_id) is int, "num_workers and worker_id must be type 'int'"
    assert num_workers > 0, "num_workers must be a positive integer"
    assert 0 <= worker_id < num_workers, "worker_id must follow 0-indexing so it should satisfy 0 <= worker_id < num_workers"

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

    num_batch = len(dpo_dataset)
    B = (num_batch + num_workers - 1) // num_workers
    indices = np.arange(B * worker_id, min(B * (worker_id+1), num_batch), 1)
    target_subset = Subset(dpo_dataset, indices)
        
    if dataset == "single":
        loader_valid = DataLoader(target_subset, batch_size=1, shuffle=False)
        logging.info(f"Executing on protein {target_protein} (Batch repeats: {batch_repeat}, Proteins per batch: {val_batch_size})")
    else:
        loader_valid = DataLoader(target_subset, batch_size=val_batch_size, shuffle=False)
        logging.info(f"Executing on {dataset_name} set (Batch repeats: {batch_repeat}, Proteins per batch: {val_batch_size}, Total proteins: {len(dpo_dataset)})")
    
    logging.info(f"WorkerID: {worker_id}, Assigned proteins: {len(target_subset)}")

    return loader_valid

def execute_on_dataset(func, base_path, dataset_name="validation", target_protein=None, batch_repeat=1, val_batch_size=1, num_workers=1, worker_id=0):
    assert dataset_name in ['validation', 'test', 'train', 'single'], f"Encountered dataset value '{dataset_name}' which is not in ['validation', 'test', 'single']"

    loader_valid = get_drakes_data(base_path, dataset_name, val_batch_size, batch_repeat, target_protein=target_protein, num_workers=num_workers, worker_id=worker_id)

    count = 0
    for batch in loader_valid:
        for _ in tqdm(range(batch_repeat)):
            func(batch)
            count += 1
            print(f"Progress: {count} / {len(loader_valid) * batch_repeat}")

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
    if args.spec_feedback_its > 0:
        out_name += f"_feedbacksteps={args.spec_feedback_its}"
        out_name += f"_feedbackmethod={args.feedback_method}"
        out_name += f"_maxspecorder={args.max_spec_order}"
        if args.feedback_method == "lasso" or args.feedback_method == "spectral" or args.feedback_method == "max-mask":
            out_name += f"_masks={args.num_spec_masks}_rmax={args.reward_batch_max}"
        if args.feedback_method == "hill-climb":
            out_name += f"_hcits={args.hill_climb_iterations}_rmax={args.reward_batch_max}"
        if args.feedback_method == "lasso":
            out_name += f"_lassolambda={args.lasso_lambda}"
        if args.feedback_method == "spectral":
            quotes = '""'
            out_name += f"_{args.gbt_args.replace(' ', '').replace(':', '=').replace(',', '_').replace('}','').replace('{','').replace(quotes, '')}"

    if args.MH_steps > 0:
        if args.MH_type == 'uniform':
            out_name += f"_{args.MH_type}_mhn={args.MH_steps}_p={args.MH_p}_beta={args.MH_b}"
        else:
            out_name += f"_{args.MH_type}_mhn={args.MH_steps}_beta={args.MH_b}"

    if args.align_type == "beam":
        out_name += f"_W={args.beam_w}"
    if args.align_type != "bon":
        out_name += f"_stepsperlevel={args.steps_per_level}"

    if args.num_workers > 1:
        out_name += f"_w{args.worker_id}"

    full_out_path = f"{args.output_folder}/{out_name}.csv"
    return full_out_path

def main():
    logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.INFO)

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    argparser.add_argument("--base_path", required=True, type=str, help="Base path for data and model") 
    argparser.add_argument("--model", required=True, choices=['pretrained', 'drakes'], help="Model must be one of ['pretrained', 'drakes']")
    argparser.add_argument("--dataset", required=True, choices=['validation', 'test', 'train', 'single'], help="Dataset must be on of ['validation', 'test', 'train']")
    argparser.add_argument("--output_folder", type=str, required=True, help="Output folder for protein generations")
    argparser.add_argument("--align_type", choices=['bon', 'beam'], required=True)
    argparser.add_argument("--oracle_mode", choices=['ddg', 'protgpt', 'scrmsd'], required=True)
    argparser.add_argument("--align_n", type=int, required=True, help="Number of samples parameter for alignment techniques")
    argparser.add_argument("--num_workers", type=int, required=True, help="Number of workers assigned to the inference task")
    argparser.add_argument("--worker_id", type=int, required=True, help="ID of the current worker")

    # Optional arguments
    argparser.add_argument("--batch_repeat", type=int, default=1, help="Number of times to repeat execution of each protein batch")
    argparser.add_argument("--batch_size", type=int, default=1, help="Number of times to generate each protein in a batch")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device to be used in execution script")
    argparser.add_argument("--oracle_alpha", type=float, help="Alpha parameter for balanced oracle mode, 1.0 = only ddg, 0.0 = only protgpt")
    argparser.add_argument("--lasso_lambda", default=0.0, type=float, help="Lambda parameter for lasso regularization, only used if align_type is 'linear'")
    argparser.add_argument("--target_protein", type=str, help="Target protein structure name for inverse folding, mandatory if dataset is 'single'")
    argparser.add_argument("--steps_per_level", type=int, default=1, help="Number of diffusion steps per alignment step (only applies for BEAM)")
    argparser.add_argument("--beam_w", type=int, default=1, help="Number of beams for BEAM sampling (only applies if align_type is 'beam')")
    argparser.add_argument("--spec_feedback_its", type=int, required=False, default=0)
    argparser.add_argument("--max_spec_order", type=int, required=False, default=10)
    argparser.add_argument("--feedback_method", type=str, required=False, default="spectral")
    argparser.add_argument("--reward_batch_max", action="store_true", default=False, help="Whether to take the max or average of the reward batch when computing feedback (applies to 'spectral', 'lasso', and 'hill-climb')")
    argparser.add_argument("--spex_analysis", action="store_true", default=False, help="Whether to perform spectral analysis")
    argparser.add_argument("--num_spec_masks", type=int, required=False, default=512)
    argparser.add_argument("--hill_climb_iterations", type=int, required=False, default=512, help="Hill-climb proposal steps per feedback iteration (feedback_method='hill-climb')")
    argparser.add_argument("--MH_steps", type=int, required=False, default=0)
    argparser.add_argument("--MH_p",type=float, required=False, default=0.5)
    argparser.add_argument("--MH_b", type=float, required=False, default=1.0)
    argparser.add_argument("--MH_type", type=str, required=False, default="uniform")
    argparser.add_argument("--seed", type=int, required=False, default=0)
    argparser.add_argument("--gbt_args", type=str, required=False, default="")

    args = argparser.parse_args()

    assert args.dataset != 'single' or args.target_protein is not None, "If dataset is 'single', --target-protein must be specified"
    assert args.oracle_mode != 'balanced' or args.oracle_alpha is not None, "If oracle_mode is 'balanced', --oracle-alpha must be specified"

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu", args.gpu)
    logging.info(f"Seting device {device}")

    if args.oracle_mode == 'scrmsd':
        import pyrosetta
        pyrosetta.init(extra_options="-out:level 100")

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
                                            N=args.align_n, \
                                            beam_w=args.beam_w, \
                                            steps_per_level=args.steps_per_level, \
                                            spec_feedback_its=args.spec_feedback_its, \
                                            max_spec_order=args.max_spec_order, \
                                            feedback_method=args.feedback_method, \
                                            reward_batch_max=args.reward_batch_max, \
                                            mh_n=args.MH_steps, \
                                            mh_p=args.MH_p, \
                                            mh_b=args.MH_b,
                                            mh_type=args.MH_type,
                                            num_spec_masks=args.num_spec_masks,
                                            seed=args.seed,
                                            gbt_args=args.gbt_args,
                                            spex_analysis=args.spex_analysis,
                                            hill_climb_iterations=args.hill_climb_iterations)
    
    execute_on_dataset(execution_func,                  \
                    args.base_path,                     \
                    dataset_name=args.dataset,          \
                    target_protein=args.target_protein, \
                    batch_repeat=args.batch_repeat,     \
                    val_batch_size=1,                   \
                    num_workers=args.num_workers,       \
                    worker_id=args.worker_id)           \
    
    output_fn = generate_output_fn(args)
    results_merge = pd.concat(results)
    
    logging.info(f"Saving to file {output_fn}")
    results_merge.to_csv(output_fn, index=False)

if __name__ == "__main__":
    main()