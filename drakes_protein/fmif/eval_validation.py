import os
import pickle
import argparse
from tqdm import tqdm
import logging
import torch

from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
from protein_oracle.model_utils import ProteinMPNNOracle
from torch.utils.data import DataLoader
from model_utils import ProteinMPNNFMIF
from fm_utils import Interpolant

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class InterpolantConfig:
    def __init__(self, min_t=1e-2, interpolant_type='masking', temp=0.1, num_timesteps=50):
        self.min_t = min_t
        self.interpolant_type = interpolant_type
        self.temp = temp
        self.num_timesteps = num_timesteps

def execute_on_validation(func, base_path, batch_repeat=1, val_batch_size=1):
    pdb_path = os.path.join(base_path, 'proteindpo_data/AlphaFold_model_PDBs')
    logging.info(f"Reading protein set ({pdb_path})")

    max_len = 75  # Define the maximum length of proteins
    dataset = ProteinStructureDataset(pdb_path, max_len) # max_len set to 75 (sequences range from 31 to 74)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    # make a dict of pdb filename: index
    pdb_idx_dict = {}
    pdb_structures = None
    for batch in loader:
        pdb_structures = batch[0]
        pdb_filenames = batch[1]
        pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
        break

    dpo_dict_path_v = os.path.join(base_path, 'proteindpo_data/processed_data')
    logging.info(f"Reading validation set ({dpo_dict_path_v})")
    dpo_valid_dict = pickle.load(open(os.path.join(dpo_dict_path_v, 'dpo_valid_dict_wt.pkl'), 'rb'))
    dpo_valid_dataset = ProteinDPODataset(dpo_valid_dict, pdb_idx_dict, pdb_structures)
    loader_valid = DataLoader(dpo_valid_dataset, batch_size=val_batch_size, shuffle=False)

    logging.info(f"Executing on validation set (BATCH REPEATS: {batch_repeat}, PROTEINS PER BATCH: {val_batch_size})")
    
    for batch in tqdm(loader_valid):
        for _ in range(batch_repeat):
            func(batch)

def build_reward_oracle(reward_model, X, mask, chain_M, residue_idx, chain_encoding_all):
    cached_batches = {}
    def ddg_oracle(samples):
        n = samples.shape[0]
        if n not in cached_batches:
            X_n = X.repeat(n, 1, 1, 1)
            mask_n = mask.repeat(n, 1)
            chain_M_n = chain_M.repeat(n, 1)
            residue_idx_n = residue_idx.repeat(n, 1)
            chain_encoding_all_n = chain_encoding_all.repeat(n, 1)
            
            cached_batches[n] = {}
            cached_batches[n]["X"] = X_n
            cached_batches[n]["mask"] = mask_n
            cached_batches[n]["chain_M"] = chain_M_n
            cached_batches[n]["residue_idx"] = residue_idx_n
            cached_batches[n]["chain_encoding_all"] = chain_encoding_all_n
        
        cache = cached_batches[n]

        with torch.no_grad():
            X_n = cache["X"]
            mask_n = cache["mask"]
            chain_M_n = cache["chain_M"]
            residue_idx_n = cache["residue_idx"]
            chain_encoding_all_n = cache["chain_encoding_all"]
            return reward_model(X_n, samples, mask_n, chain_M_n, residue_idx_n, chain_encoding_all_n)
    return ddg_oracle

def generate_validation_func(device, model, base_path, repeat_num=1, hidden_dim=128, num_encoder_layers=3, num_neighbors=30, dropout=0.1):
    assert model in ['pretrained', 'drakes'], f"Encountered model value '{model}' which is not in ['pretrained' or 'drakes']"

    logging.info(f"Generating validation evaluator (REPEATS PER PROTEIN: {repeat_num})")

    if model == 'pretrained':
        relative_ckpt_path = 'pmpnn/outputs/pretrained_if_model.pt'
    elif model == 'drakes':
        relative_ckpt_path = 'protein_rewardbp/finetuned_if_model.ckpt'
    else:
        return ValueError()

    state_dict_path = os.path.join(base_path, relative_ckpt_path)
    logging.info(f"Loading {str.upper(model)} model state dict ({state_dict_path})")

    fmif_model = ProteinMPNNFMIF(node_features=hidden_dim,
                edge_features=hidden_dim,
                hidden_dim=hidden_dim,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_encoder_layers,
                k_neighbors=num_neighbors,
                dropout=dropout)
    fmif_model.to(device)
    fmif_model.load_state_dict(torch.load(state_dict_path)['model_state_dict'])
    fmif_model.finetune_init() # TODO: Check order of this line and the above for pretrained vs drakes
    fmif_model.eval()

    logging.info(f"Setting up interpolant")
    interpolant_cfg = InterpolantConfig()
    noise_interpolant = Interpolant(interpolant_cfg)
    noise_interpolant.set_device(device)

    reward_dict_path = os.path.join(base_path, 'protein_oracle/outputs/reward_oracle_ft.pt')
    logging.info(f"Loading ddg reward model ({reward_dict_path})")

    reward_model = ProteinMPNNOracle(node_features=hidden_dim,
                        edge_features=hidden_dim,
                        hidden_dim=hidden_dim,
                        num_encoder_layers=num_encoder_layers,
                        num_decoder_layers=num_encoder_layers,
                        k_neighbors=num_neighbors,
                        dropout=dropout)
    reward_model.to(device)
    reward_model.load_state_dict(torch.load(reward_dict_path)['model_state_dict'])
    reward_model.finetune_init()
    reward_model.eval()

    reward_eval_dict_path = os.path.join(base_path, 'protein_oracle/outputs/reward_oracle_eval.pt')
    logging.info(f"Loading ddg reward evaluation model ({reward_eval_dict_path})")

    reward_model_eval = ProteinMPNNOracle(node_features=hidden_dim,
                    edge_features=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_encoder_layers,
                    k_neighbors=num_neighbors,
                    dropout=dropout)
    reward_model_eval.to(device)
    reward_model_eval.load_state_dict(torch.load(reward_eval_dict_path)['model_state_dict'])
    reward_model_eval.finetune_init()
    reward_model_eval.eval()

    # HARD CODING FUNCTION TESTING CHARACTERISTICS #
    N = 1
    align_type='bon'
    ################################################

    def validation_func(batch):
        X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
        balanced_oracle = build_reward_oracle(reward_model, X, mask, chain_M, residue_idx, chain_encoding_all)
        X = X.repeat(repeat_num, 1, 1, 1)
        mask = mask.repeat(repeat_num, 1)
        chain_M = chain_M.repeat(repeat_num, 1)
        residue_idx = residue_idx.repeat(repeat_num, 1)
        chain_encoding_all = chain_encoding_all.repeat(repeat_num, 1)
        S_sp, prot_traj, clean_traj = noise_interpolant.sample(fmif_model, \
                                                                X, \
                                                                mask, \
                                                                chain_M, \
                                                                residue_idx, \
                                                                chain_encoding_all,\
                                                                reward_model=reward_model, \
                                                                batch_oracle=balanced_oracle, \
                                                                n=N, \
                                                                steps_per_level=1, \
                                                                align_type=align_type)

    return validation_func

# Can develop directly in eval validation, using it as a script, or just calling the validation function
def main():
    logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.INFO)

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--base_path", type=str, help="Base path for data and model") 
    argparser.add_argument("--model", choices=['pretrained', 'drakes'], help="Model must be one of ['pretrained', 'drakes']")
    argparser.add_argument("--batch_repeat", type=int, default=1, help="Number of times to repeat execution of each validation protein batch")
    argparser.add_argument("--batch_size", type=int, default=1, help="Number of times to generate each protein in a batch")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device to be used in validation script")

    args = argparser.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu", args.gpu)
    logging.info(f"Seting device {device}")

    validation_func = generate_validation_func(device, args.model, args.base_path, repeat_num=args.batch_size)
    execute_on_validation(validation_func,          \
                        args.base_path,             \
                        batch_repeat=args.batch_repeat, \
                        val_batch_size=1)

if __name__ == "__main__":
    main()