import os
import torch
import numpy as np 
import pandas as pd
import argparse
from tqdm import tqdm
from protein_oracle.model_utils import ProteinMPNNOracle
from protein_oracle.data_utils import featurize
from protein_oracle.data_utils import ALPHABET
from utils import process_seq_data_directory, get_drakes_test_data # type: ignore
import warnings
warnings.filterwarnings("ignore", message=".*use_reentrant.*")

def extract_features(protein_name, protein_batches, cached_features):
    if protein_name not in cached_features:
        batch = protein_batches[protein_name]
        cached_features[protein_name] = featurize(batch, device)
    return cached_features[protein_name]

def pred_ddg_wrapper(seq, model, device, protein_name, protein_batches, cached_features):
    S_sp = torch.tensor([ALPHABET.index(c) for c in seq], device=device).reshape(1, -1)
    X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = extract_features(protein_name, protein_batches, cached_features)
    dg_pred = model(X, S_sp, mask, chain_M, residue_idx, chain_encoding_all)
    return dg_pred.detach().cpu().numpy()

def extract_ddg_distr(df, seq_label, true_seq_label, protein_label, model, protein_batches):
    assert seq_label in df.columns, f"'{seq_label}' must be a label in the data frame"
    sequences = df[seq_label]
    true_sequences = df[true_seq_label]
    true_protein_fns = df[protein_label]
    values = []
    assert sequences.shape == true_sequences.shape == true_protein_fns.shape, "Must have same number of sequences and true sequences"
    cached_ddg = {}
    cached_features = {}
    for i in tqdm(range(sequences.size)):
        seq = sequences[i]
        true_sequence = true_sequences[i]
        true_protein_fn = true_protein_fns[i]
        if seq == true_sequence:
            values.append(0)
        elif (seq, true_sequence) in cached_ddg:
            values.append(cached_ddg[(seq, true_sequence)])
        else:
            pred_ddg = pred_ddg_wrapper(seq, model, device, true_protein_fn, protein_batches, cached_features)
            values.append(pred_ddg)    
    values = np.array(values)
    return values

def extract_ddg_directory(dir_name, seq_label, true_seq_label, protein_label, device):
    hidden_dim = 128
    num_encoder_layers = 3
    num_neighbors = 30
    dropout = 0.1
    base_path, _, loader_test = get_drakes_test_data()
    
    reward_model = ProteinMPNNOracle(node_features=hidden_dim,
                        edge_features=hidden_dim,
                        hidden_dim=hidden_dim,
                        num_encoder_layers=num_encoder_layers,
                        num_decoder_layers=num_encoder_layers,
                        k_neighbors=num_neighbors,
                        dropout=dropout)
    reward_model.to(device)
    reward_model.load_state_dict(torch.load(os.path.join(base_path, 'protein_oracle/outputs/reward_oracle_ft.pt'))['model_state_dict'])
    reward_model.finetune_init()
    reward_model.eval()

    reward_model_eval = ProteinMPNNOracle(node_features=hidden_dim,
                        edge_features=hidden_dim,
                        hidden_dim=hidden_dim,
                        num_encoder_layers=num_encoder_layers,
                        num_decoder_layers=num_encoder_layers,
                        k_neighbors=num_neighbors,
                        dropout=dropout)
    reward_model_eval.to(device)
    reward_model_eval.load_state_dict(torch.load(os.path.join(base_path, 'protein_oracle/outputs/reward_oracle_eval.pt'))['model_state_dict'])
    reward_model_eval.finetune_init()
    reward_model_eval.eval()
    
    protein_batches = {}
    for batch in loader_test:
        protein_batches[batch['protein_name'][0]] = batch

    print("---Running ddg Training Predictions---")
    process_seq_data_directory(dir_name, 'ddg', lambda df : extract_ddg_distr(df, seq_label, true_seq_label, protein_label, reward_model, protein_batches))
    print("---Running ddg Evaluation Predictions---")
    process_seq_data_directory(dir_name, 'ddg_eval', lambda df : extract_ddg_distr(df, seq_label, true_seq_label, protein_label, reward_model_eval, protein_batches))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Log Likelihood Computation")
    parser.add_argument("dir", help="Directory containing distribution csv files")
    parser.add_argument("--gpu", help="GPU device index", default=0)
    args = parser.parse_args()

    gpu_idx = int(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', gpu_idx)

    extract_ddg_directory(args.dir, 'seq', 'true_seq', 'protein_name', device)