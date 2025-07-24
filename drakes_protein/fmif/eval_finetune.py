import argparse
import os.path
import pickle
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
from torch.utils.data import DataLoader
import os.path
from protein_oracle.utils import set_seed
from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
from protein_oracle.model_utils import ProteinMPNNOracle
#from fmif.model_utils import ProteinMPNNFMIF
from model_utils import ProteinMPNNFMIF
#from fmif.fm_utils import Interpolant
from fm_utils import Interpolant
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

def gen_results(S_sp, S, batch, mask_for_loss, save_path, args, item_idx, base_path):
    with torch.no_grad():
        results_list = []
        run_name = save_path.split('/')
        if run_name[-1] == '':
            run_name = run_name[-2]
        else:
            run_name = run_name[-1]
        true_detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(S[0]) if mask_for_loss[0][_ix] == 1])
        
        print(f"     - Memory: {torch.cuda.mem_get_info()}")
        for _it, ssp in enumerate(S_sp):
            num = item_idx * 16 + _it
            seq_revovery = (S_sp[_it] == S[0]).float().mean().item()
            resultdf = pd.DataFrame(columns=['seq_recovery'])
            resultdf.loc[0] = [seq_revovery]
            resultdf['seq'] = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            resultdf['true_seq'] = true_detok_seq
            resultdf['protein_name'] = batch['protein_name'][0]
            resultdf['WT_name'] = batch['WT_name'][0]
            resultdf['num'] = num
            results_list.append(resultdf)

    return results_list

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--base_path", type=str, default="/data/scratch/wangchy/seqft/", help="base path for data and model") 
argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for") # 200
argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
argparser.add_argument("--batch_size", type=int, default=32, help="number of sequences for one batch") # 128
argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")   # 48
argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout") # TODO
argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")   # TODO
argparser.add_argument("--gradient_norm", type=float, default=1.0, help="clip gradient norm, set to negative to omit clipping")
argparser.add_argument("--mixed_precision", type=str2bool, default=True, help="train with mixed precision")
argparser.add_argument("--wandb_name", type=str, default="debug", help="wandb run name")
argparser.add_argument("--lr", type=float, default=1e-4)
argparser.add_argument("--wd", type=float, default=1e-4)

argparser.add_argument("--min_t", type=float, default=1e-2)
argparser.add_argument("--schedule", type=str, default='linear') # other schedule is not implemented
argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
argparser.add_argument("--temp", type=float, default=0.1)
argparser.add_argument("--noise", type=float, default=1.0) # 20.0
argparser.add_argument("--interpolant_type", type=str, default='masking')
argparser.add_argument("--num_timesteps", type=int, default=50) # 500
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--eval_every_n_epochs", type=int, default=1)
argparser.add_argument("--num_samples_per_eval", type=int, default=10)

argparser.add_argument("--accum_steps", type=int, default=1)
argparser.add_argument("--truncate_steps", type=int, default=10)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument("--alpha", type=float, default=0.001)
argparser.add_argument("--gumbel_softmax_temp", type=float, default=0.5)

argparser.add_argument("--decoding", type=str, default='original')
argparser.add_argument("--dps_scale", type=float, default=10)
argparser.add_argument("--tds_alpha", type=float, default=0.5)
argparser.add_argument("--base_model", type=str, default='old')

argparser.add_argument("--align_type", type=str, default='bon')
argparser.add_argument("--n_align", type=int, default=1)
argparser.add_argument("--align_oracle", type=str, default="ddg")
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument("--balanced_alpha", type=float, default=0.01, help="alpha scales the protgpt reward added to the ddg reward")

args = argparser.parse_args()
pdb_path = os.path.join(args.base_path, 'proteindpo_data/AlphaFold_model_PDBs')
max_len = 75  # Define the maximum length of proteins
dataset = ProteinStructureDataset(pdb_path, max_len) # max_len set to 75 (sequences range from 31 to 74)
loader = DataLoader(dataset, batch_size=1000, shuffle=False)

# make a dict of pdb filename: index
for batch in loader:
    pdb_structures = batch[0]
    pdb_filenames = batch[1]
    pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
    break

dpo_dict_path = os.path.join(args.base_path, 'proteindpo_data/processed_data')
dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
dpo_test_dataset = ProteinDPODataset(dpo_test_dict, pdb_idx_dict, pdb_structures)
loader_test = DataLoader(dpo_test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu", args.gpu)
print(f"DEVICE: {device}")
old_fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                    edge_features=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_encoder_layers,
                    k_neighbors=args.num_neighbors,
                    dropout=args.dropout,
                    )
old_fmif_model.to(device)
old_fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'pmpnn/outputs/pretrained_if_model.pt'))['model_state_dict'])
old_fmif_model.finetune_init()

if args.base_model == 'new':
    new_fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
    new_fmif_model.to(device)
    new_fmif_model.finetune_init()
    new_fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_rewardbp/finetuned_if_model.ckpt')))
    #new_fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_rewardbp/finetuned_if_model.ckpt'))['model_state_dict'])
    model_to_test_list = [new_fmif_model]
elif args.base_model == 'old':
    model_to_test_list = [old_fmif_model]
elif args.base_model == 'zero_alpha':
    zero_alpha_fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
    zero_alpha_fmif_model.to(device)
    zero_alpha_fmif_model.finetune_init()
    zero_alpha_fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_rewardbp/zeroalpha_if_model.ckpt'))['model_state_dict'])
    model_to_test_list = [zero_alpha_fmif_model]

reward_model = ProteinMPNNOracle(node_features=args.hidden_dim,
                    edge_features=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_encoder_layers,
                    k_neighbors=args.num_neighbors,
                    dropout=args.dropout,
                    )
reward_model.to(device)
reward_model.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_oracle/outputs/reward_oracle_ft.pt'))['model_state_dict'])
reward_model.finetune_init()
reward_model.eval()

reward_model_eval = ProteinMPNNOracle(node_features=args.hidden_dim,
                    edge_features=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_encoder_layers,
                    k_neighbors=args.num_neighbors,
                    dropout=args.dropout,
                    )
reward_model_eval.to(device)
reward_model_eval.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_oracle/outputs/reward_oracle_eval.pt'))['model_state_dict'])
reward_model_eval.finetune_init()
reward_model_eval.eval()

path_for_outputs = os.path.join(args.base_path, 'protein_rewardbp')
save_path = os.path.join(path_for_outputs, 'eval')

noise_interpolant = Interpolant(args)
noise_interpolant.set_device(device)

set_seed(args.seed, use_cuda=True)

from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2')
model = GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2').to(device)

def protgpt_oracle(samples):
    rewards = []
    for seq in samples:
        seq_str = "".join([ALPHABET[x] for x in seq])
        out = tokenizer(seq_str, return_tensors="pt")
        input_ids = out.input_ids.cuda().to(seq.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        log_likelihood = -1 * (outputs.loss * input_ids.shape[1]).item()
    
        rewards.append(log_likelihood)
    return torch.tensor(rewards, device=seq.device)

def build_reward_oracle(reward_model, X, mask, chain_M, residue_idx, chain_encoding_all, mode="ddg", alpha=1):
    cached_batches = {}
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1 as it is the weight for the tradeoff between two oracles"

    if mode == "balanced":
        if alpha == 1:
            mode = "ddg"
        elif alpha == 0:
            mode = "protgpt"

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
        
    valid_modes = ["ddg", "protgpt", "balanced"]
    assert mode in valid_modes, f"Invalid mode: {mode} (Choose from {valid_modes})"
    protgpt_scaling = 0.005 # This scaling is to make choosing the alpha value more linear, it is okay that it is hard coded as we can choose alpha to compensate
    def balanced_reward(samples):
        if mode == "ddg":
            ddg_rewards = ddg_oracle(samples)
            return ddg_rewards
        elif mode == "protgpt":
            prot_gpt_rewards = protgpt_oracle(samples)
            return prot_gpt_rewards
        elif mode == "balanced":
            ddg_rewards = ddg_oracle(samples)
            prot_gpt_rewards = protgpt_oracle(samples)
            return alpha * ddg_rewards + (1 - alpha) * prot_gpt_rewards * protgpt_scaling
    return balanced_reward

save_trajectory = False

for n in [args.n_align]:
    for align_step_interval in [1]:
        for testing_model in model_to_test_list:
            balanced_alpha_suffix = f"_{args.balanced_alpha}" if args.align_oracle == "balanced" else ""
            balanced_alpha_str = f" Balanced Alpha: {args.balanced_alpha}" if args.align_oracle == "balanced" else ""
            test_name = f"{args.base_model}_7JJK_{args.align_type}_{args.align_oracle}_{n}_{align_step_interval}" + balanced_alpha_suffix
            print(f'Testing Model ({args.align_type}-{args.align_oracle}-N: {n}, Interval: {align_step_interval}{balanced_alpha_str})... Sampling {args.decoding} - {args.base_model}')
            testing_model.eval()
            repeat_num=16
            valid_sp_acc, valid_sp_weights = 0., 0.
            results_merge = []
            all_model_logl = []
            rewards_eval = []
            rewards = []
            mask_proportion = []
            ddg_eval_average = []
            ddg_train_average = []
            protgpt_average = []
            reward_average = []
            total_batch_count = 0
            num = 1
            for batch in loader_test:
                if batch['protein_name'][0] != '7JJK.pdb':
                    continue
                for item_idx in range(8):
                    print(f"    [{num}] {batch['protein_name'][0]}")
                    num += 1
                    X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
                    balanced_oracle = build_reward_oracle(reward_model, X, mask, chain_M, residue_idx, chain_encoding_all, mode=args.align_oracle, alpha=args.balanced_alpha)
                    X = X.repeat(repeat_num, 1, 1, 1)
                    mask = mask.repeat(repeat_num, 1)
                    chain_M = chain_M.repeat(repeat_num, 1)
                    residue_idx = residue_idx.repeat(repeat_num, 1)
                    chain_encoding_all = chain_encoding_all.repeat(repeat_num, 1)

                    if args.decoding == 'cg':
                        S_sp, _, _ = noise_interpolant.sample_controlled_CG(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                            guidance_scale=args.dps_scale, reward_model=reward_model)
                    elif args.decoding == 'smc':
                        S_sp, _, _ = noise_interpolant.sample_controlled_SMC(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                            reward_model=reward_model, alpha=args.tds_alpha)
                    elif args.decoding == 'tds': 
                        S_sp, _, _ = noise_interpolant.sample_controlled_TDS(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                            reward_model=reward_model, alpha=args.tds_alpha, guidance_scale=args.dps_scale) 
                    elif args.decoding == 'original':
                        mask_for_loss = mask*chain_M
                        S_sp, prot_traj, clean_traj = noise_interpolant.sample(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all, batch_oracle=balanced_oracle, n=n, steps_per_level=align_step_interval, align_type=args.align_type)
                        for i, S_sp_traj in enumerate(prot_traj):
                            if i < len(clean_traj):
                                dg_pred_eval = reward_model_eval(X, clean_traj[i].to(device), mask, chain_M, residue_idx, chain_encoding_all)
                                dg_pred_eval = dg_pred_eval.detach().cpu().numpy()
                                dg_pred_train = reward_model(X, clean_traj[i].to(device), mask, chain_M, residue_idx, chain_encoding_all)
                                dg_pred_train = dg_pred_train.detach().cpu().numpy()
                                if len(reward_average) == 0:
                                    reward_average = [0] * len(clean_traj)
                                    ddg_eval_average = [0] * len(clean_traj)
                                    ddg_train_average = [0] * len(clean_traj)
                                    protgpt_average = [0] * len(clean_traj)
                            reward_average[i] += balanced_oracle(clean_traj[i].to(device)).cpu().numpy().mean()
                            ddg_eval_average[i] += dg_pred_eval.mean()
                            ddg_train_average[i] += dg_pred_train.mean()
                            protgpt_average[i] += protgpt_oracle(clean_traj[i]).cpu().numpy().mean()
                            total_prop = 0
                            if len(mask_proportion) == 0:
                                mask_proportion = [0] * len(prot_traj)
                            for _it, ssp in enumerate(S_sp_traj):
                                mask_detect = [(x >= len(ALPHABET)).item() for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1]
                                total_prop += sum(mask_detect) / len(mask_detect)
                            mask_proportion[i] += total_prop / len(S_sp_traj)
                        total_batch_count += 1
                    true_false_sp = (S_sp == S).float()
                    mask_for_loss = mask*chain_M
                    valid_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
                    valid_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                    results_list = gen_results(S_sp, S, batch, mask_for_loss, save_path, args, item_idx, args.base_path)
                    results_merge.extend(results_list)
                #     break
                # break
            mask_proportion = [x / total_batch_count for x in mask_proportion]
            reward_average = [x / total_batch_count for x in reward_average]
            ddg_train_average = [x / total_batch_count for x in ddg_train_average]
            ddg_eval_average = [x / total_batch_count for x in ddg_eval_average]
            protgpt_average = [x / total_batch_count for x in protgpt_average]
            range_column = list(range(1, len(mask_proportion) + 1))
            if save_trajectory:
                data = zip(range_column, mask_proportion, reward_average, ddg_train_average, ddg_eval_average, protgpt_average)
                with open(f'trajectory_{test_name}.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Iteration', 'Mask Proportion', 'Reward Average', 'DDG Train Average', 'DDG Eval Average', 'ProtGPT Average'])
                    writer.writerows(data)
            valid_sp_accuracy = valid_sp_acc / valid_sp_weights
            print(f"Recovery Accuracy: {round(valid_sp_accuracy, 3)} Reward Avg: {round(reward_average[-1], 3)}, DDG Train Avg: {round(ddg_train_average[-1], 3)} DDG Eval Avg: {round(ddg_eval_average[-1], 3)} ProtGPT Avg: {round(protgpt_average[-1], 3)}")

            results_merge = pd.concat(results_merge)
            results_merge.to_csv(f'./eval_results/sequence_{test_name}.csv', index=False)
