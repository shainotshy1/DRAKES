import os
import torch
import logging
import pandas as pd

from protein_oracle.data_utils import featurize
from protein_oracle.model_utils import ProteinMPNNOracle
from protein_oracle.data_utils import ALPHABET

from align_loglikelihood_oracle import build_protgpt_oracle

from model_utils import ProteinMPNNFMIF
from fm_utils import Interpolant

def gen_results(S_sp, S, batch, mask_for_loss):
    with torch.no_grad():
        results_list = []
        true_detok_seq = "".join([ALPHABET[x] for _ix, x in enumerate(S[0]) if mask_for_loss[0][_ix] == 1])
        
        for _it, ssp in enumerate(S_sp):
            seq_revovery = (S_sp[_it] == S[0]).float().mean().item()
            resultdf = pd.DataFrame(columns=['seq_recovery'])
            resultdf.loc[0] = [seq_revovery]
            resultdf['seq'] = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            resultdf['true_seq'] = true_detok_seq
            resultdf['protein_name'] = batch['protein_name'][0]
            results_list.append(resultdf)

    return results_list

class InterpolantConfig:
    def __init__(self, min_t=1e-2, interpolant_type='masking', temp=0.1, num_timesteps=50):
        self.min_t = min_t
        self.interpolant_type = interpolant_type
        self.temp = temp
        self.num_timesteps = num_timesteps

def build_reward_oracle(reward_model, device, X, mask, chain_M, residue_idx, chain_encoding_all, mode="ddg", alpha=1.0):
    cached_batches = {}

    valid_modes = ["ddg", "protgpt", "balanced"]
    assert mode in valid_modes, f"Invalid mode: {mode} (Choose from {valid_modes})"
    assert mode != "balanced" or (type(alpha) is float and 0 <= alpha <= 1), "If mode is 'balanced', alpha must be a float between 0 and 1"

    if mode == "balanced":
        if alpha == 1.0:
            mode = "ddg"
        elif alpha == 0.0:
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

    if mode == 'ddg':
        return ddg_oracle
    else:
        protgpt_oracle = build_protgpt_oracle(device)

    if mode == 'protgpt':
        return protgpt_oracle
    elif mode == 'balanced':
        protgpt_scaling = 0.005 # This scaling is to make choosing the alpha value more linear, it is okay that it is hard coded as we can choose alpha to compensate
        def balanced_reward(samples):
            ddg_rewards = ddg_oracle(samples)
            prot_gpt_rewards = protgpt_oracle(samples)
            return alpha * ddg_rewards + (1 - alpha) * prot_gpt_rewards * protgpt_scaling
        return balanced_reward
    else:
        raise ValueError()

def generate_execution_func(out_lst, 
                            device, 
                            model, 
                            base_path, 
                            align_type='bon', 
                            oracle_mode='ddg', 
                            oracle_alpha=1.0,
                            N=1, 
                            beam_w=1,
                            steps_per_level=1,
                            spec_feedback_its=0,
                            lasso_lambda=0.0, 
                            repeat_num=1, 
                            hidden_dim=128, 
                            num_encoder_layers=3, 
                            num_neighbors=30, 
                            dropout=0.1):
    assert model in ['pretrained', 'drakes'], f"Encountered model value '{model}' which is not in ['pretrained' or 'drakes']"
    assert align_type in ['bon', 'beam'], f"Encountered align_type value '{align_type}' which is not in ['bon', 'beam']"
    assert type(N) is int and N > 0
    assert type(lasso_lambda) is float
    assert oracle_mode in ['ddg', 'protgpt', 'balanced']
    assert oracle_mode != 'balanced' or type(oracle_alpha) is float and 0 <= oracle_alpha <= 1

    logging.info(f"Generating dataset evaluator (Repeats per protein: {repeat_num})")

    if model == 'pretrained':
        relative_ckpt_path = 'pmpnn/outputs/pretrained_if_model.pt'
    elif model == 'drakes':
        relative_ckpt_path = 'protein_rewardbp/finetuned_if_model.ckpt'
    else:
        raise ValueError()
    
    assert type(out_lst) is list and len(out_lst) == 0, "VALUE ERROR: out_lst MUST be an empty python 'list'"

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
    if model == 'pretrained': fmif_model.load_state_dict(torch.load(state_dict_path)['model_state_dict'])
    else: fmif_model.load_state_dict(torch.load(state_dict_path))
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

    func_descr = f"align_type={align_type}, oracle_mode={oracle_mode}, N={N}, spec_feedback={spec_feedback_its}"
    if align_type == 'beam':
        func_descr += f", W={beam_w}"
    if align_type in ['beam']:
        func_descr += f", steps_per_level={steps_per_level}"
    if oracle_mode == 'balanced':
        func_descr += f", balanced_alpha={oracle_alpha}"

    logging.info(f"Setup execution function ({func_descr})")
    def validation_func(batch):
        X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
        balanced_oracle = build_reward_oracle(reward_model, device, X, mask, chain_M, residue_idx, chain_encoding_all, mode=oracle_mode, alpha=oracle_alpha)
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
                                                                batch_oracle=balanced_oracle, \
                                                                n=N, \
                                                                beam_w=beam_w, \
                                                                steps_per_level=steps_per_level, \
                                                                align_type=align_type,
                                                                lasso_lambda=lasso_lambda,
                                                                spec_feedback_its=spec_feedback_its)
        
        mask_for_loss = mask*chain_M
        results_list = gen_results(S_sp, S, batch, mask_for_loss)
        out_lst.extend(results_list)

    return validation_func