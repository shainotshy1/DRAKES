from dataclasses import dataclass
from typing import Optional
import pandas as pd
import os

from utils import get_eval_stats, analyze_protein_gen_helper, analyze_protein_gen_helper_violin
import seaborn as sn
import importlib

@dataclass
class BLISSExperiment:
    name: str
    base_path: str
    model: Optional[str] = None
    dataset: Optional[str] = None
    align_type: Optional[str] = None
    align_n: Optional[int] = None
    oracle_mode: Optional[str] = None
    oracle_alpha: Optional[float] = None
    lasso_lambda: Optional[float] = None
    beam_w: Optional[int] = None
    steps_per_level: Optional[int] = None
    target_protein: Optional[str] = None
    specfeedback: Optional[int] = None

    def get_test_name(self) -> str:
        out_name = f"{self.model}" 
        if self.dataset == "single":
            out_name += f"_{self.target_protein}"
        else:
            out_name += f"_{self.dataset}"
        if self.oracle_mode:
            out_name += f"_{self.oracle_mode}"
        if self.oracle_mode == "balanced":
            out_name += f"_alpha={self.oracle_alpha}"
        if self.align_type and self.align_n:
            out_name += f"_{self.align_type}_N={self.align_n}_specfeedback={self.specfeedback}"
        if self.align_type == "linear":
            out_name += f"_lambda={self.lasso_lambda}"
        elif self.align_type == "beam":
            out_name += f"_W={self.beam_w}"
        if self.align_type != "bon" and self.align_type is not None:
            if self.steps_per_level:
                out_name += f"_stepsperlevel={self.steps_per_level}"
            else:
                out_name += f"_stepsperlevel=1" # default assumes stepsperlevel value to 1
        return out_name
    
    def get_df(self) -> pd.DataFrame:
        data_path = f"{self.base_path}{self.get_test_name()}.csv"
        return pd.read_csv(data_path)
    
    def get_stats(self) -> dict:
        df = self.get_df()
        return get_eval_stats(df)

# Based off order of parsing: model, dataset, oracle_mode, [oracle_alpha], align_type, align_n, [lasso_lambda]
def collect_experiments(n, oracle, dataset, model, target_protein, target_alg) -> list[BLISSExperiment]:
    bliss_dir = '/home/shai/BLISS_Experiments/DRAKES/'
    exp_dir = 'DRAKES/drakes_protein/fmif/eval_results/'
    base_path = bliss_dir + exp_dir + dataset + '/'
    assert dataset in ['test', 'validation', 'train'] # TODO: support 'single' parsing
    assert model in ['all', 'pretrained', 'drakes']
    assert oracle in ['ddg', 'protgpt', 'scrmsd']
    assert target_alg is None or target_alg in ['bon', 'beam', 'linear', 'spectral']
    assert type(n) == int

    # Collect experiment names 
    target_dir = bliss_dir + exp_dir + dataset + '/'
    all_experiments_fn = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    valid_experiments = []
    for f in all_experiments_fn:
        f = f[:-4]  # Remove .csv
        exp_name = f
        exp = BLISSExperiment(exp_name, base_path) # type: ignore
        components = f.split('_')

        if model != 'all' and components[0] != model: continue
        exp.model = components[0]

        exp.dataset = components[1]
        if exp.dataset != dataset:
            exp.target_protein = components[1]
            exp.dataset = 'single'
            if exp.target_protein != target_protein: continue

        idx_offset = 0
        if len(components) > 3: # Process inference alignment experiments
            exp.oracle_mode = components[2 + idx_offset]
            if exp.oracle_mode != oracle: continue
            if exp.oracle_mode == 'balanced':
                exp.oracle_alpha = float(components[3 + idx_offset][6:]) # Removing prefix 'alpha='
                idx_offset += 1
            
            exp.align_type = components[3 + idx_offset]
            if target_alg is not None and target_alg != exp.align_type: continue
            exp.align_n = int(components[4 + idx_offset][2:]) # Removing prefix 'N='
            if n > 0 and exp.align_n != n: continue
            exp.specfeedback = int(components[5 + idx_offset][13:]) # Removing prefix 'specfeedback='

            if exp.align_type == 'linear':
                exp.lasso_lambda = float(components[6 + idx_offset][7:]) # Removing prefix 'lambda='
                idx_offset += 1
            elif exp.align_type == 'beam':
                exp.beam_w = int(components[6 + idx_offset][2:]) # Removing prefix 'W='
                idx_offset += 1

            if exp.align_type != 'bon':
                exp.steps_per_level = int(components[6 + idx_offset][14:]) # Removing prefix 'stepsperlevel='
                idx_offset += 1

        valid_experiments.append(exp)
        print(exp.name, exp.get_stats())

    valid_experiments = sorted(valid_experiments, key=lambda e: e.name)
        
    return valid_experiments

def display_experiments_helper(experiments, target_protein=None):
    data = [exp.get_df() for exp in experiments]
    labels = [exp.name for exp in experiments]
    colors = sn.color_palette("Set2", len(experiments))
    protein_output = target_protein + " " if target_protein is not None else ""

    analyze_protein_gen_helper_violin(target_protein, data, labels, colors, 'ddg_eval', y_label='Predicted ΔΔG', legend_pos='right', title=f'{protein_output}ΔΔG (Eval)')
    analyze_protein_gen_helper_violin(target_protein, data, labels, colors, 'loglikelihood', y_label='Log Likelihood', legend_pos='right', title=f'{protein_output}Log Likelihood Evaluation')
    analyze_protein_gen_helper_violin(target_protein, data, labels, colors, 'scrmsd', y_label='scRMSD', legend_pos='right', title=f'{protein_output}scRMSD Evaluation')

def display_experiments(oracle='ddg', dataset='test', model='all', target_protein=None, target_alg=None, n=0):
    experiments = collect_experiments(n, oracle, dataset, model, target_protein, target_alg)
    display_experiments_helper(experiments, target_protein=target_protein)