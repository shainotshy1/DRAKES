import transformers
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
import argparse
from tqdm import tqdm
from utils import process_seq_data_directory

def protgpt_wrapper(samples, model, tokenizer):
    res = []
    for seq in tqdm(samples):
        out = tokenizer(seq, return_tensors="pt")
        input_ids = out.input_ids.cuda(device=model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        ppl = (outputs.loss * input_ids.shape[1]).item()
        res.append(ppl)
    
    res = np.array(res)
    return res

def extract_ll_distr(df, seq_label, model, tokenizer):
    assert seq_label in df.columns, f"'{seq_label}' must be a label in the data frame"
    sequences = df[seq_label]
    res = -1 * protgpt_wrapper(sequences, model, tokenizer)
    return res

def extract_ll_directory(dir_name, seq_label, model, tokenizer):
    process_seq_data_directory(dir_name, 'loglikelihood', lambda df : extract_ll_distr(df, seq_label, model, tokenizer))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Log Likelihood Computation")
    parser.add_argument("dir", help="Directory containing distribution csv files")
    parser.add_argument("--gpu", help="GPU device index", default=0)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    gpu_idx = int(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', gpu_idx)
    tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2')
    model = GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2').to(device) 

    extract_ll_directory(args.dir, 'seq', model, tokenizer)