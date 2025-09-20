import os
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

from protein_oracle.data_utils import ALPHABET
import torch

def build_protgpt_oracle(device):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2')
    model = GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2').to(device)
    
    def protgpt_oracle(samples):
        rewards = []
        assert len(samples) > 0, "Empty samples list passed to oracle"

        alph = list(ALPHABET) #+ ['X'] # added 'X' to represent a mask since X = any in FASTA
        for seq in samples:
            seq_str = "".join([alph[x] for x in seq])
            out = tokenizer(seq_str, return_tensors="pt")
            input_ids = out.input_ids.cuda().to(seq.device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            log_likelihood = -1 * (outputs.loss * input_ids.shape[1]).item()
        
            rewards.append(log_likelihood)
        return torch.tensor(rewards, device=seq.device) # type: ignore
    
    return protgpt_oracle