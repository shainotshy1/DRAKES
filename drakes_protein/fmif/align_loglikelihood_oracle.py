import os
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

from protein_oracle.data_utils import ALPHABET
import torch

def build_protgpt_oracle(device):
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2')
    model = GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2').to(device)
    tokenizer.pad_token = tokenizer.eos_token

    def protgpt_oracle(samples):
        assert len(samples) > 0, "Empty samples list passed to oracle"

        alph = list(ALPHABET) # + ['X'] # added 'X' to represent a mask since X = any in FASTA

        seq_strings = ["".join([alph[x] for x in seq]) for seq in samples] # Maybe parallelize?

        tokenized = tokenizer(seq_strings, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.to(samples[0].device)

        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs.logits[:, :-1, :] # (B, T-1, V)
        labels = input_ids[:, 1:] # (B, T-1)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1) # (B, T-1)

        log_likelihoods = token_log_probs.sum(dim=1) # (B,)
        
        return log_likelihoods
    
    return protgpt_oracle