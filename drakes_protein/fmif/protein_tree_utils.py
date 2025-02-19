import torch
from align_utils import AlignSamplerState

class BatchSampleState:
    def get_data_in(self):
        raise NotImplemented

    def load_data_out(self, data_out):
        raise NotImplemented

class ProteinDiffusionState(AlignSamplerState, BatchSampleState):
    def __init__(self, masked_seq, clean_seq, q_xs, step, parent_state, reward_oracle, mu, ts):
        self.masked_seq = masked_seq
        self.q_xs = q_xs
        self.step = step
        self.clean_seq = clean_seq
        self.parent_state = parent_state
        self.reward_oracle = reward_oracle
        self.mu = mu
        self.ts = ts
    
    def calc_reward(self):
        return self.reward_oracle(self.clean_seq)

    def get_data_in(self):
        return self.masked_seq

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)


    def gen_next_state(self, data_out):
        _x = self._sample_categorical(self.q_xs)
        copy_flag = (self.masked_seq != self.mu.MASK_TOKEN_INDEX).to(self.masked_seq.dtype)
        aatypes_t = self.masked_seq * copy_flag + _x * (1 - copy_flag)

        pred_logits_1 = data_out # [bsz, seqlen, 22]
        pred_logits_wo_mask = pred_logits_1.clone()
        pred_logits_wo_mask[:, :, self.mu.MASK_TOKEN_INDEX] = -1e9
        pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)

        pred_logits_1[:, :, self.mu.MASK_TOKEN_INDEX] = self.neg_infinity
        pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, dim=-1, keepdim=True)
        unmasked_indices = (self.masked_seq != self.mu.MASK_TOKEN_INDEX)
        pred_logits_1[unmasked_indices] = self.neg_infinity
        pred_logits_1[unmasked_indices, self.masked_seq[unmasked_indices]] = 0
        
        t_1, t_2 = self.ts[self.state.step], self.ts[self.state.step + 1]
        d_t = t_2 - t_1
        move_chance_s = 1.0 - t_2
        q_xs = pred_logits_1.exp() * d_t
        q_xs[:, :, self.mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
        sample_state = self.ProteinDiffusionState(aatypes_t, pred_aatypes_1, q_xs, self.step + 1, self, self.reward_oracle)
        return sample_state

class BatchSampler:
    def get_num_samples():
        raise NotImplemented
    
    def load_data_in(self, i, data_in):
        raise NotImplemented

    def get_data_out(self, i):
        raise NotImplemented
    
    def sample(self):
        raise NotImplemented

class BatchSamplerGenerator:
    def __init__(self, batch_sampler):
        self.num_samples = batch_sampler.get_num_samples()
        assert type(self.num_samples) is int and self.num_samples > 0, "batch_sampler.get_num_samples() must return a positive integer"
        self.batch_sampler = batch_sampler
        self.ready_samples = 0

    def generate_sampler_gens(self, batch_sampler):
        sampler_gens = []
        for i in range(self.num_samples):
            def sampler_gen(state):
                assert isinstance(state, BatchSampleState)
                def sample():
                    # Load data into self.data
                    self.batch_sampler.load_data_in(i, state.get_data_in())
                    # condition.acquire()
                    # increment num samples ready
                    self.ready_samples += 1
                    # if all samples are ready, call run_sample() and then condition.notify_all()
                    if self.ready_samples == self.num_samples:
                        self.run_sampler(batch_sampler)
                        self.ready_samples = 0
                    # else condition.wait()
                    return state.gen_next_state(self.batch_sampler.get_data_out(i))
                return sample
            sampler_gens.append(sampler_gen)

    def run_sampler(self):
        self.batch_sampler.sample()

class ProteinMPNNBatchSampler(BatchSampler):
    def __init__(self, num_samples, data_in_shape, data_out_shape, device):
        assert type(num_samples) is int and num_samples > 0
        self.data_in = torch.zeros((num_samples, ) + data_in_shape, device=device)
        self.data_out = torch.zeros((num_samples, ) + data_out_shape, device=device)

    def get_num_samples():
        raise NotImplemented
    
    def load_data_in(self, i, data_in):
        raise NotImplemented

    def get_data_out(self, i):
        raise NotImplemented
    
    def sample(self):
        q_xs, pred_aatypes_1 = self.generate_state_values(self.model, self.model_params, self.data_in, self.t_1, self.t_2, self.mu)
        
    
    def generate_state_values(self, model, model_params, masked_seq, t_1, t_2, mu):
        # Extract parameters
        X = model_params.X
        mask = model_params.mask
        chain_M = model_params.chain_M
        residue_idx = model_params.residue_idx
        chain_encoding_all = model_params.chain_encoding_all
        cls = model_params.cls
        w = model_params.w
        d_t = t_2 - t_1

        with torch.no_grad():
            if cls is not None:
                uncond = (2 * torch.ones(X.shape[0], device=X.device)).long()
                cond = (cls * torch.ones(X.shape[0], device=X.device)).long()
                model_out_uncond = model(X, masked_seq, mask, chain_M, residue_idx, chain_encoding_all, cls=uncond)
                model_out_cond = model(X, masked_seq, mask, chain_M, residue_idx, chain_encoding_all, cls=cond)
                model_out = (1+w) * model_out_cond - w * model_out_uncond
            else:
                model_out = model(X, masked_seq, mask, chain_M, residue_idx, chain_encoding_all)
        pred_logits_1 = model_out # [bsz, seqlen, 22]
        pred_logits_wo_mask = pred_logits_1.clone()
        pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
        pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)

        pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
        pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, dim=-1, keepdim=True)
        unmasked_indices = (masked_seq != mu.MASK_TOKEN_INDEX)
        pred_logits_1[unmasked_indices] = self.neg_infinity
        pred_logits_1[unmasked_indices, masked_seq[unmasked_indices]] = 0
        
        move_chance_s = 1.0 - t_2
        q_xs = pred_logits_1.exp() * d_t
        q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
        return q_xs, pred_aatypes_1