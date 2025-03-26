import torch
import copy
import torch.nn.functional as F
from collections import defaultdict
import model_utils as mu
#from fmif import model_utils as mu
import numpy as np
from torch.distributions.categorical import Categorical
from align_utils import BeamSampler, MCTSSampler, BONSampler, AlignSamplerState

def _masked_categorical(num_batch, num_res, device):
    return torch.ones(
        num_batch, num_res, device=device) * mu.MASK_TOKEN_INDEX


def _sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _sample_categorical_gradient(categorical_probs, temp = 1.0):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    output = torch.nn.functional.softmax((torch.log(categorical_probs)-torch.log(gumbel_norm))/temp, 2)
    return output


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self.num_tokens = 22
        self.neg_infinity = -1000000.0

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_aatypes(self, aatypes_1, t, res_mask): #, diffuse_mask):
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)

        if self._cfg.interpolant_type == "masking":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1 - t) # (B, N) # t=1 is clean data

            aatypes_t[corruption_mask] = mu.MASK_TOKEN_INDEX

            aatypes_t = aatypes_t * res_mask + mu.MASK_TOKEN_INDEX * (1 - res_mask)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._cfg.interpolant_type}")

        return aatypes_t.long()

    def corrupt_batch(self, batch, t=None):
        noisy_batch = copy.deepcopy(batch)
        X, S, mask, chain_M, residue_idx, chain_encoding_all = noisy_batch
        noisy_batch = {}
        noisy_batch['X'] = X
        noisy_batch['S'] = S
        noisy_batch['mask'] = mask
        noisy_batch['chain_M'] = chain_M
        noisy_batch['residue_idx'] = residue_idx
        noisy_batch['chain_encoding_all'] = chain_encoding_all
        aatypes_1 = S
        num_batch, num_res = aatypes_1.shape
        
        if t is None:
            t = self.sample_t(num_batch)[:, None]
        else:
            t = torch.ones((num_batch, 1), device=self._device) * t
        noisy_batch['t'] = t
        res_mask = mask * chain_M
        aatypes_t = self._corrupt_aatypes(aatypes_1, t, res_mask)
        noisy_batch['S_t'] = aatypes_t
        return noisy_batch

    class ProteinDiffusionState(AlignSamplerState):
        def __init__(self, masked_seq, clean_seq, q_xs, step, parent_state, reward_oracle):
            self.masked_seq = masked_seq
            self.q_xs = q_xs
            self.step = step
            self.clean_seq = clean_seq
            self.parent_state = parent_state
            self.reward_oracle = reward_oracle
        
        def calc_reward(self):
            return self.reward_oracle(self.clean_seq)
    
    class ProteinModelParams():
        def __init__(self, X, mask, chain_M, residue_idx, chain_encoding_all, cls=None, w=None):
            self.X = X
            self.mask = mask
            self.chain_M = chain_M
            self.residue_idx = residue_idx
            self.chain_encoding_all = chain_encoding_all
            self.cls = cls
            self.w = w

    class ProteinBatchTreeSearch():
        def __init__(self):
            return

    def generate_state_values(self, model, model_params, masked_seq, t_1, t_2):
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

    def build_sampler_gen(self, model, model_params, ts, reward_oracle, num_timesteps, steps_per_level=1):
        assert type(steps_per_level) is int, "steps_per_level must be of type 'int'"
        assert steps_per_level > 0, "steps_per_level must be a positive integer"
        # Generate sampler
        def sampler_gen(state):
            def sample():
                sample_state = state
                for _ in range(steps_per_level):
                    if sample_state.step >= num_timesteps - 1:
                        return sample_state
                    _x = _sample_categorical(sample_state.q_xs)
                    copy_flag = (sample_state.masked_seq != mu.MASK_TOKEN_INDEX).to(sample_state.masked_seq.dtype)
                    aatypes_t = sample_state.masked_seq * copy_flag + _x * (1 - copy_flag)
                    t_1, t_2 = ts[sample_state.step], ts[sample_state.step + 1]
                    q_xs, pred_aatypes_1 = self.generate_state_values(model, model_params, aatypes_t, t_1, t_2)
                    sample_state = self.ProteinDiffusionState(aatypes_t, pred_aatypes_1, q_xs, sample_state.step + 1, sample_state, reward_oracle)
                return sample_state
            return sample
        return sampler_gen

    def sample(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            cls=None, w=None,
            reward_model=None,
            scRMSD_reward=None,
            n = 1,
            bon_step_inteval=1, # 1 bon step for each interval
            steps_per_level=1
        ):

        if type(n) != int or n < 1 or reward_model is None:
            print("Invalid BON configuration (n must be positive integer, reward_model can't be None - Using normal denoising")
            n = 1
        assert type(bon_step_inteval) is int and bon_step_inteval > 0, "BON step interval must be positive integer"

        num_batch, num_res = mask.shape        
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        
        samplers = [] # (num_batch, )
        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            mask_i = mask[i].unsqueeze(0)
            chain_M_i = chain_M[i].unsqueeze(0)
            residue_idx_i = residue_idx[i].unsqueeze(0)
            chain_encoding_all_i = chain_encoding_all[i].unsqueeze(0)
            cls_i = cls[i].unsqueeze(0) if cls is not None else None
            w_i = w[i].unsqueeze(0) if w is not None else None
            model_params = self.ProteinModelParams(X_i, mask_i, chain_M_i, residue_idx_i, chain_encoding_all_i, cls=cls_i, w=w_i)

            reward_model_i = scRMSD_reward#lambda S : reward_model(X_i, S, mask_i, chain_M_i, residue_idx_i, chain_encoding_all_i)
            sampler_gen = self.build_sampler_gen(model, model_params, ts, reward_model_i, num_timesteps, steps_per_level=steps_per_level)
            aatypes_0 = _masked_categorical(1, num_res, self._device).long() # single sample
            q_xs, pred_aatypes_1 = self.generate_state_values(model, model_params, aatypes_0, ts[0], ts[1])
            initial_state = self.ProteinDiffusionState(aatypes_0, pred_aatypes_1, q_xs, 1, None, reward_model_i)

            # Currently just using Beam for the normal diffusion process and using BON on the whole process
            total_steps = num_timesteps // steps_per_level + 1
            sampler = BeamSampler(sampler_gen, initial_state, total_steps, n, 1)
                # MCTSSampler(sampler_gen, initial_state, total_steps, n, 1)
            # BON on entire process
            #bon_sampler = BONSampler(sampler.sample_aligned, n, 1)
            
            samplers.append(sampler)
            #samplers.append(bon_sampler)
        
        best_samples = [] # (num_batch, )
        prot_traj = [] # (num_batch, num_timesteps - 1) since initial state not included now
        clean_traj = [] # (num_batch, num_timesteps - 1) since initial state not included now
        for i, sampler in enumerate(samplers):
            best_sample = sampler.sample_aligned()
            if type(sampler) is BONSampler: # TODO: probably temporary
                best_idx = best_sample[1][0]
                best_sample = best_sample[0][best_idx] # If BON just take first element of the list => hard coded since bon returns 3 element tuple oops
            prot_traj.append([])
            clean_traj.append([])
            best_samples.append(best_sample)
            curr = best_sample
            while curr is not None:
                prot_traj[-1].append(curr.masked_seq)
                clean_traj[-1].append(curr.clean_seq)
                curr = curr.parent_state
            prot_traj[-1] = prot_traj[-1][::-1]
            clean_traj[-1] = clean_traj[-1][::-1]

        seq_dtype = prot_traj[0][0].dtype
        concat_prot_traj = []
        concat_clean_traj = []
        concat_best_samples = torch.zeros(mask.shape, device=mask.device, dtype=seq_dtype)
        for i in range(len(prot_traj[0])):
            concat_prot_traj.append(torch.zeros(mask.shape, device=mask.device, dtype=seq_dtype))
            concat_clean_traj.append(torch.zeros(mask.shape, device=mask.device, dtype=seq_dtype))
            for j, best_sample in enumerate(best_samples):
                concat_prot_traj[i][j] = prot_traj[j][i]
                concat_clean_traj[i][j] = clean_traj[j][i]
        for i, best_sample in enumerate(best_samples):
            concat_best_samples[i] = best_sample.clean_seq
        return concat_best_samples, concat_prot_traj, concat_clean_traj

    def sample_gradient(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            truncate_steps, gumbel_softmax_temp
        ):
        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        aatypes_0 = F.one_hot(aatypes_0, num_classes=self.num_tokens).float()

        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0
        last_x_list = []
        move_chance_t_list = []
        copy_flag_list = []

        for _ts, t_2 in enumerate(ts[1:]):
            d_t = t_2 - t_1
            model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
            pred_logits_1 = model_out.clone()
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)

            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                             dim=-1, keepdim=True)
            if aatypes_t_1.ndim > 2 and aatypes_t_1.shape[-1] == self.num_tokens:
                aatypes_t_1_argmax = aatypes_t_1.argmax(dim=-1)
            else:
                aatypes_t_1_argmax = aatypes_t_1
            unmasked_indices = (aatypes_t_1_argmax != mu.MASK_TOKEN_INDEX)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1_argmax[unmasked_indices]] = 0
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t
            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s
            if _ts < num_timesteps - truncate_steps:
                _x = _sample_categorical(q_xs)
                _x = F.one_hot(_x, num_classes=self.num_tokens).float()
                copy_flag = (aatypes_t_1.argmax(dim=-1) != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype).unsqueeze(-1)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)

                aatypes_t_2 = aatypes_t_2.detach()
                aatypes_t_1 = aatypes_t_1.detach()
            else:
                q_xs = q_xs + 1e-8
                _x = _sample_categorical_gradient(q_xs, gumbel_softmax_temp)
                copy_flag = 1 - aatypes_t_1[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1)
                aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)

            last_x_list.append(aatypes_t_1)
            move_chance_t_list.append(move_chance_t + self._cfg.min_t)
            copy_flag_list.append(copy_flag)
            aatypes_t_1 = aatypes_t_2
            t_1 = t_2

        last_x_list.append(aatypes_t_1)
        move_chance_t_list.append(1.0 - t_1 + self._cfg.min_t)
        copy_flag_list.append(1 - aatypes_t_1[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1))

        aatypes_t_1_argmax = aatypes_t_1[:, :, :-1].argmax(dim=-1) # to avoid the mask token
        aatypes_t_1_argmax = F.one_hot(aatypes_t_1_argmax, num_classes=self.num_tokens).float()
        return aatypes_t_1 + (aatypes_t_1_argmax - aatypes_t_1).detach(), last_x_list, move_chance_t_list, copy_flag_list


    def sample_controlled_CG(self,
                              model,
                              X, mask, chain_M, residue_idx, chain_encoding_all,
                              guidance_scale, reward_model):
        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())

            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                            dim=-1, keepdim=True)
            unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t

            x_onehot = F.one_hot(aatypes_t_1, num_classes=self.num_tokens).float()
            x_grad = self.compute_gradient_CG(model, x_onehot, reward_model, X, mask, chain_M, residue_idx, chain_encoding_all)
            guidance = guidance_scale * (x_grad - x_grad[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1))
            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s
            q_xs = q_xs * guidance.exp()
            _x = _sample_categorical(q_xs)
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)
            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())
            t_1 = t_2

        t_1 = ts[-1]
        return pred_aatypes_1, prot_traj, clean_traj

    def compute_gradient_CG(self, model, x_onehot, reward_model,
                             X, mask, chain_M, residue_idx, chain_encoding_all):
        x_onehot.requires_grad_(True)
        expected_x0 = model(X, x_onehot, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_t]
        scores = reward_model(X, expected_x0, mask, chain_M, residue_idx, chain_encoding_all)
        scores = scores.mean()
        scores.backward()
        x_grad = x_onehot.grad.clone()
        return x_grad


    def sample_controlled_SMC(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            reward_model, alpha
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)

            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())
            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                            dim=-1, keepdim=True)
            unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
            
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t
            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
            _x = _sample_categorical(q_xs)
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)

            '''
            Calcualte exp(v_{t-1}(x_{t-1})/alpha)
            '''
            expected_x0_pes = model(X, aatypes_t_2, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_{t-1}]
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_2.dtype)
            one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
            improve_hot_x0 = copy_flag * aatypes_t_2 + (1 - copy_flag) *  one_hot_x0
            reward_num = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

            '''
            Calcualte exp(v_{t}(x_{t})/alpha)
            '''
            expected_x0_pes = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_t]
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
            improve_hot_x0 = copy_flag * aatypes_t_1 + (1 - copy_flag) *  one_hot_x0
            reward_den = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

            ratio = torch.exp(1.0/alpha * (reward_num - reward_den)) # Now calculate exp( (v_{t-1}(x_{t-1) -v_{t}(x_{t}) /alpha) 
            ratio = ratio.detach().cpu().numpy()
            final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p =  ratio/ratio.sum() ) 
            aatypes_t_2 = aatypes_t_2[final_sample_indices]
            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())
            t_1 = t_2

        t_1 = ts[-1]
        return pred_aatypes_1, prot_traj, clean_traj



    def sample_controlled_TDS(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            reward_model, alpha, guidance_scale
        ):

        num_batch, num_res = mask.shape
        aatypes_0 = _masked_categorical(num_batch, num_res, self._device).long()
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        aatypes_t_1 = aatypes_0 # [bsz, seqlen]
        prot_traj = [aatypes_0.detach().cpu()] 
        clean_traj = []
        for t_2 in ts[1:]:
            d_t = t_2 - t_1
            with torch.no_grad():
               model_out = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all)
            pred_logits_1 = model_out # [bsz, seqlen, 22]
            pred_logits_wo_mask = pred_logits_1.clone()
            pred_logits_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
            pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)
            clean_traj.append(pred_aatypes_1.detach().cpu())
            pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
            pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, 
                                                                            dim=-1, keepdim=True)
            unmasked_indices = (aatypes_t_1 != mu.MASK_TOKEN_INDEX)
            pred_logits_1[unmasked_indices] = self.neg_infinity
            pred_logits_1[unmasked_indices, aatypes_t_1[unmasked_indices]] = 0
            move_chance_t = 1.0 - t_1
            move_chance_s = 1.0 - t_2
            q_xs = pred_logits_1.exp() * d_t
            x_onehot = F.one_hot(aatypes_t_1, num_classes=self.num_tokens).float()
            x_grad = self.compute_gradient_CG(model, x_onehot, reward_model, X, mask, chain_M, residue_idx, chain_encoding_all)
            guidance = guidance_scale * (x_grad - x_grad[:, :, mu.MASK_TOKEN_INDEX].unsqueeze(-1))

            q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s
            q_xs = q_xs * guidance.exp()
            _x = _sample_categorical(q_xs)
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            aatypes_t_2 = aatypes_t_1 * copy_flag + _x * (1 - copy_flag)
            prob_multiplier = (1 - copy_flag) * torch.gather(guidance.exp(), 2, _x.unsqueeze(-1)).squeeze(-1) + copy_flag * torch.ones_like(_x)

            '''
            Calcualte exp(v_{t-1}(x_{t-1})/alpha)
            '''
            expected_x0_pes = model(X, aatypes_t_2, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_{t-1}]
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_2.dtype)
            one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
            improve_hot_x0 = copy_flag * aatypes_t_2 + (1 - copy_flag) *  one_hot_x0
            reward_num = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

            '''
            Calcualte exp(v_{t}(x_{t})/alpha)
            '''
            expected_x0_pes = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all) # Calcualte E[x_0|x_t]
            copy_flag = (aatypes_t_1 != mu.MASK_TOKEN_INDEX).to(aatypes_t_1.dtype)
            one_hot_x0 = torch.argmax(expected_x0_pes, dim = 2)
            improve_hot_x0 = copy_flag * aatypes_t_1 + (1 - copy_flag) *  one_hot_x0
            reward_den = reward_model(X, 1.0 * F.one_hot(improve_hot_x0, num_classes= 22) , mask, chain_M, residue_idx, chain_encoding_all)

            ratio = torch.exp(1.0/alpha * (reward_num - reward_den)) / prob_multiplier.prod(dim=-1) # Now calculate exp( (v_{t-1}(x_{t-1) -v_{t}(x_{t}) /alpha) 
            ratio = ratio.detach().cpu().numpy()
            final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p =  ratio/ratio.sum() ) 
            aatypes_t_2 = aatypes_t_2[final_sample_indices]
            aatypes_t_1 = aatypes_t_2.long()
            prot_traj.append(aatypes_t_2.cpu().detach())
            t_1 = t_2

        t_1 = ts[-1]
        return pred_aatypes_1, prot_traj, clean_traj


def fm_model_step(model, noisy_batch, cls=None):
    loss_mask = noisy_batch['mask'] * noisy_batch['chain_M']
    if torch.any(torch.sum(loss_mask, dim=-1) < 1):
        raise ValueError('Empty batch encountered')
    
    # Model output predictions.
    X = noisy_batch['X']
    aatypes_t_1 = noisy_batch['S_t']
    mask = noisy_batch['mask']
    chain_M = noisy_batch['chain_M']
    residue_idx = noisy_batch['residue_idx']
    chain_encoding_all = noisy_batch['chain_encoding_all']
    pred_logits = model(X, aatypes_t_1, mask, chain_M, residue_idx, chain_encoding_all, cls=cls)

    return pred_logits
