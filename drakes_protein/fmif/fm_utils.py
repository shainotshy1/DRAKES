import torch
import copy
import torch.nn.functional as F
from collections import defaultdict
import model_utils as mu
#from fmif import model_utils as mu
import numpy as np
from torch.distributions.categorical import Categorical
from align_utils import BeamSampler, BONSampler, AlignSamplerState, OptSampler, InteractionSampler
from tree_spex import lgboost_fit, lgboost_to_fourier, lgboost_tree_to_fourier, ExactSolver
import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

def _masked_categorical(num_batch, num_res, device):
    return torch.ones(
        num_batch, num_res, device=device) * mu.MASK_TOKEN_INDEX


def _sample_categorical(categorical_probs, n = 1):
    rep_probs = categorical_probs if n == 1 else categorical_probs.repeat(n, 1, 1)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(rep_probs) + 1e-10).log())
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

    # TODO: FIX bug where clean sequences ARE NOT CLEAN...
    class ProteinDiffusionState(AlignSamplerState):
        def __init__(self, masked_seq, q_xs, demask, step, parent_state, reward_oracle):
            self.masked_seq = masked_seq
            self.pred_seq = None
            self.demask = demask
            self.q_xs = q_xs
            self.step = step
            self.parent_state = parent_state
            self.reward_oracle = reward_oracle
            self.done = (self.masked_seq != mu.MASK_TOKEN_INDEX).all()

            self.q_xs_no_mask = self.q_xs.clone()
            self.q_xs_no_mask[:, :, mu.MASK_TOKEN_INDEX] = 0

        def gen_clean_seq(self, select_argmax=False):
            if self.pred_seq is not None: return self.pred_seq.clone()
            
            if self.done:
                self.pred_seq = self.masked_seq.clone() # if masked seq is already unmasked then just return that
            else:
                copy_flag = (self.masked_seq == mu.MASK_TOKEN_INDEX).to(self.masked_seq.dtype)
                if select_argmax:
                    clean_pred = torch.argmax(self.q_xs_no_mask, dim=-1)
                else:
                    clean_pred = _sample_categorical(self.q_xs_no_mask)
                clean_pred = clean_pred * copy_flag + self.masked_seq * (1 - copy_flag)
                self.pred_seq = clean_pred
            return self.pred_seq

        def calc_reward(self, n=1, select_argmax=False):
            if select_argmax: return self.reward_oracle(self.gen_clean_seq(select_argmax=True))

            avg_reward = 0
            for _ in range(n):
                avg_reward += (1.0 / n) * self.reward_oracle(self.gen_clean_seq())

            return avg_reward
        
        def get_num_states(self):
            return self.masked_seq.shape[0]
        
        def get_state(self, i):
            res = copy.copy(self)
            res.masked_seq = res.masked_seq[i, :].unsqueeze(0)
            if res.pred_seq is not None: res.pred_seq = res.pred_seq[i, :].unsqueeze(0)
            res.q_xs = res.q_xs[i, :].unsqueeze(0)
            res.q_xs_no_mask = res.q_xs_no_mask[i, :].unsqueeze(0)
            return res

        def return_early(self):
            return self.done

        def copy_to_next_state(self):
            next_state = copy.copy(self)
            next_state.step += 1
            next_state.parent_state = self
            return next_state
            
    class ProteinModelParams():
        def __init__(self, X, mask, chain_M, residue_idx, chain_encoding_all, cls=None, w=None, n=1):
            if n > 1:
                self.X = X.repeat(n, 1, 1, 1)
                self.mask = mask.repeat(n, 1)
                self.chain_M = chain_M.repeat(n, 1)
                self.residue_idx = residue_idx.repeat(n, 1)
                self.chain_encoding_all = chain_encoding_all.repeat(n, 1)
            else:
                self.X = X
                self.mask = mask
                self.chain_M = chain_M
                self.residue_idx = residue_idx
                self.chain_encoding_all = chain_encoding_all
            self.cls = cls
            self.w = w

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
        # pred_aatypes_1 = torch.argmax(pred_logits_wo_mask, dim=-1)

        pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.neg_infinity
        pred_logits_1 = pred_logits_1 / self._cfg.temp - torch.logsumexp(pred_logits_1 / self._cfg.temp, dim=-1, keepdim=True)
        unmasked_indices = (masked_seq != mu.MASK_TOKEN_INDEX)
        pred_logits_1[unmasked_indices] = self.neg_infinity
        pred_logits_1[unmasked_indices, masked_seq[unmasked_indices]] = 0
        
        move_chance_s = 1.0 - t_2
        q_xs = pred_logits_1.exp() * d_t
        q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s #[:, :, 0]
        return q_xs#, pred_aatypes_1

    def mask_batch_to_state(self, _x, model, model_params, ts, reward_oracle, prev_state):
        copy_flag = (prev_state.masked_seq != mu.MASK_TOKEN_INDEX).to(prev_state.masked_seq.dtype)
        demask = (_x != mu.MASK_TOKEN_INDEX) * (1 - copy_flag)
        aatypes_t = prev_state.masked_seq * copy_flag + _x * (1 - copy_flag)
        t_1, t_2 = ts[prev_state.step], ts[prev_state.step + 1]
        q_xs = self.generate_state_values(model, model_params, aatypes_t, t_1, t_2)
        sample_state = self.ProteinDiffusionState(aatypes_t, q_xs, demask, prev_state.step + 1, prev_state, reward_oracle)
        return sample_state
    
    def mask_to_state_batch(self, _x, model, model_params, ts, reward_oracle, prev_state):
        n = _x.shape[0]
        copy_flag = (prev_state.masked_seq != mu.MASK_TOKEN_INDEX).to(prev_state.masked_seq.dtype).repeat(n, 1)
        demask = (_x != mu.MASK_TOKEN_INDEX) * (1 - copy_flag)
        aatypes_t = prev_state.masked_seq * copy_flag + _x * (1 - copy_flag)
        t_1, t_2 = ts[prev_state.step], ts[prev_state.step + 1]
        q_xs = self.generate_state_values(model, model_params, aatypes_t, t_1, t_2)
        sample_states = self.ProteinDiffusionState(aatypes_t, q_xs, demask, prev_state.step + 1, prev_state, reward_oracle)
        return sample_states

    def demask_to_pred_state(self, demask, model, model_params, ts, reward_oracle, prev_state):
        pred_wo_mask = prev_state.q_xs.clone()
        pred_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
        pred = torch.argmax(pred_wo_mask, dim=-1)#_sample_categorical(pred_wo_mask) #
        _x = (pred * demask + mu.MASK_TOKEN_INDEX * (1 - demask))

        copy_flag = (prev_state.masked_seq != mu.MASK_TOKEN_INDEX).to(prev_state.masked_seq.dtype)
        aatypes_t = prev_state.masked_seq * copy_flag + _x * (1 - copy_flag)
        t_1, t_2 = ts[prev_state.step], ts[prev_state.step + 1]
        q_xs = self.generate_state_values(model, model_params, aatypes_t, t_1, t_2)
        sample_state = self.ProteinDiffusionState(aatypes_t, q_xs, demask.unsqueeze(0), prev_state.step + 1, prev_state, reward_oracle)
        return sample_state

    def build_sampler_gen(self, model, model_params, ts, reward_oracle, num_timesteps, steps_per_level=1, beam_model_params=None):
        assert type(steps_per_level) is int, "steps_per_level must be of type 'int'"
        assert steps_per_level > 0, "steps_per_level must be a positive integer"
        def sampler_n_gen(state, n = 1):
            def sample():
                sample_states = state
                for i in range(steps_per_level):  
                    if sample_states.step >= num_timesteps - 1:
                        break
                    if sample_states.return_early():
                        sample_states = sample_states.copy_to_next_state()
                        continue
                    if i == 0:
                        _x = _sample_categorical(sample_states.q_xs, n=n)
                    else:
                        _x = _sample_categorical(sample_states.q_xs, n=1)
                    params = beam_model_params if (beam_model_params is not None and state.step != 1) else model_params # This model params is size n // W since each child generates this many samples instead of n, so the total is n/W * W = n
                    sample_states = self.mask_batch_to_state(_x, model, params, ts, reward_oracle, sample_states)
                if state != sample_states: # This condition is true if have we reach sample_states.step = num_timesteps - 1; i.e reached the end
                    sample_states.parent_state = state # Override to make the parent state include the whole trajectory
                return sample_states
            return sample
        return sampler_n_gen

    def build_spectral_sampler_gen(self, model, model_params, ts, reward_oracle, num_timesteps, steps_per_level=1, beam_model_params=None):
        assert type(steps_per_level) is int, "steps_per_level must be of type 'int'"
        assert steps_per_level > 0, "steps_per_level must be a positive integer"
        # Generate sampler
        def sampler_n_gen(state, n = 1):
            def sample():
                sample_states = state
                for _ in range(steps_per_level):                        
                    if sample_states.step >= num_timesteps - 1:
                        return sample_states
                    if sample_states.return_early():
                        sample_states = sample_states.copy_to_next_state()
                        continue
                    unmasked = (sample_states.masked_seq == mu.MASK_TOKEN_INDEX).squeeze()
                    num_masked = torch.sum(unmasked).cpu()
                    demask_temp = torch.bernoulli(torch.full((n, num_masked), 0.5, device=state.q_xs.device)).int() # type: ignore
                    demask = torch.zeros(size=(sample_states.masked_seq.shape[0], sample_states.q_xs.shape[1], ), device=demask_temp.device, dtype=demask_temp.dtype).repeat(n, 1)
                    demask[:, unmasked] = demask_temp
                    pred_wo_mask = sample_states.q_xs.clone()
                    pred_wo_mask[:, :, mu.MASK_TOKEN_INDEX] = -1e9
                    best_pred = torch.argmax(pred_wo_mask, dim=-1)#_sample_categorical(pred_wo_mask) #
                    _x = (best_pred * demask + mu.MASK_TOKEN_INDEX * (1 - demask))
                    sample_states = self.mask_to_state_batch(_x, model, model_params, ts, reward_oracle, sample_states)
                return sample_states
            return sample
        return sampler_n_gen

    def build_opt_selector(self, model, model_params, ts, reward_oracle, opt="linear", max_solution_order=4, lasso_lambda=0.005):
        opt_options = ["linear", "spectral"]
        assert opt in opt_options, f"{opt} not in {opt_options}"

        exact_solver = ExactSolver(maximize=True, max_solution_order=max_solution_order)

        def opt_selector(samples):
            parent_state = samples.parent_state

            if samples.return_early():
               best_demask = samples.demask[0].clone()
            else: 
                seq_len = samples.masked_seq.shape[0]
                # Assumes all samples have same parent state
                rewards = np.zeros(seq_len) if seq_len <= 1 else samples.calc_reward().cpu().numpy()
                target = rewards[0]
                eps = 1e-8
                if (np.abs(rewards - target) <= eps).all():
                    best_demask = samples.demask[0].clone()
                else:
                    all_masks = samples.demask.cpu().numpy()
                    if opt == "linear":
                        target_tokens = (samples.parent_state.masked_seq == mu.MASK_TOKEN_INDEX).detach().cpu().numpy().squeeze()
                        X = all_masks[:, target_tokens]
                        if X.shape[1] > max_solution_order:
                            # min ||Xa - r||_2^2 + lambda * ||a||_0
                            # return arg-top-k(a)
                            clf = Lasso(alpha=lasso_lambda)
                            clf.fit(X, rewards)
                            a = np.array(clf.coef_.flatten().tolist())
                            b = clf.intercept_
                            indices = np.argpartition(-a, max_solution_order-1)[:max_solution_order]
                            demask_compact = np.zeros(a.shape, dtype=all_masks.dtype)
                            demask_compact[indices] = 1
                            #pred_reward = a.T @ demask_compact + b
                        else:
                            demask_compact = np.ones(X.shape[1], dtype=all_masks.dtype)
                        best_demask = np.zeros(target_tokens.shape, dtype=all_masks.dtype)
                        best_demask[target_tokens] = demask_compact
                        best_demask = torch.tensor(best_demask, device=samples.demask.device)
                    elif opt == "spectral":    
                        best_model, cv_r2 = lgboost_fit(all_masks, rewards)
                        fourier_dict = lgboost_to_fourier(best_model)
                        fourier_dict_trunc = dict(sorted(fourier_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:2000])
                        exact_solver.load_fourier_dictionary(fourier_dict_trunc)
                        best_demask = exact_solver.solve()
                        best_demask = torch.tensor(best_demask, device=samples.demask.device)
                    else:
                        raise ValueError(f"{opt} is invalid Optimizer name")
            best_state = self.demask_to_pred_state(best_demask, model, model_params, ts, reward_oracle, parent_state)
            return best_state
        return opt_selector

    def gen_masked_state_builder(self, model, single_model_params, ts, reward_oracle):
        def masked_state_builder(masked_seq, step, parent_state):
            assert type(step) and 0 <= step < len(ts)
            q_xs = self.generate_state_values(model, single_model_params, masked_seq, ts[step - 1], ts[step])
            state = self.ProteinDiffusionState(masked_seq, q_xs, (masked_seq != mu.MASK_TOKEN_INDEX).type(torch.int8), 1, parent_state, reward_oracle)
            return state
        return masked_state_builder

    def sample(
            self,
            model,
            X, mask, chain_M, residue_idx, chain_encoding_all,
            cls=None, w=None,
            batch_oracle=None,
            n = 1,
            beam_w = 1,
            steps_per_level=1,
            align_type="bon",
            lasso_lambda=0.005,
            spec_feedback_its=0
        ):

        if type(n) != int or n < 1:
            print("Invalid BON configuration (n must be positive integer - Using normal denoising")
            n = 1

        num_batch, num_res = mask.shape        
        num_timesteps = self._cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)

        if align_type == "bon":
            steps_per_level = num_timesteps
        
        samplers = [] # (num_batch, )
        for i in range(num_batch):
            X_i = X[i].unsqueeze(0)
            mask_i = mask[i].unsqueeze(0)
            chain_M_i = chain_M[i].unsqueeze(0)
            residue_idx_i = residue_idx[i].unsqueeze(0)
            chain_encoding_all_i = chain_encoding_all[i].unsqueeze(0)
            cls_i = cls[i].unsqueeze(0) if cls is not None else None
            w_i = w[i].unsqueeze(0) if w is not None else None
            single_model_params = self.ProteinModelParams(X_i, mask_i, chain_M_i, residue_idx_i, chain_encoding_all_i, cls=cls_i, w=w_i, n=1)
            align_n = n if align_type in ["bon", "beam"] else 1
            model_params = self.ProteinModelParams(X_i, mask_i, chain_M_i, residue_idx_i, chain_encoding_all_i, cls=cls_i, w=w_i, n=align_n) # TODO: Make this shape more dynamic setting instead of this very confusing code :(
            beam_model_params = None if align_type != "beam" else self.ProteinModelParams(X_i, mask_i, chain_M_i, residue_idx_i, chain_encoding_all_i, cls=cls_i, w=w_i, n=n//beam_w) # n is n // beam_W per child (W children so total n per level)

            aatypes_0 = _masked_categorical(1, num_res, self._device).long() # single sample
            q_xs = self.generate_state_values(model, single_model_params, aatypes_0, ts[0], ts[1])
            initial_state = self.ProteinDiffusionState(aatypes_0, q_xs, torch.zeros(aatypes_0.shape, device=q_xs.device, dtype=torch.int8), 1, None, batch_oracle)
            total_steps = num_timesteps // steps_per_level + 1

            sample_gen_builder = self.build_sampler_gen # self.build_spectral_sampler_gen if align_type == "spectral" or align_type == "linear" else 
            sampler_gen = sample_gen_builder(model, model_params, ts, batch_oracle, num_timesteps, steps_per_level=steps_per_level, beam_model_params=beam_model_params)

            beam_init_model_params = self.ProteinModelParams(X_i, mask_i, chain_M_i, residue_idx_i, chain_encoding_all_i, cls=cls_i, w=w_i, n=n) # TODO: Make this shape more dynamic setting instead of this very confusing code :(
            beam_model_params = self.ProteinModelParams(X_i, mask_i, chain_M_i, residue_idx_i, chain_encoding_all_i, cls=cls_i, w=w_i, n=n//beam_w) # n is n // beam_W per child (W children so total n per level)
            beam_sampler_gen = sample_gen_builder(model, beam_init_model_params, ts, batch_oracle, num_timesteps, steps_per_level=steps_per_level, beam_model_params=beam_model_params)
            resampler = BeamSampler(beam_sampler_gen, initial_state, total_steps, n, beam_w)

            sampler = InteractionSampler(sampler_gen, initial_state, total_steps, spec_feedback_its, self.gen_masked_state_builder(model, single_model_params, ts, batch_oracle), resampler)         

            # if align_type == "linear": raise NotImplementedError()
            # elif align_type == "spectral":
            #     # opt_selector = self.build_opt_selector(model, single_model_params, ts, batch_oracle, opt=align_type, lasso_lambda=lasso_lambda)
            #     # sampler = OptSampler(sampler_gen, initial_state, total_steps, n, opt_selector)
            # else: # BEAM / BON
            #     sampler = BeamSampler(sampler_gen, initial_state, total_steps, n, beam_w)   
            samplers.append(sampler)
        best_samples = [] # (num_batch, )
        prot_traj = [] # (num_batch, num_timesteps - 1) since initial state not included now
        for i, sampler in enumerate(samplers):
            best_sample = sampler.sample_aligned()
            prot_traj.append([])
            best_samples.append(best_sample)
            curr = best_sample
            while curr is not None:
                prot_traj[-1].append(curr.masked_seq)
                curr = curr.parent_state
            prot_traj[-1] = prot_traj[-1][::-1]
        seq_dtype = prot_traj[0][0].dtype
        concat_prot_traj = []
        concat_clean_traj = []
        concat_best_samples = torch.zeros(mask.shape, device=mask.device, dtype=seq_dtype)
        # Not using, will currently comment this out
        # for i in range(len(prot_traj[0])):
        #     concat_prot_traj.append(torch.zeros(mask.shape, device=mask.device, dtype=seq_dtype))
        #     concat_clean_traj.append(torch.zeros(mask.shape, device=mask.device, dtype=seq_dtype))
        #     for j, best_sample in enumerate(best_samples):
        #         concat_prot_traj[i][j] = prot_traj[j][i]
        #         concat_clean_traj[i][j] = clean_traj[j][i]
        for i, best_sample in enumerate(best_samples):
            concat_best_samples[i] = best_sample.gen_clean_seq()
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
        return pred_aatypes_1, prot_traj, clean_traj # type: ignore

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
