import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tree_spex import lgboost_fit, lgboost_to_fourier, lgboost_tree_to_fourier, ExactSolver # type:ignore
from protein_oracle.data_utils import ALPHABET, ALPHABET_WITH_MASK
from sklearn.linear_model import Lasso
import model_utils as mu # type:ignore
from math import floor
from tqdm import tqdm
import copy
from itertools import chain, combinations

def power_subset(iterable):
    "power_subset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


class AlignSamplerState():
    def calc_reward(self):
        raise NotImplemented
    
    def get_num_states(self):
        raise NotImplemented
    
    def get_state(self, i):
        raise NotImplemented
    
    def return_early(self):
        return False

class BONSampler():
    def __init__(self, sampler, W, soft=False):
        # Parameter validation
        assert type(W) is int, "W must be type 'int"
        assert W > 0, "W must be a positive integer"
        self.sampler = sampler
        self.W = W # top W results returned
        self.soft = soft # Soft max sampling used instead of argma
        if self.soft:
            self.sm = nn.Softmax()
        super().__init__()

    def sample_aligned(self):
        samples = self.sampler()
        assert isinstance(samples, AlignSamplerState), "Sample must be instance of AlignSamplerState"
        rewards = samples.calc_reward(select_argmax=True) # type: ignore
        if self.soft:
            sm_rewards = self.sm(rewards)
            top_indices = torch.multinomial(sm_rewards, num_samples=self.W, replacement=True)
        else:
            _, top_indices = torch.topk(rewards, self.W, dim=0)
        return samples, top_indices, rewards

class TreeStateSampler():
    def __init__(self, sampler_gen, initial_state, depth, child_n):
        # Parameter validation
        assert type(depth) is int, "depth must be type 'int'"
        assert depth > 0, "depth must be a positive integer"
        assert type(child_n) is int, "child_n must be type 'int'"
        assert child_n > 0, "child_n must be a positive integer"
        self.sampler_gen = sampler_gen
        self.initial_state = initial_state
        self.depth = depth
        self.child_n = child_n

    def gen_tree_visual(self, gen_states, num_states, labels):
        G = nx.DiGraph()

        color_grad = colormaps['inferno']

        G.add_node(len(G.nodes), label=labels[0][0][0])

        prev_layer_parents = [0]
        global_min = float(labels[0][0][0])
        global_max = float(labels[0][0][0])
        for i, level in enumerate(num_states):
            new_prev_layer = []
            original_pos = {}
            for j, n in sorted(list(enumerate(level)), key=lambda x : prev_layer_parents[x[0]]):
                float_labels = [float(k) for k in labels[i][j]]
                global_min = min(global_min, min(float_labels))
                global_max = max(global_max, max(float_labels))
                gen_state = gen_states[i][j]
                parent = prev_layer_parents[j]
                new_nodes = []
                new_parents = []
                for k in range(n):
                    new_node = len(G.nodes) + 1
                    new_nodes.append(new_node)
                    G.add_node(new_node, label=labels[i][j][k])
                    G.add_edge(parent, new_node)
                for k in gen_state:
                    next_gen = new_nodes[k]
                    new_parents.append(next_gen)
                    original_pos[next_gen] = j
                new_prev_layer.extend(new_parents)
            new_prev_layer = sorted(new_prev_layer, key=lambda x : original_pos[x])

            prev_layer_parents = new_prev_layer

        i = 0
        for i, layer in enumerate(reversed(list(nx.topological_generations(G)))):
            for n in layer:
                G.nodes[n]["layer"] = i
        max_layer = i + 1
        
        pad = (global_max - global_min) / 4
        min_val, max_val = global_min - pad, global_max + pad
        norm = plt.Normalize(min_val, max_val) # type: ignore
        node_colors = [color_grad(1 - norm(float(data["label"]))) for _, data in G.nodes(data=True)]
        labels = {node: data["label"] for node, data in G.nodes(data=True)}

        fig_width = 6
        fig_height = max_layer * 1.3
        plt.figure(figsize=(fig_width, fig_height))

        pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=900, font_weight='bold', node_color=node_colors, font_size=6, font_color='white', arrows=False)
        plt.savefig('tree.png')

class OptSampler(TreeStateSampler):   
    def __init__(self, sampler_gen, initial_state, depth, child_n, opt_selector):
        self.opt_selector = opt_selector
        super().__init__(sampler_gen, initial_state, depth, child_n)

    def sample_aligned(self):
        state = self.initial_state
        for _ in range(self.depth - 1):
            assert isinstance(state, AlignSamplerState), "State must be instance of AlignSamplerState"
            sampler = self.sampler_gen(state, n=self.child_n)
            if state.return_early():
                samples = sampler()
            else:
                samples = sampler()
            state = self.opt_selector(samples)
        return state

class MHSampler():
    def __init__(self, initial_state, denoise_steps, state_builder, sampler, mh_type='uniform'):
        assert type(denoise_steps) is int, "denoise_steps must be type 'int'"
        assert denoise_steps > 0, "denoise_steps must be a positive integer"
        method_opts = ['uniform', 'split-gibbs']
        assert mh_type in method_opts, f"mh_type must be in {method_opts}"
        self.mh_type = mh_type
        
        self.initial_state = initial_state
        self.state_builder = state_builder
        self.sampler = sampler
        self.num_tokens = initial_state.masked_seq.shape[1]

    # Split-Gibbs does a annealed noising + symmetric pertubation of x to get z, sampling via Metropolis-within-Gibbs

    def gen_remasked_state(self, state, mask, inplace=False):
        masked_seq = state.gen_clean_seq(inplace)
        masked_seq[0][mask == 1] = mu.MASK_TOKEN_INDEX # Here a 1 means we mask the token
        new_step = 1
        state = self.state_builder(masked_seq, new_step, state) # Re-run diffusion process from this partially masked state
        return state

    def sample_aligned(self, N=0, p=0.5, beta=1.0):
        if self.mh_type == 'uniform':
            return self.sample_aligned_uniform(N, p, beta)
        elif self.mh_type == 'split-gibbs':
            return self.sample_aligned_split_gibbs(N, beta)
        else:
            raise NotImplementedError(f"Unsupported MH sampler method: '{self.mh_type}'")

    def sample_aligned_uniform(self, N, p, beta):
        mask_sampler = Bernoulli(probs=torch.full((self.num_tokens,), p))
        self.sampler.initial_state = self.initial_state
        state = self.sampler.sample_aligned()
        prev_reward = state.calc_reward()
        acceptances = 0
        reward_traj = [prev_reward.item()]
        for _ in range(N):
            mask = mask_sampler.sample()
            remasked_state = self.gen_remasked_state(state, mask)
            self.sampler.initial_state = remasked_state
            proposed_state = self.sampler.sample_aligned()
            proposed_reward = proposed_state.calc_reward()
            proposal_prob = min(torch.tensor(1.0), torch.exp((proposed_reward - prev_reward) / beta))
            accept = torch.bernoulli(proposal_prob)

            acceptances += accept.item()
            if accept == 1:
                state = proposed_state
                prev_reward = proposed_reward
            reward_traj.append(prev_reward.item())

        rate = acceptances / N
        print(f"Final Reward: {prev_reward.item()}, Acceptance Rate: {rate}")
        state.reward_traj = reward_traj
        return state

    def sample_aligned_split_gibbs(self, N, beta, mh_steps=100):
        self.sampler.initial_state = self.initial_state
        state = self.sampler.sample_aligned()
        clean_seq = state.gen_clean_seq(inplace=True).clone()
        prev_reward = state.calc_reward()
        reward_traj = [prev_reward.item()]
        L = clean_seq.shape[-1]

        vocab_size = len(ALPHABET)
        for n in range(N):
            # pi(z | x)
            d_prev = 0
            z_prev_reward = prev_reward
            # m_prob = (N - n) / (N + 1)
            acceptances = 0
            prop_probs = 0
            nu_min = 1e-4
            nu_max = 20.0
            nu = (nu_min ** ((n + 1) / (N+1))) * (nu_max ** (1 - (n + 1) / (N+1)))
            m_prob = (N - 1) / N * (1 - np.exp(-nu))
            for mh_it in range(mh_steps):
                idx = torch.randint(L, (1,), device=clean_seq.device)
                token = torch.randint(vocab_size, (1,), device=clean_seq.device)
                old_token = state.get_token(idx)
                state.set_token(idx, token)
                prop_seq = state.gen_clean_seq(inplace=True)
                prop_reward = state.calc_reward()
                d = int(torch.sum(prop_seq != clean_seq))
                proposal_prob = torch.exp((prop_reward - z_prev_reward) / beta + (d - d_prev) * np.log(m_prob / (1 - m_prob))).clamp(min=0.0, max=1.0) # This coefficient is probably wrong
                accept = torch.bernoulli(proposal_prob)
                
                if accept == 1:
                    z_prev_reward = prop_reward
                    d_prev = d
                    acceptances += 1
                else:
                    state.set_token(idx, old_token)
                prop_probs += proposal_prob
            # print("Acceptance Rate:", acceptances / mh_steps, prop_probs / mh_steps)
            
            # pi(x | z)
            mask_sampler = Bernoulli(probs=torch.full((self.num_tokens,), m_prob))
            mask = mask_sampler.sample()
            remasked_state = self.gen_remasked_state(state, mask)
            self.sampler.initial_state = remasked_state
            state = self.sampler.sample_aligned()
            prev_reward = state.calc_reward()
            reward_traj.append(prev_reward.item())
            clean_seq = state.gen_clean_seq().clone()

        print(f"Final Reward: {prev_reward.item()}")
        state.reward_traj = reward_traj
        return state

class InteractionSampler():
    def __init__(self, initial_state, depth, feedback_steps, max_spec_order, feedback_method, state_builder, resampler, interpolant, model, model_params, lasso_pen=0.0, num_masks=512, batch_max=False):
        # Parameter validation
        assert type(depth) is int, "depth must be type 'int'"
        assert depth > 0, "depth must be a positive integer"
        self.initial_state = initial_state
        self.depth = depth
        self.feedback_steps = feedback_steps
        self.state_builder = state_builder
        self.resampler = resampler
        self.max_spec_order = max_spec_order if max_spec_order > 0 else None
        self.feedback_method = feedback_method
        self.lasso_pen = lasso_pen
        self.exact_solver = ExactSolver(maximize=True, max_solution_order=self.max_spec_order)
        self.interpolant = interpolant

        max_batch = 512
        self.num_masks=num_masks
        self.mask_batch=min(max_batch, self.num_masks)
        self.remask_batch=min(max_batch, self.num_masks)
        self.reward_batch=min(max_batch, self.num_masks)

        self.batch_max = batch_max
        self.reward_avg_n=5
        self.p = 0.75

        # copy so that we can update the values of the params
        # yes my code in this repo is quite bad, but I'm too far in - trust the process >:)
        self.model_params = copy.copy(model_params)

        self.model_params.X = self.model_params.X.repeat(self.remask_batch, 1, 1, 1)
        self.model_params.mask = self.model_params.mask.repeat(self.remask_batch, 1)
        self.model_params.chain_M = self.model_params.chain_M.repeat(self.remask_batch, 1)
        self.model_params.residue_idx = self.model_params.residue_idx.repeat(self.remask_batch, 1)
        self.model_params.chain_encoding_all = self.model_params.chain_encoding_all.repeat(self.remask_batch, 1)

        self.model = model

    def generate_remasked_state(self, state, mask):
        masked_seq = state.gen_clean_seq()
        masked_seq[0][mask == 0] = mu.MASK_TOKEN_INDEX
        new_step = 1

        state = self.state_builder(masked_seq, new_step, state) # p(x | x_t)
        return state
    
    def generate_remasked_state_batch(self, state, masks):
        masked_seq = state.gen_clean_seq()[0]
        res = torch.zeros_like(torch.tensor(masks), device=masked_seq.device)
        res[:] = masked_seq
        res[masks == 0] = mu.MASK_TOKEN_INDEX
        return res
    
    def sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    
    def diffusion_mask_infill(self, batch, q_xs_no_mask):
        copy_flag = (batch == mu.MASK_TOKEN_INDEX).to(batch.dtype)
        pred = self.sample_categorical(q_xs_no_mask)
        pred = pred * copy_flag + batch * (1 - copy_flag) # (B, L)
        return pred
    
    def sample_batch_step_qx(self, masked_seq, t_1, t_2):
        # Extract parameters
        X = self.model_params.X
        mask = self.model_params.mask
        chain_M = self.model_params.chain_M
        residue_idx = self.model_params.residue_idx
        chain_encoding_all = self.model_params.chain_encoding_all
        cls = self.model_params.cls
        w = self.model_params.w
        d_t = t_2 - t_1

        with torch.no_grad():
            if cls is not None:
                uncond = (2 * torch.ones(X.shape[0], device=X.device)).long()
                cond = (cls * torch.ones(X.shape[0], device=X.device)).long()
                model_out_uncond = self.model(X, masked_seq, mask, chain_M, residue_idx, chain_encoding_all, cls=uncond)
                model_out_cond = self.model(X, masked_seq, mask, chain_M, residue_idx, chain_encoding_all, cls=cond)
                model_out = (1+w) * model_out_cond - w * model_out_uncond
            else:
                model_out = self.model(X, masked_seq, mask, chain_M, residue_idx, chain_encoding_all)
        pred_logits_1 = model_out # [bsz, seqlen, 22]
        pred_logits_1[:, :, mu.MASK_TOKEN_INDEX] = self.interpolant.neg_infinity
        pred_logits_1 = pred_logits_1 / self.interpolant._cfg.temp - torch.logsumexp(pred_logits_1 / self.interpolant._cfg.temp, dim=-1, keepdim=True)
        unmasked_indices = (masked_seq != mu.MASK_TOKEN_INDEX)
        pred_logits_1[unmasked_indices] = self.interpolant.neg_infinity
        pred_logits_1[unmasked_indices, masked_seq[unmasked_indices]] = 0
        
        move_chance_s = 1.0 - t_2
        q_xs = pred_logits_1.exp() * d_t
        q_xs[:, :, mu.MASK_TOKEN_INDEX] = move_chance_s
        return q_xs

    def diffusion_qx_calc(self, batch, t1, t2):
        B, L = batch.shape
        V = self.interpolant.num_tokens
        res = torch.zeros((B, L, V), device=batch.device)
        M = 0
        while M < B:
            res[M:M+self.remask_batch] = self.sample_batch_step_qx(batch[M:M+self.remask_batch], t1, t2)
            M += self.remask_batch
        return res
    
    def calc_batched_reward(self, out, batch, reward_oracle, alpha=1.0):
        assert out.shape[0] == batch.shape[0], f"{out.shape[0]} != {batch.shape[0]}"
        M = 0
        N = batch.shape[0]
        while M < N:
            if self.batch_max:
                out[M:M+self.reward_batch] = torch.max(out[M:M+self.reward_batch], alpha * reward_oracle(batch[M:M+self.reward_batch]))
            else:
                out[M:M+self.reward_batch] += alpha * reward_oracle(batch[M:M+self.reward_batch])
            M += self.reward_batch

    def sample_aligned(self):
        print("----------------------------------")
        state = self.initial_state
        num_tokens = state.masked_seq.shape[1]
        reward_traj = []
        curr_iter = 0
        mask = np.zeros(num_tokens)
        r2_traj = []
        while curr_iter < (self.feedback_steps + 1): # Run a total of (self.feedback_steps + 1) iterations so that the last iteration is not a spectral iter
            print("Executing realign")
            self.resampler.initial_state = state
            state = self.resampler.sample_aligned()
            curr_res = state.gen_clean_seq()
            curr_res = curr_res[0]
            seq_str = "".join([ALPHABET[x] for x in curr_res])
            # untargeted = (mask == 0)
            # num_untargeted = int(np.sum(untargeted)) # count the number of tokens that we have not already locked via a previous spectral iteration
            if curr_iter == self.feedback_steps: # or num_untargeted == 0: 
                print(seq_str)
                reward_traj.append(state.calc_reward().item())
                state.spec_reward_traj = list(reward_traj)
                state.r2_traj = list(r2_traj)
                print(f"Reward Trajectory: {[np.round(r, 4) for r in reward_traj]}")
                break

            reward_traj.append(state.calc_reward().item())
            print(f"Previous true reward: {reward_traj[-1]}")

            cv_r2 = 0 # default

            if state.spec_selections is None:
                spec_selections = []
                top_spec_interactions = []
                spec_reward_traj = []
            else:
                spec_selections = list(state.spec_selections)
                top_spec_interactions = list(state.top_spec_interactions)
                spec_reward_traj = list(state.spec_reward_traj)

            if self.feedback_method in ['spectral', 'lasso']:
                all_masks = np.random.choice(2, size=(self.num_masks, num_tokens), p = np.array([self.p, 1-self.p])) # 0.75 prob of being a 0 i.e being "kept"
                
                num_timesteps = self.interpolant._cfg.num_timesteps
                ts = torch.linspace(self.interpolant._cfg.min_t, 1.0, num_timesteps)
                step = 0 # min(floor(self.interpolant._cfg.num_timesteps * self.p), num_timesteps - 2) # bound to ensure target_step + 1 <= n - 1
                t1, t2 = ts[step], ts[step + 1]

                print("Calculating reward estimates...")
                if self.batch_max:
                    alpha = 1.0
                    rewards_torch = torch.full((self.num_masks, ), float("-inf"), device=curr_res.device)
                else:
                    alpha = 1.0 / self.reward_avg_n
                    rewards_torch = torch.zeros((self.num_masks, ), device=curr_res.device)
                M = 0
                its = (self.num_masks + self.mask_batch - 1) // self.mask_batch
                for _ in tqdm(range(its)):
                    batched_states = self.generate_remasked_state_batch(state, 1-all_masks[M:M+self.mask_batch])
                    batched_qxs = self.diffusion_qx_calc(batched_states, t1, t2)
                    batched_qxs[:, :, mu.MASK_TOKEN_INDEX] = 0
                    for _ in range(self.reward_avg_n):
                        sampled_next_states = self.diffusion_mask_infill(batched_states, batched_qxs)
                        self.calc_batched_reward(rewards_torch[M:M+self.mask_batch], sampled_next_states, state.reward_oracle, alpha=alpha)
                    M += self.mask_batch
                rewards = rewards_torch.cpu().numpy()
                
                print("Executing Edit Position Selection...")
                if self.feedback_method == 'spectral':
                    print(" [Fitting Fourier Coefficients]")
                    best_model, cv_r2 = lgboost_fit(all_masks, rewards)
                    fourier_dict = lgboost_to_fourier(best_model)

                    # with open(f"fourier_2kru_it{curr_iter+1}.txt", "w") as f:
                    #     for item in fourier_dict.values():
                    #         f.write(f"{item}\n")

                    sorted_fourier = sorted(fourier_dict.items(), key=lambda item: abs(item[1]), reverse=True)
                    fourier_dict_trunc = dict(sorted_fourier[:2000])
                    # Dictionary items: (bitmask tuple, coefficient)
                    # print(" [Calculating SHR]")
                    # k_vals = np.power(2, np.arange(3, 9))
                    # dsr_vals = []
                    # for top_k in k_vals:
                    #     fourier_dict_trunc_k = dict(sorted_fourier[:top_k])
                    #     tot = 0
                    #     for S, coefficient in fourier_dict_trunc_k.items():
                    #         S_arr = np.array(S)
                    #         # SHR Computation
                    #         S_valid = 1
                    #         nonzero_indices = np.where(S_arr == 1)[0]
                    #         # print(list(nonzero_indices)[1:-1])
                    #         for target_set in power_subset(list(nonzero_indices)): # Iterate through all except empty and full set
                    #             S_arr_subset = S_arr.copy()
                    #             S_arr_subset[list(target_set)] = 0
                    #             if tuple(S_arr_subset) not in fourier_dict_trunc_k:
                    #                 S_valid = 0
                    #                 break
                    #         tot += S_valid
                            
                    #         # DSR Computation

                    #         # S_tot = 0
                    #         # S_mag = np.sum(S_arr)
                    #         # nonzero_indices = np.where(S_arr == 1)[0]
                    #         # for i in nonzero_indices:
                    #         #     S_arr_i = S_arr.copy()
                    #         #     S_arr_i[i] = 0
                    #         #     if tuple(S_arr_i) in fourier_dict_trunc_k:
                    #         #         S_tot += 1
                    #         # if S_mag > 0: 
                    #         #     S_tot /= S_mag
                    #         #     tot += S_tot
                    #         # else:
                    #         #     tot += 1
                    #     tot /= top_k
                    #     dsr_vals.append(tot)
                    # print(f"    SHR values: {[(k, dsr) for k, dsr in zip(k_vals, dsr_vals)]}")

                    print(" [Finding optimal mask]")
                    self.exact_solver.load_fourier_dictionary(fourier_dict_trunc)
                    best_demask = 1 - np.array(self.exact_solver.solve()) # Flip definition of 1 to being "kept"

                    top_spec_interactions.append([])
                    for interactions, coefficient in list(fourier_dict_trunc.items())[:10]: # Saving top interactions)
                        top_interactions = []
                        for i in range(len(interactions)):
                            if interactions[i] == 1: #  and mask[i] == 0
                                top_interactions.append(i)
                        top_spec_interactions[-1].append((top_interactions, round(coefficient, 3))) # extract the index of the interaction instead of the bit map
                elif self.feedback_method == 'lasso':
                    clf = Lasso(alpha=self.lasso_pen)
                    clf.fit(all_masks, rewards)
                    cv_r2 = clf.score(all_masks, rewards)  
                    a = np.array(clf.coef_.flatten().tolist())
                    # b = clf.intercept_
                    lasso_res = a if self.max_spec_order is None else np.argpartition(a, -self.max_spec_order)[-self.max_spec_order:]
                    bad_indices = (a <= 0)
                    best_demask = np.ones_like(a)
                    # Like above, flip definition of a 1
                    best_demask[lasso_res] = 0
                    best_demask[bad_indices] = 1 # Don't include zeros or negatively contributing amino acids in the top-k
                else:
                    raise ValueError("Selection method is invalid")
            elif self.feedback_method == 'exclusion':
                # Mask out each token, and select the top-k tokens which value increases upon removal
                exclusion_rewards = np.zeros_like(mask)
                for i in range(len(mask)):
                    m = np.ones_like(mask)
                    m[i] = 0
                    exclusion_rewards[i] = self.generate_remasked_state(state, m).calc_reward(n=4).item()
                exclusion_res = np.argpartition(exclusion_rewards, -self.max_spec_order)[-self.max_spec_order:] # top-k rewards
                best_demask = np.ones_like(mask)
                best_demask[exclusion_res] = 0
            elif self.feedback_method == 'inclusion':
                # Mask out all but one token, and select the top-k tokens which value increases upon inclusion
                inclusion_rewards = np.zeros_like(mask)
                for i in range(len(mask)):
                    m = np.zeros_like(mask)
                    m[i] = 1
                    inclusion_rewards[i] = self.generate_remasked_state(state, m).calc_reward(n=4).item()
                inclusion_res = np.argpartition(inclusion_rewards, -self.max_spec_order)[-self.max_spec_order:] # top-k rewards
                best_demask = np.zeros_like(mask)
                best_demask[inclusion_res] = 1 # Include the tokens that when included have the highest reward
            else:
                raise ValueError("Selection method is invalid")

            target_features = set()

            for i in range(len(best_demask)):
               if self.feedback_method == 'inclusion':
                   if best_demask[i] == 1:
                    target_features.add(i)
               elif best_demask[i] == 0:
                   target_features.add(i)
            
            spec_reward_traj.append(reward_traj[-1])
            spec_selections.append(list(target_features))

            # print(best_demask)
            mask[:] = best_demask[:] # list(target_features)
            # print(mask)

            print(f"Clean Sequence: {seq_str}")

            tokens = list(seq_str)
            for j in range(num_tokens):
                if mask[j] == 0:
                    tokens[j] = '-'

            if self.feedback_method == 'spectral' or self.feedback_method == 'lasso': r2_traj.append(cv_r2)

            state = self.generate_remasked_state(state, mask)
            state.spec_selections = spec_selections
            state.top_spec_interactions = top_spec_interactions
            state.spec_reward_traj = spec_reward_traj
            state.r2_traj = list(r2_traj)

            if self.feedback_method == 'spectral' or self.feedback_method == 'lasso': print("".join(tokens), f'| r2: {np.round(cv_r2, 4)}', f'Targets: {list(target_features)}')                
            else:  print("".join(tokens), f'Targets: {list(target_features)}')                
                
            curr_iter += 1
        return state
    
class BeamSampler(TreeStateSampler):   
    def __init__(self, sampler_gen, initial_state, depth, child_n, W, save_visual=False, soft=False, reward_threshold=None):
        # Parameter validation
        assert type(W) is int, "W must be type 'int"
        assert W > 0, "W must be a positive integer"
        assert W <= child_n, "W must be less than or equal to child_n"
        self.W = W
        self.save_visual = save_visual
        self.soft = soft
        super().__init__(sampler_gen, initial_state, depth, child_n)

    def sample_aligned(self):
        states = [self.initial_state]
        gen_states = []
        num_states = []
        num_gens = []
        labels = []
        for i in range(self.depth - 1):
            if self.save_visual:
                gen_states.append([])
                num_states.append([])
                num_gens.append([])
                labels.append([])

            next_states = []
            for state in states:
                assert isinstance(state, AlignSamplerState), "State must be instance of AlignSamplerState"
                # if state.return_early(): 
                #     next_states.append(state)
                #     continue
                # else:
                w_ = self.W if i == 0 else 1
                n_ = self.child_n if i == 0 else self.child_n // self.W
                sampler = self.sampler_gen(state, n_)
                bon_sampler = BONSampler(sampler=sampler, W=w_, soft=self.soft)
                samples, top_indices, rewards = bon_sampler.sample_aligned() # type: ignore
                next_states.extend([samples.get_state(i.item()) for i in top_indices])
                if self.save_visual:
                    gen_states[-1].append([int(k.item()) for k in top_indices])
                    num_states[-1].append(n_)
                    labels[-1].append([f"{r.item():.1e}" for r in rewards])
                     
            states = next_states
        
        max_state = max(states, key=lambda s : s.calc_reward(select_argmax=True).item())

        if self.save_visual:
            max_state_visual = 4 # len(num_states) - 2
            self.gen_tree_visual(gen_states, num_states[:max_state_visual], labels)

        max_state.pred_seq = None

        return max_state
