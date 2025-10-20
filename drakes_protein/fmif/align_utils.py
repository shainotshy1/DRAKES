import torch
from torch import nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tree_spex import lgboost_fit, lgboost_to_fourier, lgboost_tree_to_fourier, ExactSolver # type:ignore
from protein_oracle.data_utils import ALPHABET, ALPHABET_WITH_MASK
from sklearn.linear_model import Lasso
import model_utils as mu # type:ignore
from math import floor

class AlignSamplerState():
    def calc_reward(self):
        raise NotImplemented
    
    def get_num_states(self):
        raise NotImplemented
    
    def get_state(self, i):
        raise NotImplemented
    
    def return_early(self):
        return False

reward_avg_n = 2 # parameter use in all sampling algorithms, so it is fair

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
        rewards = samples.calc_reward(n=reward_avg_n) # type: ignore
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
         
class InteractionSampler():
    def __init__(self, sampler_gen, initial_state, depth, feedback_steps, max_spec_order, feedback_method, state_builder, resampler, lasso_pen=0.0):
        # Parameter validation
        assert type(depth) is int, "depth must be type 'int'"
        assert depth > 0, "depth must be a positive integer"
        self.sampler_gen = sampler_gen
        self.initial_state = initial_state
        self.depth = depth
        self.feedback_steps = feedback_steps
        self.state_builder = state_builder
        self.resampler = resampler
        self.max_spec_order = max_spec_order
        self.feedback_method = feedback_method
        self.lasso_pen = lasso_pen
        self.exact_solver = ExactSolver(maximize=True, max_solution_order=max_spec_order)
        
    def generate_remasked_state(self, state, mask):
        masked_seq = state.gen_clean_seq()
        masked_seq[0][mask == 0] = mu.MASK_TOKEN_INDEX
        new_step = 1 #int(floor((1 - num_untargeted * 1.0 / num_tokens) * (depth - 1)))

        state = self.state_builder(masked_seq, new_step, state) # back track in diffusion process to when mask would be masked like so
        return state

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
            untargeted = (mask == 0)
            num_untargeted = int(np.sum(untargeted)) # count the number of tokens that we have not already locked via a previous spectral iteration
            if curr_iter == self.feedback_steps or num_untargeted == 0: 
                print(seq_str)
                reward_traj.append(state.calc_reward().item())
                state.spec_reward_traj = list(reward_traj)
                state.r2_traj = list(r2_traj)
                print(f"Reward Trajectory: {[np.round(r, 4) for r in reward_traj]}")
                break

            reward_traj.append(state.calc_reward().item())
            print(f"Previous true reward: {reward_traj[-1]}")

            num_masks = 500
            p = 0.25

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

                mask_samples = np.random.choice(2, size=(num_masks, num_tokens), p = np.array([p, 1-p])) # 0.25 prob of being a 1 i.e being "kept"
                all_masks = mask_samples * untargeted.astype(mask_samples.dtype) # zero out currently targeted

                rewards_lst = []
                for m in all_masks:
                    rewards_lst.append(self.generate_remasked_state(state, m).calc_reward(n=reward_avg_n).item())
                rewards = np.array(rewards_lst)

                if self.feedback_method == 'spectral':
                    best_model, cv_r2 = lgboost_fit(all_masks, rewards)
                    fourier_dict = lgboost_to_fourier(best_model)
                    fourier_dict_trunc = dict(sorted(fourier_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:2000])
                    self.exact_solver.load_fourier_dictionary(fourier_dict_trunc)
                    best_demask = self.exact_solver.solve()

                    top_spec_interactions.append([])
                    for interactions, coefficient in list(fourier_dict_trunc.items())[:10]: # Saving top interactions)
                        top_interactions = []
                        for i in range(len(interactions)):
                            if interactions[i] == 1 and mask[i] == 0:
                                top_interactions.append(i)
                        top_spec_interactions[-1].append((top_interactions, round(coefficient, 3))) # extract the index of the interaction instead of the bit map

                elif self.feedback_method == 'lasso':
                    clf = Lasso(alpha=self.lasso_pen)
                    clf.fit(all_masks, rewards)
                    a = np.array(clf.coef_.flatten().tolist())
                    # b = clf.intercept_
                    lasso_res = np.argpartition(a, -self.max_spec_order)[-self.max_spec_order:]
                    bad_indices = (a <= 0)
                    best_demask = np.zeros_like(a)
                    best_demask[lasso_res] = 1
                    best_demask[bad_indices] = 0 # Don't include zeros or negatively contributing aminoa acids in the top-k
                else:
                    raise ValueError("Selection method is invalid")
            elif self.feedback_method == 'exclusion':
                # Mask out each token, and select the top-k tokens which value decreases upon removal
                exclusion_rewards = np.full(mask.shape, np.inf)
                for i in range(len(mask)):
                    if mask[i] == 0: # Non-locked amino acids
                        m = np.ones_like(mask)#.copy()
                        m[i] = 0
                        exclusion_rewards[i] = self.generate_remasked_state(state, m).calc_reward(n=5).item()
                exclusion_res = np.argpartition(exclusion_rewards, self.max_spec_order)[:self.max_spec_order] # bottom k rewards
                best_demask = np.zeros_like(mask)
                best_demask[exclusion_res] = 1
            elif self.feedback_method == 'inclusion':
                # Mask out all but one token (keeping the currently locked amino acids locked), and select the top-k tokens which value increases upon removal
                exclusion_rewards = np.full(mask.shape, -np.inf)
                for i in range(len(mask)):
                    if mask[i] == 0: # Non-locked amino acids
                        m = mask.copy()
                        m[i] = 1
                        exclusion_rewards[i] = self.generate_remasked_state(state, m).calc_reward(n=5).item()
                exclusion_res = np.argpartition(exclusion_rewards, -self.max_spec_order)[-self.max_spec_order:] # top k rewards
                best_demask = np.zeros_like(mask)
                best_demask[exclusion_res] = 1
            else:
                raise ValueError("Selection method is invalid")

            target_features = set()

            for i in range(len(best_demask)):
                if best_demask[i] == 1 and mask[i] == 0: # if the demask is 1 then we want to KEEP that amino acid; mask[i] == 0 => currently it is not locked
                    target_features.add(i)
            
            spec_reward_traj.append(reward_traj[-1])
            spec_selections.append(list(target_features))

            mask[list(target_features)] = 1

            tokens = list(seq_str)
            for j in range(num_tokens):
                if mask[j] == 0:
                    tokens[j] = '-'

            if self.feedback_method == 'spectral': r2_traj.append(cv_r2)

            state = self.generate_remasked_state(state, mask)
            state.spec_selections = spec_selections
            state.top_spec_interactions = top_spec_interactions
            state.spec_reward_traj = spec_reward_traj
            state.r2_traj = list(r2_traj)

            if self.feedback_method == 'spectral': print("".join(tokens), f'| r2: {np.round(cv_r2, 4)}', f'Targets: {list(target_features)}')                
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
        
        max_state = max(states, key=lambda s : s.calc_reward(n=reward_avg_n).item())

        if self.save_visual:
            max_state_visual = 4 # len(num_states) - 2
            self.gen_tree_visual(gen_states, num_states[:max_state_visual], labels)

        max_state.pred_seq = None

        return max_state
