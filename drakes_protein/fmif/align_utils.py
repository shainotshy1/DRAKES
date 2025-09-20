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
    def __init__(self, sampler_gen, initial_state, depth, feedback_steps, state_builder, resampler):
        # Parameter validation
        assert type(depth) is int, "depth must be type 'int'"
        assert depth > 0, "depth must be a positive integer"
        self.sampler_gen = sampler_gen
        self.initial_state = initial_state
        self.depth = depth
        self.feedback_steps = feedback_steps
        self.state_builder = state_builder
        self.resampler = resampler
        

    #TODO: After having generated a sample, generate N masks on the sequence to train a ProxySPEX function 
    #      and figure out where the interactions are by looking at the top-k coefficients
    #TODO: Either make N a hyperparameter or repeatedly train the function till we achieve high faithfulness (like done in ProxySPEX)

    def generate_remasked_state(self, state, mask):
        # num_tokens = mask.shape[0]
        # num_untargeted = np.sum(mask)
        masked_seq = state.gen_clean_seq()
        print(masked_seq)
        masked_seq[0][mask == 1] = mu.MASK_TOKEN_INDEX
        # print("MASK")
        # print(masked_seq)
        new_step = 0 #int(floor((1 - num_untargeted * 1.0 / num_tokens) * (depth - 1)))

        state = self.state_builder(masked_seq, new_step, state) # back track in diffusion process to when mask would be masked like so
        # print("PROBS")
        # print(state.q_xs)
        return state

    def sample_aligned(self):
        print("----------------------------------")
        # TODO: nest loop in multiple "Interaction" iterations
# For more liited number of samples, tlak about influence scores
# Address MCTS in the paper to predict review
# Can consider decreasing sampling as we repeat iterations
# Make calc reward as expected value instead of argmax
        state = self.initial_state
        num_tokens = state.masked_seq.shape[1]
        reward_traj = []
        curr_iter = 0
        mask = np.ones(num_tokens)
        while curr_iter < (self.feedback_steps + 1): # Run a total of (self.feedback_steps + 1) iterations so that the last iteration is not a spectral iter
            print("Executing realign")
            self.resampler.initial_state = state
            state = self.resampler.sample_aligned()
            curr_res = state.gen_clean_seq()
            curr_res = curr_res[0]
            seq_str = "".join([ALPHABET[x] for x in curr_res])
            untargeted = (mask == 1)
            num_untargeted = int(np.sum(untargeted)) # count the number of tokens that we have not already locked via a previous spectral iteration
            if curr_iter == self.feedback_steps or num_untargeted == 0: 
                print(seq_str)
                reward_traj.append(state.calc_reward().item())
                print(f"Reward Trajectory: {[np.round(r, 4) for r in reward_traj]}")
                break

            # if curr_iter == self.feedback_steps:
            #     seq_str = "".join([ALPHABET[x] for x in curr_res])
            #     reward_traj.append(state.calc_reward().item())
            #     print(f"Reward Trajectory: {[np.round(r, 4) for r in reward_traj]}")
            #     return state
            # else:
            reward_traj.append(state.calc_reward().item())
            print(f"Previous true reward: {reward_traj[-1]}")
            # print(seq_str, state.calc_reward().item())
            # print()

            # mask_samples = torch.bernoulli(torch.full((num_masks, num_untargeted), p, device=device)).int() # type: ignore
            # all_masks = torch.zeros(size=(num_masks, num_tokens), device=device, dtype=mask_samples.dtype).repeat(num_masks, 1)

            # TODO: switch to reward as predicted reward rather than reward of masked sequence
            # TODO: try instead of locking the best ones, run sampling on the coefficients with the highest interactions, regardless of positive or negative coefficient
            num_masks = 500
            # p1 = 0.3
            # p2 = 0.4
            # alpha = num_untargeted / num_tokens
            # p = p1 * alpha + (1 - alpha) * p2
            p = 0.25

            mask_samples = np.random.choice(2, size=(num_masks, num_tokens), p = np.array([1-p, p]))
            all_masks = mask_samples * untargeted.astype(mask_samples.dtype) # zero out currently targeted

            rewards_lst = []
            for m in all_masks:
                rewards_lst.append(self.generate_remasked_state(state, m).calc_reward(n=reward_avg_n).item())
            rewards = np.array(rewards_lst)
            # for i in range(3):
            #     tokens = list(seq_str)
            #     for j in range(num_tokens):
            #         if all_masks[i,j]:
            #             tokens[j] = ' '
            #     print("".join(tokens), rewards[i])                

            # Spectral Method

            best_model, cv_r2 = lgboost_fit(all_masks, rewards)
            # print(f'CV r2: {cv_r2}')

            # Algorithm: select top parent and its children
            top_interactions = 10
            fourier_dict = lgboost_to_fourier(best_model)
            fourier_dict_trunc = dict(sorted(fourier_dict.items(), key=lambda item: item[1], reverse=True)[:(top_interactions + 1)])
            
            # tmp = dict(sorted(fourier_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:(top_interactions * 2 + 1)])
            # for k in tmp:
            #     nonzero_pos, = np.where(np.array(k) == 1)
            #     if len(nonzero_pos) == 0: continue
            #     descr = "("
            #     for i in range(len(nonzero_pos) - 1):
            #         descr += f"{nonzero_pos[i]}, "
            #     if len(nonzero_pos) > 0: descr += str(nonzero_pos[-1])
            #     descr += ")"                    
            #     print(descr, tmp[k])
            # print("-----")
            # fourier_iter = iter(fourier_dict_trunc)

            # top_coefficient = next(fourier_iter, None)
            # if top_coefficient is None: # Catch no iteractions => exit early
            #     curr_iter = self.feedback_steps
            #     continue

            # if sum(top_coefficient) == 0:
            #     top_coefficient = next(fourier_iter, None)
            #     if top_coefficient is None: # Catch no iteractions => exit early
            #         curr_iter = self.feedback_steps
            #         continue

            target_features = set()
            total_found = 0
            for k in fourier_dict_trunc:
                if fourier_dict_trunc[k] < 0: break # no more positive coefficients left
                nonzero_pos, = np.where(np.array(k) == 1)
                if len(nonzero_pos) == 0: continue
                else: total_found += 1
                target_features.update(nonzero_pos)
                # descr = "("
                # for i in range(len(nonzero_pos) - 1):
                #     descr += f"{nonzero_pos[i]}, "
                # if len(nonzero_pos) > 0: descr += str(nonzero_pos[-1])
                # descr += ")"
                # print(descr, fourier_dict_trunc[k])
                if total_found == top_interactions: break
            # print(f"SPECTRAL targets: {sorted(list(target_features))}")

            mask[list(target_features)] = 0
            # num_untargeted = num_tokens - int(np.sum(mask))
            # if num_untargeted == 0: # just in case handling
            #     curr_iter = self.feedback_steps
            #     continue

            tokens = list(seq_str)
            for j in range(num_tokens):
                if mask[j] == 1:
                    tokens[j] = '-'

            state = self.generate_remasked_state(state, mask)

            print("".join(tokens), f'| r2: {np.round(cv_r2, 4)}', f'Targets: {list(target_features)}')                

            # LASSO Baseline
            # max_solution_order = len(target_features)
            # clf = Lasso(alpha=0.005)
            # clf.fit(all_masks, rewards)
            # a = np.array(clf.coef_.flatten().tolist())
            # b = clf.intercept_
            # lasso_res = np.argpartition(a, -max_solution_order)[-max_solution_order:]
            # print(f"LASSO targets: {sorted(lasso_res)}")
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
