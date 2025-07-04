import torch
from torch import nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tqdm import tqdm

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
    def __init__(self, sampler, n, W, soft=False):
        # Parameter validation
        assert type(n) is int, "n must be type 'int'"
        assert type(W) is int, "W must be type 'int"
        assert n > 0, "n must be a positive integer"
        assert W > 0, "W must be a positive integer"
        assert W <= n, "W must be less than or equal to n"
        self.sampler = sampler
        self.n = n # number of samples to generate
        self.W = W # top W results returned
        self.soft = soft # Soft max sampling used instead of argma
        if self.soft:
            self.sm = nn.Softmax()
        super().__init__()

    def sample_aligned(self):
        samples = self.sampler()#[self.sampler() for _ in range(self.n)]
        # for s in samples:
        assert isinstance(samples, AlignSamplerState), "Sample must be instance of AlignSamplerState"
        rewards = samples.calc_reward()#[s.calc_reward() for s in samples]
        if self.soft:
            sm_rewards = self.sm(torch.tensor(rewards))
            top_indices = torch.multinomial(sm_rewards, num_samples=self.W, replacement=True)
        else:
            _, top_indices = torch.topk(torch.tensor(rewards), self.W, dim=0)
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

        G.add_node(G.size(), label=labels[0][0][0])

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
                    new_node = G.size() + 1
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

        for i, layer in enumerate(reversed(list(nx.topological_generations(G)))):
            for n in layer:
                G.nodes[n]["layer"] = i
        max_layer = i + 1
        
        pad = (global_max - global_min) / 4
        min_val, max_val = global_min - pad, global_max + pad
        norm = plt.Normalize(min_val, max_val)
        node_colors = [color_grad(1 - norm(float(data["label"]))) for _, data in G.nodes(data=True)]
        labels = {node: data["label"] for node, data in G.nodes(data=True)}

        fig_width = 6
        fig_height = max_layer * 1.2
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
        for _ in tqdm(range(self.depth - 1)):
            assert isinstance(state, AlignSamplerState), "State must be instance of AlignSamplerState"
            sampler = self.sampler_gen(state, n=self.child_n)
            if state.return_early():
                samples = sampler()
            else:
                samples = sampler()#[sampler() for _ in range(self.child_n)]
            state = self.opt_selector(samples)
        return state
         
class BeamSampler(TreeStateSampler):   
    def __init__(self, sampler_gen, initial_state, depth, child_n, W, save_visual=False, soft=False, reward_threshold=None):
        # Parameter validation
        assert type(W) is int, "W must be type 'int"
        assert W > 0, "W must be a positive integer"
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
        for i in tqdm(range(self.depth - 1)):
            if self.save_visual:
                gen_states.append([])
                num_states.append([])
                num_gens.append([])
                labels.append([])

            next_states = []
            for state in states:
                assert isinstance(state, AlignSamplerState), "State must be instance of AlignSamplerState"
                sampler = self.sampler_gen(state, self.child_n)

                w_ = self.W if i == 0 else 1
                n_ = self.child_n if i == 0 else self.child_n // self.W
                bon_sampler = BONSampler(sampler=sampler, n=n_, W=w_, soft=self.soft)
                samples, top_indices, rewards = bon_sampler.sample_aligned()
                next_states += [samples.get_state(i) for i in top_indices]
                if self.save_visual:
                    gen_states[-1].append([int(k.item()) for k in top_indices])
                    num_states[-1].append(n_)
                    labels[-1].append([f"{r.item():.1e}" for r in rewards])
                    
            states = next_states
        
        if self.save_visual:
            self.gen_tree_visual(gen_states, num_states, labels)

        max_state = max(states, key=lambda s : s.calc_reward())

        if self.save_visual:
            gen_states[-1].append([states.index(max_state)])
        return max_state
