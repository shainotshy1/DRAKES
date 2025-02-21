import inspect
import torch
from torch.distributions import Categorical
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class AlignSamplerState():
    def calc_reward(self):
        raise NotImplemented

class BONSampler():
    def __init__(self, sampler, n, W):
        # Parameter validation
        assert type(n) is int, "n must be type 'int'"
        assert type(W) is int, "W must be type 'int"
        assert n > 0, "n must be a positive integer"
        assert W > 0, "W must be a positive integer"
        assert W <= n, "W must be less than or equal to n"
        self.sampler = sampler
        self.n = n # number of samples to generate
        self.W = W # top W results returned
        super().__init__()

    def sample_aligned(self):
        samples = [self.sampler() for _ in range(self.n)]
        for s in samples:
            assert isinstance(s, AlignSamplerState), "Sample must be instance of AlignSamplerState"
        rewards = [s.calc_reward() for s in samples]
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

    def gen_tree_visual(self, gen_states, num_states, num_gens, labels):
        G = nx.DiGraph()
        color_map = {}

        G.add_node(G.size(), label=labels[0][0][0])
        color_map[G.size()] = '#83F061'

        prev_layer_parents = [0]
        for i, level in enumerate(num_states):
            new_prev_layer = []
            gen_state = gen_states[i]
            for j, n in enumerate(level):
                parent = prev_layer_parents[j]
                new_nodes = []
                for k in range(n):
                    new_node = G.size() + 1
                    new_nodes.append(new_node)
                    G.add_node(new_node, label=labels[i][j][k])
                    G.add_edge(parent, new_node)
                w = num_gens[i][j]
                new_parents = []
                for k in range(w):
                    color_map[new_nodes[gen_state[k]]] = '#83F061'
                    new_parents.append(new_nodes[gen_state[k]])
                new_prev_layer.extend(sorted(new_parents))
                gen_state = gen_state[w:]
            prev_layer_parents = new_prev_layer
            if i == 3:
                break
        
        for i, layer in enumerate(reversed(list(nx.topological_generations(G)))):
            for n in layer:
                G.nodes[n]["layer"] = i
        
        labels = {node: data["label"] for node, data in G.nodes(data=True)}
        node_colors = [color_map.get(node, "#2BB6F0") for node in G.nodes()]
        pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color=node_colors, font_size=7, arrows=False)
        plt.savefig('tree.png')



class MCTSSampler(TreeStateSampler):   
    class Node():
        def __init__(self, state, state_value, C):
            self.state = state # State that the node represents
            self.state_value = state_value # Value of the node state
            self.S = np.inf # UCB value of node
            self.v = 0 # Empirical mean of node
            self.n = 0 # Number of times visited
            self.depth = 0 # Default depth is 0 since root
            self.children = [] # Children nodes for MCTS
            self.parent = None # None represents root node
            self.C = C # MCTS parameter

        def is_root(self):
            return self.parent is None

        def is_leaf(self):
            return len(self.children) == 0

        def add_child(self, child):
            child.depth = self.depth + 1
            child.parent = self
            self.children.append(child)

        def backpropogate(self):
            self.n += 1
            if self.is_leaf():
                self.v = self.state_value
            self.v = sum([c.v * c.n / self.n for c in self.children]) + self.state_value / self.n
            if self.parent:
                self.parent.backpropogate()
            self.update_value()

        def update_value(self):
            if self.parent is None:
                self.S = np.inf # default UCB value
            else:
                self.S = self.v + self.C * np.sqrt(np.log(self.parent.n) / self.n) # Update UCB Value

    def __init__(self, sampler_gen, initial_state, depth, child_n, C, max_iter=1000):
        # Parameter validation
        assert C > 0, "C must be value greater than 0"
        assert isinstance(initial_state, AlignSamplerState), "Initial state must be instance of AlignSamplerState"
        self.C = C
        self.max_iter = max_iter
        super().__init__(sampler_gen, initial_state, depth, child_n)

    def select_best_ucb(self, node):
        candidates = [self.select_best_ucb(c) for c in node.children]
        if len(node.children) < self.child_n:
            candidates.append(node) # Don't allow nodes to spawn more than child_n children
        best_ucb = max(candidates, key=lambda c : c.S) # Select node with best UCB
        return best_ucb

    def sample_aligned(self):
        # Initialize node to initial state
        root = self.Node(state=self.initial_state, value=self.initial_state.calc_reward(), C=self.C)
        # MCTS
        for _ in range(self.max_iter):
            # Selection
            best_node = self.select_best_ucb(root)
            if best_node.depth == self.depth:
                return best_node.state # Return once the best node is at the final depth
            # Expansion
            sampler = self.sampler_gen(best_node.state)
            state = sampler.sample()
            assert isinstance(state, AlignSamplerState), "State must be instance of AlignSamplerState"
            # Simulation
            reward = state.calc_reward()
            new_child = self.Node(state=state, value=reward) # TODO: construct the sampler only one time - the first time best_node is sampled from
            # Backpropogation
            best_node.add_child(new_child)
            new_child.backpropogate()

        # If unable to converge, return the best node so far
        best_node = self.select_best_ucb(root)
        best_state = best_node.state
        return best_state

class BeamSampler(TreeStateSampler):   
    def __init__(self, sampler_gen, initial_state, depth, child_n, W, save_visual=False):
        # Parameter validation
        assert type(W) is int, "W must be type 'int"
        assert W > 0, "W must be a positive integer"
        self.W = W
        self.save_visual = save_visual
        super().__init__(sampler_gen, initial_state, depth, child_n)

    def sample_aligned(self):
        states = [self.initial_state]
        gen_states = []
        num_states = []
        num_gens = []
        labels = []
        sampler = self.sampler_gen(self.initial_state)
        bon_sampler = BONSampler(sampler=sampler, W=self.W, n=self.child_n) 
        for i in range(self.depth - 1):
            if self.save_visual:
                gen_states.append([])
                num_states.append([])
                num_gens.append([])
                labels.append([])

            next_states = []
            for state in states:
                assert isinstance(state, AlignSamplerState), "State must be instance of AlignSamplerState"
                sampler = self.sampler_gen(state)

                w_ = self.W if i == 0 else 1
                n_ = self.child_n if i == 0 else self.child_n // self.W
                bon_sampler = BONSampler(sampler=sampler, n=n_, W=w_)

                samples, top_indices, rewards = bon_sampler.sample_aligned()
                next_states += [samples[i] for i in top_indices]

                if self.save_visual:
                    gen_states[-1].extend([int(k.item()) for k in top_indices])
                    num_states[-1].append(n_)
                    num_gens[-1].append(w_)
                    labels[-1].append([f"{r.item():.1e}" for r in rewards])
                    
            states = next_states
        
        if self.save_visual:
            self.gen_tree_visual(gen_states, num_states, num_gens, labels)

        max_state = max(states, key=lambda s : s.calc_reward())

        if self.save_visual:
            gen_states[-1].append([states.index(max_state)])
        #print(gen_states)
        return max_state

    
# distr = torch.tensor([0.3, 0.5, 0.2])
# reward_oracle = lambda x : x
# prob = Categorical(probs=distr)
# sampler = lambda : prob.sample()
# a = BONSampler(sampler, n=100, W=1)
# print([a.sample_aligned(reward_oracle).item() for _ in range(10)])