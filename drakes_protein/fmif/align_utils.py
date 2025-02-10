import inspect
import torch
from torch.distributions import Categorical
import numpy as np

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
        return [samples[i] for i in top_indices]
    
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
    def __init__(self, sampler_gen, initial_state, depth, child_n, W):
        # Parameter validation
        assert type(W) is int, "W must be type 'int"
        assert W > 0, "W must be a positive integer"
        self.W = W
        super().__init__(sampler_gen, initial_state, depth, child_n)

    def sample_aligned(self):
        states = [self.initial_state]
        sampler = self.sampler_gen(self.initial_state)
        bon_sampler = BONSampler(sampler=sampler, W=self.W, n=self.child_n) 
        for _ in range(self.depth):
            next_states = []
            for state in states:
                assert isinstance(state, AlignSamplerState), "State must be instance of AlignSamplerState"
                sampler = self.sampler_gen(state)
                if self.child_n > 1:
                    bon_sampler = BONSampler(sampler=sampler, W=self.W, n=self.child_n)
                    next_states += bon_sampler.sample_aligned()
                else:
                    next_states.append(sampler())
            states = next_states
        return max(states, key=lambda s : s.calc_reward())
    
# distr = torch.tensor([0.3, 0.5, 0.2])
# reward_oracle = lambda x : x
# prob = Categorical(probs=distr)
# sampler = lambda : prob.sample()
# a = BONSampler(sampler, n=100, W=1)
# print([a.sample_aligned(reward_oracle).item() for _ in range(10)])