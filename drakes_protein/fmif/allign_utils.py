import inspect
import torch
from torch.distributions import Categorical
import numpy as np

class AlignSampler():
    def __init__(self):
        # 'sample' function must have no arguments
        subclass_method = self.__class__.__dict__.get("sample")
        if subclass_method:
            sig = inspect.signature(subclass_method)
            if len(sig.parameters) != 1:
                raise TypeError(f"{self.__class__.__name__}.sample() must be defined without additional parameters.")        
    
    def sample_aligned(self, reward_oracle):
        raise NotImplemented

class AlignSamplerState():
        def __init__(self, state_value):
            assert type(state_value) is torch.Tensor, f"Got type {type(state_value)} but State value must be of type 'torch.Tensor'"
            self.state_value = state_value
            
        def get_state_value(self):
            return self.state_value
        
        def set_state_value(self, state_value):
            self.state_value = state_value

class BONSampler(AlignSampler):
    def __init__(self, sampler, n, W, bon_batch_size):
        # Parameter validation
        assert type(n) is int, "n must be type 'int'"
        assert type(W) is int, "W must be type 'int"
        assert type(bon_batch_size) is int, "bon_batch_size must be type 'int'"
        assert n > 0, "n must be a positive integer"
        assert bon_batch_size > 0, "bon_batch_size must be a positive integer"
        assert W > 0, "W must be a positive integer"
        assert W <= n, "W must be less than or equal to n"
        self.sampler = sampler
        self.n = n # number of samples to generate
        self.W = W # top W results returned
        self.bon_batch_size = bon_batch_size
        super().__init__()

    def sample_aligned(self, reward_oracle):

        sample_mat, reward_mat, curr_reward, best_sample = None, None, None, None
        target_n = self.n
        while target_n > 0:
            batch_size = min(target_n, self.bon_batch_size)
            target_n -= batch_size
            for i in range(batch_size):
                # Sample from sampler
                sample = self.sampler()
                assert isinstance(sample, AlignSamplerState), f"{type(sample)} is not an instance of 'AlignSamplerState'"
                # Sample shape validation
                state_val = sample.get_state_value()
                assert len(state_val.shape) <= 2, "State value shape must be length 1 or 2: (sample_shape,) or (num_samples, sample_shape)"
                if len(state_val.shape) == 2:
                    num_samples, sample_shape = state_val.shape
                elif len(state_val.shape) == 1:
                    num_samples, sample_shape = 1, state_val.shape[0]
                else:
                    num_samples, sample_shape = 1, 1
                # Initialize sample and reward matrices
                if sample_mat is None:
                    sample_mat = torch.zeros((batch_size, num_samples, sample_shape), device=state_val.device, dtype=state_val.dtype)
                    reward_mat = torch.full((batch_size, num_samples), float('-inf'), device=torch.device('cpu'))
                    curr_reward = torch.full((num_samples, ), float('-inf'), device=torch.device('cpu'))
                    if num_samples > 1:
                        best_sample = torch.zeros(state_val.shape, device=state_val.device, dtype=state_val.dtype)
                # Store samples
                reward_mat[i] = reward_oracle(sample)
                sample_mat[i] = state_val
            best_rewards = torch.argmax(reward_mat, dim=0)
            # top_rewards, top_indices = torch.topk(reward_mat, self.W, dim=0) => concatenate everything globally and then to one top-k
            for i in range(num_samples):
                reward = reward_mat[best_rewards[i]][i]
                if reward > curr_reward[i]:
                    curr_reward[i] = reward
                    # Sample() yields 1 sample per call
                    if num_samples == 1:
                        best_sample = sample_mat[best_rewards[i]][i].clone()
                    # Sample() yields multiple samples per call
                    else:
                        best_sample[i] = sample_mat[best_rewards[i]][i].clone()
        sample.set_state_value(best_sample) # Currently only returns top-1, not top-W
        return [sample]
    
class TreeStateSampler(AlignSampler):
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
        super().__init__()

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
        self.C = C
        self.max_iter = max_iter
        super().__init__(sampler_gen, initial_state, depth, child_n)

    def select_best_ucb(self, node):
        candidates = [self.select_best_ucb(c) for c in node.children]
        if len(node.children) < self.child_n:
            candidates.append(node) # Don't allow nodes to spawn more than child_n children
        best_ucb = max(candidates, key=lambda c : c.S) # Select node with best UCB
        return best_ucb

    def sample_aligned(self, reward_oracle):
        # Initialize node to initial state
        root = self.Node(state=self.initial_state, value=reward_oracle(self.initial_state), C=self.C)
        # MCTS
        for _ in range(self.max_iter):
            # Selection
            best_node = self.select_best_ucb(root)
            if best_node.depth == self.depth:
                return best_node.state # Return once the best node is at the final depth
            # Expansion
            sampler = self.sampler_gen(best_node.state)
            state = sampler.sample()
            # Simulation
            reward = reward_oracle(state)
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

    def sample_aligned(self, reward_oracle):
        states = [self.initial_state]
        sampler = self.sampler_gen(self.initial_state)
        bon_sampler = BONSampler(sampler=sampler, W=self.W, n=self.child_n, bon_batch_size=1) 
        for _ in range(self.depth):
            next_states = []
            for state in states:
                sampler = self.sampler_gen(state)
                bon_sampler = BONSampler(sampler=sampler, W=self.W, n=self.child_n, bon_batch_size=1)
                next_states += bon_sampler.sample_aligned(reward_oracle=reward_oracle)
            states = next_states
        return max(states, key=reward_oracle)
    
# distr = torch.tensor([0.3, 0.5, 0.2])
# reward_oracle = lambda x : x
# prob = Categorical(probs=distr)
# sampler = lambda : prob.sample()
# a = BONSampler(sampler, n=100, W=1, bon_batch_size=1)
# print([a.sample_aligned(reward_oracle).item() for _ in range(10)])