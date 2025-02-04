import inspect
import torch
from torch.distributions import Categorical

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

class BONSampler(AlignSampler):
    def __init__(self, sampler, n, bon_batch_size):
        # Parameter validation
        assert type(n) is int, "n must be type 'int'"
        assert type(bon_batch_size) is int, "bon_batch_size must be type 'int'"
        assert n > 0, "n must be a positive integer"
        assert bon_batch_size > 0, "bon_batch_size must be a positive integer"
        self.sampler = sampler
        self.n = n
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
                # Sample shape validation
                assert len(sample.shape) <= 2, "Sample shape must be length 1 or 2: (sample_shape,) or (num_samples, sample_shape,)"
                if len(sample.shape) == 2:
                    num_samples, sample_shape = sample.shape
                elif len(sample.shape) == 1:
                    num_samples, sample_shape = 1, sample.shape[0]
                else:
                    num_samples, sample_shape = 1, 1
                # Initialize sample and reward matrices
                if sample_mat is None:
                    sample_mat = torch.zeros((batch_size, num_samples, sample_shape), device=sample.device, dtype=sample.dtype)
                    reward_mat = torch.full((batch_size, num_samples), float('-inf'), device=torch.device('cpu'))
                    curr_reward = torch.full((num_samples, ), float('-inf'), device=torch.device('cpu'))
                    if num_samples > 1:
                        best_sample = torch.zeros(sample.shape, device=sample.device, dtype=sample.dtype)
                # Store samples
                reward_mat[i] = reward_oracle(sample)
                sample_mat[i] = sample
            best_rewards = torch.argmax(reward_mat, dim=0)
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
        return best_sample
    
class TreeStateSampler(AlignSampler):
    def __init__(self, state_fn, sampler, depth, child_n):
        # Parameter validation
        assert type(depth) is int, "depth must be type 'int'"
        assert depth > 0, "depth must be a positive integer"
        assert type(child_n) is int, "child_n must be type 'int'"
        assert child_n > 1, "child_n must be a positive integer"
        self.state_fn = state_fn
        self.sampler = sampler
        self.depth = depth
        self.child_n = child_n
        super().__init__()

class MCTSSampler(AlignSampler):   
    def __init__(self, state_fn, sampler, depth, child_n, C):
        # Parameter validation
        assert C > 0, "C must be value greater than 0"
        self.C = C
        super().__init__(state_fn, sampler, depth, child_n)

    def sample_aligned(self, reward_oracle):
        return self.sample()

class BeamSampler(AlignSampler):   
    def __init__(self, state_fn, sampler, depth, child_n, W):
        # Parameter validation
        assert type(W) is int, "W must be type 'int"
        assert W > 0, "W must be a positive integer"
        self.W = W
        super().__init__(state_fn, sampler, depth, child_n)

    def sample_aligned(self, reward_oracle):
        return self.sample()
    
distr = torch.tensor([0.3, 0.5, 0.2])
reward_oracle = lambda x : x
prob = Categorical(probs=distr)
sampler = lambda : prob.sample()
a = BONSampler(sampler, n=100, bon_batch_size=1)
print([a.sample_aligned(reward_oracle).item() for _ in range(10)])