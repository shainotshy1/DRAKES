import inspect
import torch
from torch.distributions import Categorical

class BONSampler():   
    def __init__(self):
        # 'sample' function must have no arguments
        subclass_method = self.__class__.__dict__.get("sample")
        if subclass_method:
            sig = inspect.signature(subclass_method)
            if len(sig.parameters) != 1:
                raise TypeError(f"{self.__class__.__name__}.sample() must be defined without additional parameters.")        
    
    def sample(self):
        raise NotImplemented

    def sample_aligned(self, reward_oracle, n=1, bon_batch_size=1):
        # Parameter validation
        assert type(n) is int, "n must be type 'int'"
        assert type(bon_batch_size) is int, "bon_batch_size must be type 'int'"
        assert n > 0, "n must be a positive integer"
        assert bon_batch_size > 0, "bon_batch_size must be a positive integer"

        sample_mat, reward_mat, curr_reward, best_sample = None, None, None, None
        while n > 0:
            batch_size = min(n, bon_batch_size)
            n -= batch_size
            for i in range(batch_size):
                # Sample from sampler
                sample = self.sample()
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
    
class CategoricalBONSampler(BONSampler):
    def __init__(self, distribution, logits=False):
        # Parameter validation
        assert type(distribution) is torch.Tensor, "distribution must be a torch tensor"

        # Construct distribution
        if logits:
            self.sampler = Categorical(logits=distribution)
        else:
            self.sampler = Categorical(probs=distribution)

    def sample(self):
        return self.sampler.sample()
