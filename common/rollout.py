"""
Rollout module handles storing trajectories for both environment steps and demonstration steps. Also handles computing
the required estimates
"""

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler
import numpy as np
from collections import deque


class Rollout:
    """Rollout storage for regular PPO trajectories

    Attributes:
        obs_shape: observation shape
        hidden_state_size: hidden state size
        num_steps: number of steps per rollout
        num_envs: number of envs per rollout
        device: cpu/gpu
    """

    def __init__(self, obs_shape, hidden_state_size, num_steps, num_envs, device):
        self.obs_shape = obs_shape
        self.hidden_state_size = hidden_state_size
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()
        self.seed_log = set()

    def reset(self):
        """Reset the rollout"""
        self.obs_batch = torch.zeros(self.num_steps + 1, self.num_envs, *self.obs_shape)
        self.hidden_states_batch = torch.zeros(self.num_steps + 1, self.num_envs, self.hidden_state_size)
        self.act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.value_batch = torch.zeros(self.num_steps + 1, self.num_envs)
        self.return_batch = torch.zeros(self.num_steps, self.num_envs)
        self.adv_batch = torch.zeros(self.num_steps, self.num_envs)
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, hidden_state, act, rew, done, info, log_prob_act, value):
        """Store results in the rollout buffer incrementally"""
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.hidden_states_batch[self.step] = torch.from_numpy(hidden_state.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.log_prob_act_batch[self.step] = torch.from_numpy(log_prob_act.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)
        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_hidden_state, last_value):
        """Store data for the t+1th step. Used for computing advantages and returns"""
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.hidden_states_batch[-1] = torch.from_numpy(last_hidden_state.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        """Compute returns and advantages"""
        rew_batch = self.rew_batch
        if use_gae:
            # generalised advantage estimation
            A = 0
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]
                value = self.value_batch[i]
                next_value = self.value_batch[i + 1]

                delta = (rew + gamma * next_value * (1 - done)) - value
                self.adv_batch[i] = A = gamma * lmbda * A * (1 - done) + delta
                self.return_batch[i] = self.adv_batch[i] + self.value_batch[i]
        else:
            G = self.value_batch[-1]  # t+1th value
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]

                G = rew + gamma * G * (1 - done)
                self.return_batch[i] = G

        if normalize_adv:
            # normalise the advantages (as used in Kostrikov's implementation)
            self.adv_batch = (self.adv_batch - torch.mean(self.adv_batch)) / (torch.std(self.adv_batch) + 1e-8)

    def fetch_train_generator(self, mini_batch_size=None, recurrent=False):
        """Create generators that sample from the rollout buffer"""
        batch_size = self.num_steps * self.num_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size
        # If the agent's policy is not recurrent, data can be sampled without considering the time-horizon
        if not recurrent:
            sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                                   mini_batch_size,
                                   drop_last=True)
            for indices in sampler:
                obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
                hidden_state_batch = torch.FloatTensor(self.hidden_states_batch[:-1]).reshape(-1,
                                                                                              self.hidden_state_size).to(
                    self.device)
                act_batch = torch.FloatTensor(self.act_batch).reshape(-1)[indices].to(self.device)
                done_batch = torch.FloatTensor(self.done_batch).reshape(-1)[indices].to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1)[indices].to(self.device)
                value_batch = torch.FloatTensor(self.value_batch[:-1]).reshape(-1)[indices].to(self.device)
                return_batch = torch.FloatTensor(self.return_batch).reshape(-1)[indices].to(self.device)
                adv_batch = torch.FloatTensor(self.adv_batch).reshape(-1)[indices].to(self.device)
                yield obs_batch, hidden_state_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch
        # If agent's policy is recurrent, sample data along the time-horizon (not used in this project so far)
        else:
            num_mini_batch_per_epoch = batch_size // mini_batch_size
            num_envs_per_batch = self.num_envs // num_mini_batch_per_epoch
            perm = torch.randperm(self.num_envs)
            for start_ind in range(0, self.num_envs, num_envs_per_batch):
                idxes = perm[start_ind:start_ind + num_envs_per_batch]
                obs_batch = torch.FloatTensor(self.obs_batch[:-1, idxes]).reshape(-1, *self.obs_shape).to(self.device)
                # [0:1] instead of [0] to keep two-dimensional array
                hidden_state_batch = torch.FloatTensor(self.hidden_states_batch[0:1, idxes]).reshape(
                    -1, self.hidden_state_size).to(self.device)
                act_batch = torch.FloatTensor(self.act_batch[:, idxes]).reshape(-1).to(self.device)
                done_batch = torch.FloatTensor(self.done_batch[:, idxes]).reshape(-1).to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch[:, idxes]).reshape(-1).to(self.device)
                value_batch = torch.FloatTensor(self.value_batch[:-1, idxes]).reshape(-1).to(self.device)
                return_batch = torch.FloatTensor(self.return_batch[:, idxes]).reshape(-1).to(self.device)
                adv_batch = torch.FloatTensor(self.adv_batch[:, idxes]).reshape(-1).to(self.device)
                yield obs_batch, hidden_state_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch

    def fetch_log_data(self):
        """Extract rewards from info - to be sent to the logger"""
        if 'env_reward' in self.info_batch[0][0]:
            rew_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = self.rew_batch.numpy()
        if 'env_done' in self.info_batch[0][0]:
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = self.done_batch.numpy()

        for step in range(self.num_steps):
            infos = self.info_batch[step]
            for info in infos:
                self.seed_log.add(info["level_seed"])

        return rew_batch, done_batch


class DemoRollout:
    """Rollout for single demonstration trajectories. Used to collect transitions

    Attributes:
        obs_shape: observation shape
        hidden_state_size: hidden state size
        num_steps: maximum length of demo trajectories permitted
        step: tracks steps of the trajectory
        device: gpu/cpu
    """

    def __init__(self, obs_shape, hidden_state_size, num_steps, device):
        self.num_steps = num_steps
        self.hidden_state_size = hidden_state_size
        self.obs_shape = obs_shape
        self.device = device
        self.step = 0
        self.reset()

    def reset(self):
        """Reset the storage"""
        self.obs_store = torch.zeros(self.num_steps, *self.obs_shape)
        self.hidden_states_store = torch.zeros(self.num_steps, self.hidden_state_size)
        self.act_store = torch.zeros(self.num_steps)
        self.rew_store = torch.zeros(self.num_steps)
        self.done_store = torch.ones(self.num_steps)
        self.info_store = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, hidden_state, act, rew, done, info):
        """Insert transition data into the rollout """
        self.obs_store[self.step] = torch.from_numpy(obs.copy())
        self.hidden_states_store[self.step] = torch.from_numpy(hidden_state.copy())
        self.act_store[self.step] = torch.from_numpy(act.copy())
        self.rew_store[self.step] = torch.from_numpy(rew.copy())
        self.done_store[self.step] = torch.from_numpy(done.copy())
        self.info_store.append(info)
        self.step += 1

    def get_demo_trajectory(self):
        """Return the trajectories collected"""
        env_reward = self.get_env_rewards()
        return self.obs_store, self.hidden_states_store, self.act_store, self.rew_store, self.done_store, env_reward

    def get_env_rewards(self):
        sum_rewards = 0
        for step in range(len(self.info_store)):
            info = self.info_store[step]
            sum_rewards += info[0]['env_reward']

        return sum_rewards


class DemoStorage:
    """Storage for demonstrations by seed. Used when we keep a limited number of demonstrations per seed

    Attributes:
        obs_shape: observation shape
        hidden_state_size: hidden state size
        max_samples: maximum capacity of the demo storage
        num_steps: maximum length of demo trajectories permitted
        device: gpu/cpu
        seed_to_ind: hashtable with keys being seeds and values being the indices of the corresponding demo trajectories
        curr_ind: index of the row to insert a trajectory
        step: tracks steps of the trajectory
    """

    def __init__(self, obs_shape, hidden_state_size, max_samples, num_steps, device):
        self.obs_shape = obs_shape
        self.hidden_state_size = hidden_state_size
        self.num_steps = num_steps
        self.max_samples = max_samples
        self.device = device
        self.seed_to_ind = {}  # maps a seed to the indices containing corresponding demos (one-to-many)
        self.ind_to_seed = {}  # maps storage index to seed (many-to-one)
        self.curr_ind = 0
        self.step = 0
        self.storage_full = False
        self.reset()

    def reset(self):
        """Reset the demo storage"""
        self.obs_store = torch.zeros(self.max_samples, self.num_steps, *self.obs_shape)
        self.hidden_states_store = torch.zeros(self.max_samples, self.num_steps, self.hidden_state_size)
        self.act_store = torch.zeros(self.max_samples, self.num_steps)
        self.rew_store = torch.zeros(self.max_samples, self.num_steps)
        self.done_store = torch.zeros(self.max_samples, self.num_steps)
        self.env_rewards = torch.zeros(self.max_samples)
        self.curr_ind = 0
        self.step = 0
        self.storage_full = False
        self.seed_to_ind = {}  # guide - maps seed to indices
        self.ind_to_seed = {}  # guide - maps index to seed

    def store(self, obs_trajectory, hidden_state_trajectory, act_trajectory, rew_trajectory, done_trajectory,
              env_reward, store_index=None):
        """Insert trajectory data in the demo store. Inserts entire trajectories"""
        if store_index is not None:
            index = store_index
        else:
            index = self.curr_ind
        self.obs_store[index, :] = obs_trajectory
        self.hidden_states_store[index, :] = hidden_state_trajectory
        self.act_store[index, :] = act_trajectory
        self.rew_store[index, :] = rew_trajectory
        self.done_store[index, :] = done_trajectory
        self.env_rewards[index] = env_reward
        if store_index is None:
            self.curr_ind += 1
        # once we reach full capacity, overwrite the oldest sample
        if self.curr_ind >= self.max_samples:
            self.storage_full = True
            self.curr_ind = 0

    def update_guides(self, seed, store_index=None):
        """Document seed/index pair in the guides"""
        if store_index is not None:
            # if store_index is used, store in a particular index
            index = store_index
        else:
            # else store in the next empty slot
            index = self.curr_ind
        if index in self.ind_to_seed:
            # for seed-to-ind we need to disconnect index from old seed
            old_seed = self.ind_to_seed[index]
            self.seed_to_ind[old_seed].remove(index)
        self.ind_to_seed[index] = seed
        if seed in self.seed_to_ind:
            self.seed_to_ind[seed].append(index)
        else:
            self.seed_to_ind[seed] = [index]

    def check_guide(self, seed):
        """Check if a certain seed is present in the seed_to_ind"""
        if seed not in self.seed_to_ind:
            return False
        else:
            return True

    def get_n_samples(self):
        """Get the number of samples in the demo_storage"""
        if self.storage_full:
            return self.max_samples
        else:
            return self.curr_ind

    def get_n_valid_transitions(self):
        """Count the number of non-padding transitions stored in the buffer"""
        mask = 1 - self.done_store
        valid_samples = torch.count_nonzero(mask)
        return valid_samples

    def get_demo_trajectory(self, store_index=None):
        """Extract demo trajectories from the storage by seed or by index"""
        index = store_index
        obs_trajectory = self.obs_store[index, :]
        hidden_trajectory = self.hidden_states_store[index, :]
        act_trajectory = self.act_store[index, :]
        rew_trajectory = self.rew_store[index, :]
        done_trajectory = self.done_store[index, :]

        return obs_trajectory, hidden_trajectory, act_trajectory, rew_trajectory, done_trajectory


class DemoBuffer:
    """Demonstration replay buffer. Transitions are directly sampled from here

    Attributes:
        obs_shape: observation shape
        hidden_state_size: hidden state size
        max_samples: maximum capacity of the demo storage
        max_steps: maximum length of demo trajectories permitted
        device: cpu/gpu
        curr_ind: index of the row to insert a trajectory
        buffer_full: flag to indicate if the buffer is full
    """

    def __init__(self, obs_shape, hidden_state_size, max_samples, max_steps, device):
        self.obs_shape = obs_shape
        self.hidden_state_size = hidden_state_size
        self.max_steps = max_steps
        self.max_samples = max_samples
        self.device = device
        self.curr_ind = 0
        self.buffer_full = False
        self.reset()

    def reset(self):
        """Reset the demo replay buffer """
        self.obs_store = torch.zeros(self.max_samples, self.max_steps, *self.obs_shape)
        self.hidden_states_store = torch.zeros(self.max_samples, self.max_steps, self.hidden_state_size)
        self.act_store = torch.zeros(self.max_samples, self.max_steps)
        self.rew_store = torch.zeros(self.max_samples, self.max_steps)
        self.done_store = torch.zeros(self.max_samples, self.max_steps)
        self.mask_store = torch.zeros(self.max_samples, self.max_steps)
        self.sample_mask_store = torch.zeros(self.max_samples, self.max_steps)
        self.value_store = torch.zeros(self.max_samples, self.max_steps)
        self.adv_store = torch.zeros(self.max_samples, self.max_steps)
        self.returns_store = torch.zeros(self.max_samples, self.max_steps)
        self.log_prob_act_store = torch.zeros(self.max_samples, self.max_steps)
        self.curr_ind = 0
        self.buffer_full = False

    def store(self, obs_trajectory, hidden_state_trajectory, act_trajectory, rew_trajectory, done_trajectory):
        """Insert demonstration trajectories"""
        index = self.curr_ind
        self.obs_store[index, :] = obs_trajectory
        self.hidden_states_store[index, :] = hidden_state_trajectory
        self.act_store[index, :] = act_trajectory
        self.rew_store[index, :] = rew_trajectory
        self.done_store[index, :] = done_trajectory
        self.mask_store[index, :] = 1 - done_trajectory
        self.sample_mask_store[index, :] = self.generate_sample_mask(done_trajectory)
        # sample_mask indicates the padding positions
        self.curr_ind += 1
        if self.curr_ind >= self.max_samples:
            # if the buffer is full, overwrite the oldest sample
            self.curr_ind = 0
            self.buffer_full = True

    @staticmethod
    def generate_sample_mask(done_trajectory):
        """Create a sample mask, which masks all padding data"""
        done_index = torch.nonzero(done_trajectory)[0]
        sample_mask = 1 - done_trajectory
        sample_mask[done_index] = 1  # differs from the regular mask in that the terminal transition is not masked
        return sample_mask

    def get_n_valid_transitions(self):
        """Count the number of non-padding transitions stored in the buffer"""
        valid_samples = torch.count_nonzero(self.mask_store)
        return valid_samples

    def get_n_samples(self):
        """Count the number of trajectories in the buffer"""
        if self.buffer_full:
            return self.max_samples
        else:
            return self.curr_ind

    def compute_pi_v(self, actor_critic):
        """Compute and store action logits and values for all transitions in the buffer using an actor critic -
        used for HIPPO

            actor_critic: current target policy
        """
        actor_critic.eval()
        with torch.no_grad():
            num_samples = self.get_n_samples()
            for i in range(num_samples):
                # iterate through all samples in the buffer and compute action logits and values
                dist_batch, val_batch, _ = actor_critic(self.obs_store[i].float().to(self.device),
                                                        self.hidden_states_store[i].float().to(self.device),
                                                        self.mask_store[i].float().to(self.device))
                log_prob_act_batch = dist_batch.log_prob(self.act_store[i].to(self.device))
                self.value_store[i, :] = val_batch.cpu() * self.sample_mask_store[i, :]  # mask the padding values
                self.log_prob_act_store[i, :] = log_prob_act_batch.cpu()

    def compute_estimates(self, actor_critic, gamma=0.99, lmbda=0.95, normalise_adv=False):
        """Compute the advantages of the transitions - used for HIPPO"""
        self.compute_pi_v(actor_critic)  # compute action logits and values
        # generalised advantage estimation
        A = 0
        for i in reversed(range(self.max_steps)):
            rew = self.rew_store[:, i]
            done = self.done_store[:, i]
            value = self.value_store[:, i]
            if i < self.max_steps - 1:
                next_value = self.value_store[:, i + 1]
            else:
                next_value = 0
            delta = (rew + gamma * next_value * (1 - done)) - value
            self.adv_store[:, i] = A = gamma * lmbda * A * (1 - done) + delta
            self.returns_store[:, i] = self.adv_store[:, i] + self.value_store[:, i]

        if normalise_adv:
            # when we normalise the advantages, we need to account for the fact that some transitions are paddings.
            adv_mean = torch.sum(self.adv_store) / (self.get_n_valid_transitions() + 1e-8)
            mean_sq = torch.sum(self.adv_store ** 2) / (self.get_n_valid_transitions() + 1e-8)
            sq_mean = adv_mean ** 2
            adv_std = torch.sqrt(mean_sq - sq_mean + 1e-8)  # variance is mean of the square - square of the mean
            self.adv_store = (self.adv_store - adv_mean) / adv_std

    def compute_value_losses(self):
        """Compute the average L1-value loss for all samples in the buffer"""
        n = self.get_n_samples()
        value_losses = torch.zeros(n)
        for i in range(n):
            adv_sample = torch.abs(self.adv_store[i])
            mask_sample = self.sample_mask_store[i]
            sum_adv = torch.sum(adv_sample * mask_sample)
            sample_len = torch.nonzero(1 - mask_sample)[0].item()
            value_losses[i] = sum_adv / sample_len

        return value_losses

    def demo_generator(self, batch_size, mini_batch_size, recurrent=False, sampling_method='uniform', nu=1):
        """Create generator to sample transitions from the replay buffer"""
        if not recurrent:
            if sampling_method == 'uniform':
                weights = self.sample_mask_store.squeeze(-1).reshape(-1)  # ignore all padding transitions when sampling
            elif sampling_method == 'prioritised':
                abs_advs = torch.abs(self.adv_store.reshape(-1)) * self.sample_mask_store.squeeze(-1).reshape(-1)
                weights = abs_advs ** nu
                weights /= torch.sum(weights)
            else:
                raise NotImplementedError
            sampler = BatchSampler(WeightedRandomSampler(weights, int(batch_size), replacement=False),
                                   mini_batch_size, drop_last=True)
            for indices in sampler:
                obs_batch = torch.FloatTensor(self.obs_store.float()).reshape(-1, *self.obs_shape)[indices].to(
                    self.device)
                hidden_state_batch = torch.FloatTensor(self.hidden_states_store.float()).reshape(
                    -1, self.hidden_state_size).to(self.device)
                act_batch = torch.FloatTensor(self.act_store.float()).reshape(-1)[indices].to(self.device)
                mask_batch = torch.FloatTensor(self.mask_store.float()).reshape(-1)[indices].to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_store.float()).reshape(-1)[indices].to(
                    self.device)
                val_batch = torch.FloatTensor(self.value_store.float()).reshape(-1)[indices].to(self.device)
                returns_batch = torch.FloatTensor(self.returns_store.float()).reshape(-1)[indices].to(self.device)
                adv_batch = torch.FloatTensor(self.adv_store.float()).reshape(-1)[indices].to(self.device)

                yield obs_batch, hidden_state_batch, act_batch, returns_batch, \
                      mask_batch, log_prob_act_batch, val_batch, adv_batch
        else:
            raise NotImplementedError
