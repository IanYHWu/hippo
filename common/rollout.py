import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler
import numpy as np
from collections import deque


class Rollout:
    """Rollout storage for regular PPO trajectories"""

    def __init__(self, obs_shape, hidden_state_size, num_steps, num_envs, device):
        self.obs_shape = obs_shape
        self.hidden_state_size = hidden_state_size
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
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
        """Create genertors that sample from the rollout buffer"""
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
            print('env done')
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = self.done_batch.numpy()
        return rew_batch, done_batch


class DemoRollout:
    """Rollout for single demonstration trajectories"""

    def __init__(self, obs_shape, hidden_state_size, num_steps, device):
        self.num_steps = num_steps
        self.hidden_state_size = hidden_state_size
        self.obs_shape = obs_shape
        self.device = device
        self.step = 0
        self.reset()

    def reset(self):
        """Reset the storage. Uses lists for storage because trajectory lengths vary"""
        self.obs_store = torch.zeros(self.num_steps, *self.obs_shape)
        self.hidden_states_store = torch.zeros(self.num_steps, self.hidden_state_size)
        self.act_store = torch.zeros(self.num_steps)
        self.rew_store = torch.zeros(self.num_steps)
        self.done_store = torch.ones(self.num_steps)
        self.step = 0

    def store(self, obs, hidden_state, act, rew, done):
        """Insert trajectory data in the demo store"""
        self.obs_store[self.step] = torch.from_numpy(obs.copy())
        self.hidden_states_store[self.step] = torch.from_numpy(hidden_state.copy())
        self.act_store[self.step] = torch.from_numpy(act.copy())
        self.rew_store[self.step] = torch.from_numpy(rew.copy())
        self.done_store[self.step] = torch.from_numpy(done.copy())
        self.step += 1

    def get_demo_trajectory(self):
        return self.obs_store, self.hidden_states_store, self.act_store, self.rew_store, self.done_store


class DemoStorage:
    """Storage for demonstrations by seed. Used when we keep a limited number of demonstrations per seed"""

    def __init__(self, obs_shape, hidden_state_size, max_samples, num_steps, device):
        self.obs_shape = obs_shape
        self.hidden_state_size = hidden_state_size
        self.num_steps = num_steps
        self.max_samples = max_samples
        self.device = device
        self.guide = {}
        self.curr_ind = 0
        self.step = 0
        self.reset()

    def reset(self):
        self.obs_store = torch.zeros(self.max_samples, self.num_steps, *self.obs_shape)
        self.hidden_states_store = torch.zeros(self.max_samples, self.num_steps, self.hidden_state_size)
        self.act_store = torch.zeros(self.max_samples, self.num_steps)
        self.rew_store = torch.zeros(self.max_samples, self.num_steps)
        self.done_store = torch.zeros(self.max_samples, self.num_steps)
        self.curr_ind = 0
        self.step = 0

    def store(self, obs_trajectory, hidden_state_trajectory, act_trajectory, rew_trajectory, done_trajectory):
        """Insert trajectory data in the demo store"""
        index = self.curr_ind
        self.obs_store[index, :] = obs_trajectory
        self.hidden_states_store[index, :] = hidden_state_trajectory
        self.act_store[index, :] = act_trajectory
        self.rew_store[index, :] = rew_trajectory
        self.done_store[index, :] = done_trajectory
        self.curr_ind += 1
        if self.curr_ind >= self.max_samples:
            self.curr_ind = 0

    def update_guide(self, seed):
        print(self.guide)
        self.guide[seed] = self.curr_ind

    def check_guide(self, seed):
        if seed not in self.guide:
            return False
        else:
            return True

    def get_demo_trajectory(self, seed):
        index = self.guide[seed]
        obs_trajectory = self.obs_store[index, :]
        hidden_trajectory = self.hidden_states_store[index, :]
        act_trajectory = self.act_store[index, :]
        rew_trajectory = self.rew_store[index, :]
        done_trajectory = self.done_store[index, :]
        return obs_trajectory, hidden_trajectory, act_trajectory, rew_trajectory, done_trajectory


class DemoBuffer:
    """Demonstration replay buffer. Used for both demo_multi = True and demo_multi = False"""

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
        self.curr_ind += 1
        if self.curr_ind >= self.max_samples:
            self.curr_ind = 0
            self.buffer_full = True
        print(self.act_store)

    @staticmethod
    def generate_sample_mask(done_trajectory):
        done_index = torch.nonzero(done_trajectory)[0]
        sample_mask = 1 - done_trajectory
        sample_mask[done_index] = 1
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
        used for HIPPO"""
        actor_critic.eval()
        with torch.no_grad():
            num_samples = self.get_n_samples()
            for i in range(num_samples):
                dist_batch, val_batch, _ = actor_critic(self.obs_store[i].squeeze(1).float().to(self.device),
                                                        self.hidden_states_store[i].float().to(self.device),
                                                        self.mask_store[i].float().to(self.device))
                log_prob_act_batch = dist_batch.log_prob(self.act_store[i].to(self.device))
                self.value_store[i, :] = val_batch * self.sample_mask_store[i, :]
                self.log_prob_act_store[i, :] = log_prob_act_batch

    def compute_estimates(self, actor_critic, gamma=0.99, lmbda=0.95, normalise_adv=False):
        """Compute the advantages of the transitions - used for HIPPO"""
        self.compute_pi_v(actor_critic)
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

    def demo_generator(self, batch_size, mini_batch_size, recurrent=False):
        """Create generator to sample transitions from the replay buffer - used for both HIPPO and IL"""
        if not recurrent:
            weights = self.sample_mask_store.squeeze(-1).reshape(-1)  # ignore all padding transitions when sampling
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

    """
    def update_priorities(self, updates):
        updates = self._list_to_tensor(updates).reshape(-1)
        indices = [i for sublist in self.last_updated_indices for i in sublist]
        rows, cols = self.priorities.shape[0], self.priorities.shape[1]
        self.priorities = self.priorities.reshape(-1)
        self.priorities[indices] = updates ** self.alpha
        self.priorities += self.eps  # ensure that all samples have a non-zero chance of being sampled
        self.priorities *= self.mask_store.reshape(-1)
        self._normalise_priorities()
        self.priorities = self.priorities.reshape(rows, cols).unsqueeze(-1)

    def compute_priorities_hippo(self):
        if self.sampling_strategy == 'prioritised':
            self.priorities = torch.abs(self.returns_store.squeeze(-1) - self.value_store)
        elif self.sampling_strategy == 'prioritised_clamp':
            self.priorities = torch.clamp(self.returns_store - self.value_store, min=0)
        else:
            raise NotImplementedError
        self.priorities = self.priorities ** self.alpha
        self.priorities += self.eps
        self.priorities *= self.mask_store.squeeze(-1)
        self._normalise_priorities()

    def get_per_weights(self):
        n = self.get_n_valid_transitions()
        per_weights = ((1 / n) * (1 / self.priorities)) ** self.beta
        max_weight = torch.max(per_weights)
        per_weights = per_weights / max_weight

        return per_weights

    def _normalise_priorities(self):
        summed = torch.sum(self.priorities)
        self.priorities = self.priorities / summed
        self.max_priority = torch.max(self.priorities)
    """