import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler
import numpy as np
from collections import deque


class Storage:
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


class DemoStorage:
    """Storage for single demonstration trajectories (demo_multi = False)"""

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        """Reset the storage. Uses lists for storage because trajectory lengths vary"""
        self.obs_store = []
        self.hidden_states_store = []
        self.act_store = []
        self.rew_store = []
        self.returns_store = None
        self.trajectory_length = 0

    def store(self, obs, hidden_state, act, rew):
        """Insert trajectory data in the demo store"""
        self.obs_store.append(torch.from_numpy(obs.copy()))
        self.hidden_states_store.append(torch.from_numpy(hidden_state.copy()))
        self.act_store.append(torch.from_numpy(act.copy()))
        self.rew_store.append(torch.from_numpy(rew.copy()))

        self.trajectory_length += 1  # track the current trajectory length

    @staticmethod
    def _list_to_tensor(list_of_tensors):
        """Convert list of tensors to tensor"""
        big_tensor = torch.stack(list_of_tensors)
        return big_tensor

    def _stores_to_tensors(self):
        """Convert data to tensor form"""
        self.obs_store = self._list_to_tensor(self.obs_store)
        self.hidden_states_store = self._list_to_tensor(self.hidden_states_store)
        self.act_store = self._list_to_tensor(self.act_store)
        self.rew_store = self._list_to_tensor(self.rew_store)

    def compute_returns(self, gamma=0.99):
        """Compute the returns"""
        self._stores_to_tensors()
        self.returns_store = torch.zeros(self.trajectory_length)
        G = 0
        for i in reversed(range(self.trajectory_length)):
            rew = self.rew_store[i]
            G = rew + gamma * G
            self.returns_store[i] = G

    def get_sum_rewards(self):
        return torch.sum(self.rew_store)


class MultiDemoStorage:
    """Storage for multiple demonstration trajectories - used when multiple demonstrations are generated by multiple
       oracle rollouts (demo_multi = True)
    """

    def __init__(self, obs_shape, hidden_state_size, num_steps, num_envs, device):
        self.obs_shape = obs_shape
        self.hidden_state_size = hidden_state_size
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs_batch = torch.zeros(self.num_envs, self.num_steps, *self.obs_shape)
        self.hidden_states_batch = torch.zeros(self.num_envs, self.num_steps, self.hidden_state_size)
        self.act_batch = torch.zeros(self.num_envs, self.num_steps)
        self.rew_batch = torch.zeros(self.num_envs, self.num_steps)
        self.done_batch = torch.zeros(self.num_envs, self.num_steps)
        self.step = 0

    def store(self, obs, hidden_state, act, rew, done):
        """Insert demonstration trajectories"""
        self.obs_batch[:, self.step] = torch.from_numpy(obs.copy())
        self.hidden_states_batch[:, self.step] = torch.from_numpy(hidden_state.copy())
        self.act_batch[:, self.step] = torch.from_numpy(act.copy())
        self.rew_batch[:, self.step] = torch.from_numpy(rew.copy())
        self.done_batch[:, self.step] = torch.from_numpy(done.copy())
        self.step = (self.step + 1) % self.num_steps

    @staticmethod
    def _remove_cliffhangers_helper(input_tensor, mask, non_zero_rows):
        """Helper function to remove cliffhanger trajectories. Removes non-terminating episodes,
        and masks any unwanted transitions"""
        non_zero_tensor = input_tensor[non_zero_rows]
        shapes = len(non_zero_tensor.shape[2:])
        for i in range(0, shapes):
            mask = mask.unsqueeze(-1)

        return non_zero_tensor * mask

    def _remove_cliffhangers(self):
        """Remove cliffhanger trajectories"""
        mask, non_zero_rows = self._generate_masks()
        self.obs_batch = self._remove_cliffhangers_helper(self.obs_batch, mask, non_zero_rows)
        self.hidden_states_batch = self._remove_cliffhangers_helper(self.hidden_states_batch, mask, non_zero_rows)
        self.act_batch = self._remove_cliffhangers_helper(self.act_batch, mask, non_zero_rows)
        self.rew_batch = self._remove_cliffhangers_helper(self.rew_batch, mask, non_zero_rows)

    def _generate_masks(self):
        """Masks unwanted transitions (i.e. transitions from additional episodes and non-terminating transitions).
           Generates the correct masks, and the indices of the samples to keep"""

        # EXAMPLE: done_batch = [[0, 0, 0, 1, 0],
        #                        [0, 0, 0, 0, 0],  # remove non-terminating transitions
        #                        [0, 0, 1, 0, 1]]  # only keep the first episode
        # ----->   mask =       [[1, 1, 1, 1, 0],
        #                        [1, 1, 1, 0, 0]]
        # ----->   non_zero_rows = [0, 2]

        non_zero_rows = torch.abs(self.done_batch).sum(dim=1) > 0  # find trajectories that terminate in time
        dones = self.done_batch[non_zero_rows]  # get the done data for only those trajectories
        dones = dones.cpu()
        done_arr = dones.numpy()
        one_dones = np.argwhere(done_arr == 1)  # find the ends of the first episodes for each terminating env

        # the rest of this code is used to generate the masking array
        batch_size, tensor_len = done_arr.shape
        t_len_arr = np.vstack((np.arange(batch_size), np.ones(batch_size) * tensor_len)).T
        comb_arr = np.concatenate((one_dones, t_len_arr))
        _, i = np.unique(comb_arr[:, 0], return_index=True)
        end_indices = comb_arr[i][:, 1] + 1

        mask = torch.zeros(done_arr.shape[0], done_arr.shape[1] + 1, dtype=int)
        mask[(torch.arange(done_arr.shape[0]), end_indices)] = 1
        mask = 1 - mask.cumsum(dim=1)[:, :-1]
        self.mask_batch = mask

        return mask, non_zero_rows

    def compute_returns(self, gamma=0.99):
        """Remove cliffhangers and then compute the returns"""
        self._remove_cliffhangers()
        G = 0
        self.return_batch = torch.zeros(len(self.rew_batch), self.num_steps)
        for i in reversed(range(self.num_steps)):
            rew = self.rew_batch[:, i]
            G = rew + gamma * G
            self.return_batch[:, i] = G


class DemoReplayBuffer:
    """Demonstration replay buffer. Used for both demo_multi = True and demo_multi = False"""

    def __init__(self, obs_size, hidden_state_size, device, max_samples, sampling_strategy='uniform', alpha=1.0,
                 beta=0.6, eps=1e-5, mode='hippo'):
        self.obs_size = obs_size
        self.hidden_state_size = hidden_state_size
        self.max_len = 0
        self.device = device
        self.max_samples = max_samples
        self.sampling_strategy = sampling_strategy
        self.mode = mode
        if sampling_strategy == 'prioritised' or sampling_strategy == 'prioritised_clamp':
            self.prioritised = True
        else:
            self.prioritised = False

        if self.prioritised:
            self.max_priority = 1
            self.alpha = alpha
            self.beta = beta
            self.eps = eps
            self.last_updated_indices = []

    def store(self, demo_store):
        """Extract data from demo_store and store in the replay buffer"""

        # EXAMPLE (for demo_multi = False):
        # Trajectory: [3, 2, 4]
        # Buffer: [[4, 5, 3, 7, 0], ----> [[3, 2, 4, 0, 0],
        #          [3, 2, 7, 8, 6]]        [4, 5, 3, 7, 0],
        #                                  [3, 2, 7, 8, 6]]
        #
        # Trajectory: [5, 2, 4, 7, 8]
        # Buffer: [[3, 2, 4]] ----> [[5, 2, 4, 7, 8],
        #                            [3, 2, 4, 0, 0]]

        if isinstance(demo_store, DemoStorage):
            # for demo_multi = False i.e. we sample single demonstrations from the oracle each time
            # Extract the single trajectory
            obs = demo_store.obs_store
            hidden_states = demo_store.hidden_states_store
            actions = demo_store.act_store
            rewards = demo_store.rew_store
            trajectory_len = demo_store.trajectory_length
            returns = demo_store.returns_store.reshape(trajectory_len, 1)
            demo_store.reset()  # reset the demo_store after we extract the data from it

            mask = torch.ones(trajectory_len).reshape(trajectory_len, 1)  # generate mask of ones, of trajectory length

            if self.prioritised and self.mode == 'il':
                priorities = torch.ones(trajectory_len).reshape(trajectory_len, 1) * self.max_priority
            else:
                priorities = None

            if self.max_len == 0:
                # if this is the first trajectory to be stored, just store it
                self.obs_store = obs.unsqueeze(0)
                self.hidden_states_store = hidden_states.unsqueeze(0)
                self.act_store = actions.unsqueeze(0)
                self.rew_store = rewards.unsqueeze(0)
                self.returns_store = returns.unsqueeze(0)
                self.mask_store = mask.unsqueeze(0)
                self.max_len = trajectory_len

                if self.prioritised and self.mode == 'il':
                    self.priorities = priorities.unsqueeze(0)

            else:
                # if not, we need to pad the trajectory or the buffer, because trajectories can be of different lengths
                if trajectory_len < self.max_len:
                    # if the trajectory is smaller than the max length trajectory seen so far, pad the trajectory
                    obs = self._pad_tensor(obs, self.max_len, pad_trajectory=True)
                    hidden_states = self._pad_tensor(hidden_states, self.max_len, pad_trajectory=True)
                    actions = self._pad_tensor(actions, self.max_len, pad_trajectory=True)
                    returns = self._pad_tensor(returns, self.max_len, pad_trajectory=True)
                    rewards = self._pad_tensor(rewards, self.max_len, pad_trajectory=True)
                    # pad the trajectory mask with zeros
                    mask = self._pad_tensor(mask, self.max_len, pad_trajectory=True)

                    if self.prioritised and self.mode == 'il':
                        priorities = self._pad_tensor(priorities, self.max_len, pad_trajectory=True)

                elif trajectory_len > self.max_len:
                    # if the trajectory is longer than the max length trajectory so far, pad the buffer
                    self.obs_store = self._pad_tensor(self.obs_store, trajectory_len, pad_trajectory=False)
                    self.hidden_states_store = self._pad_tensor(self.hidden_states_store, trajectory_len,
                                                                pad_trajectory=False)
                    self.act_store = self._pad_tensor(self.act_store, trajectory_len, pad_trajectory=False)
                    self.returns_store = self._pad_tensor(self.returns_store, trajectory_len, pad_trajectory=False)
                    self.rew_store = self._pad_tensor(self.rew_store, trajectory_len, pad_trajectory=False)
                    # pad the buffer mask with zeros
                    self.mask_store = self._pad_tensor(self.mask_store, trajectory_len, pad_trajectory=False)
                    self.max_len = trajectory_len

                    if self.prioritised and self.mode == 'il':
                        self.priorities = self._pad_tensor(self.priorities, trajectory_len, pad_trajectory=False)

                # add the new trajectory to the replay buffer
                self.obs_store = self._add_to_buffer(obs, self.obs_store)
                self.hidden_states_store = self._add_to_buffer(hidden_states, self.hidden_states_store)
                self.act_store = self._add_to_buffer(actions, self.act_store)
                self.returns_store = self._add_to_buffer(returns, self.returns_store)
                self.rew_store = self._add_to_buffer(rewards, self.rew_store)
                self.mask_store = self._add_to_buffer(mask, self.mask_store)

                if self.prioritised and self.mode == 'il':
                    self.priorities = self._add_to_buffer(priorities, self.priorities)

            # if the capacity of the buffer is full, remove the oldest trajectory
            n_samples = self.get_buffer_n_samples()
            if n_samples > self.max_samples:
                self.obs_store = self.obs_store[:-1, :]
                self.hidden_states_store = self.hidden_states_store[:-1, :]
                self.act_store = self.act_store[:-1, :]
                self.rew_store = self.rew_store[:-1, :]
                self.returns_store = self.returns_store[:-1, :]
                self.mask_store = self.mask_store[:-1, :]

                if self.prioritised and self.mode == 'il':
                    self.priorities = self.priorities[:-1, :]

        else:
            assert self.mode == 'hippo'  # multi-trajectory mode only available for hippo
            # if demo_multi = True, we just directly store the data
            self.obs_store = demo_store.obs_batch
            self.hidden_states_store = demo_store.hidden_states_batch
            self.act_store = demo_store.act_batch
            self.rew_store = demo_store.rew_batch
            self.returns_store = demo_store.return_batch
            self.mask_store = demo_store.mask_batch
            self.max_len = self.mask_store.shape[1]

            demo_store.reset()  # reset the demo_store after we extract data from it

    @staticmethod
    def _pad_tensor(input_tensor, new_len, pad_trajectory=False):
        """Pad a tensor with zeros or zero-tensors"""
        height = input_tensor.shape[0]
        width = input_tensor.shape[1]

        if pad_trajectory:
            data_size = input_tensor[0].shape
            zero = torch.zeros(data_size).unsqueeze(0)
            t = tuple([1 for i in range(0, len(data_size))])
            zeros = zero.repeat(new_len - height, *t)
            output_tensor = torch.cat((input_tensor, zeros), dim=0)

        else:
            data_size = input_tensor[0][0].shape
            zero = torch.zeros(data_size).unsqueeze(0).unsqueeze(0)
            t = tuple([1 for i in range(0, len(data_size))])
            zeros = zero.repeat(height, new_len - width, *t)
            output_tensor = torch.cat((input_tensor, zeros), dim=1)

        return output_tensor

    @staticmethod
    def _add_to_buffer(trajectory, buffer):
        """Add a trajectory to the buffer"""
        new_buffer = torch.cat((trajectory.unsqueeze(0), buffer), dim=0)
        return new_buffer

    @staticmethod
    def _list_to_tensor(list_of_tensors):
        """Convert a list of tensors into a tensor"""
        big_tensor = torch.stack(list_of_tensors)
        return big_tensor

    def get_n_valid_transitions(self):
        """Count the number of non-padding transitions stored in the buffer"""
        valid_samples = torch.count_nonzero(self.mask_store)
        return valid_samples

    def get_buffer_n_samples(self):
        """Count the number of trajectories in the buffer"""
        return len(self.act_store)

    def compute_pi_v(self, actor_critic):
        """Compute and store action logits and values for all transitions in the buffer using an actor critic -
        used for HIPPO"""
        log_prob_act_list = []
        val_list = []
        actor_critic.eval()
        with torch.no_grad():
            for i in range(0, len(self.obs_store)):
                dist_batch, val_batch, _ = actor_critic(self.obs_store[i].squeeze(1).float().to(self.device),
                                                        self.hidden_states_store[i].float().to(self.device),
                                                        self.mask_store[i].float().to(self.device))
                log_prob_act_batch = dist_batch.log_prob(self.act_store[i].to(self.device))
                val_list.append(val_batch)
                log_prob_act_list.append(log_prob_act_batch)
            self.value_store = self._list_to_tensor(val_list).cpu()
            self.value_store *= self.mask_store.squeeze(-1)  # set padding values to zero using the mask
            self.log_prob_act_store = self._list_to_tensor(log_prob_act_list).cpu()

    def compute_hippo_advantages(self, actor_critic, gamma=0.99, lmbda=0.95, normalise_adv=True):
        """Compute the advantages of the transitions - used for HIPPO"""
        self.compute_pi_v(actor_critic)
        adv_list = []
        # can easily vectorise this - will do at some point
        # padding
        for sample in range(0, len(self.act_store)):
            adv_store = torch.zeros(self.max_len)
            A = 0
            for i in reversed(range(self.max_len)):
                rew = self.rew_store[sample][i].cpu()
                mask = self.mask_store[sample][i].cpu()
                value = self.value_store[sample][i].cpu()
                if i == self.max_len - 1:
                    next_value = 0
                else:
                    next_value = self.value_store[sample][i + 1].cpu()
                delta = (rew + gamma * next_value * mask) - value
                adv_store[i] = A = gamma * lmbda * A * mask + delta
            adv_list.append(adv_store)

        self.adv_store = self._list_to_tensor(adv_list)

        if normalise_adv:
            # when we normalise the advantages, we need to account for the fact that some transitions are paddings.
            adv_mean = torch.sum(self.adv_store) / (self.get_n_valid_transitions() + 1e-8)
            mean_sq = torch.sum(self.adv_store ** 2) / (self.get_n_valid_transitions() + 1e-8)
            sq_mean = adv_mean ** 2
            adv_std = torch.sqrt(mean_sq - sq_mean + 1e-8)  # variance is mean of the square - square of the mean
            self.adv_store = (self.adv_store - adv_mean) / adv_std

    def demo_generator(self, batch_size, mini_batch_size, recurrent=False, mode='hippo'):
        """Create generator to sample transitions from the replay buffer - used for both HIPPO and IL"""
        if not recurrent:
            if mode == 'hippo' and self.prioritised:
                self.compute_priorities_hippo()
            if self.sampling_strategy == 'uniform':
                weights = self.mask_store.squeeze(-1).reshape(-1)  # ignore all padding transitions when sampling
                sampler = BatchSampler(WeightedRandomSampler(weights, int(batch_size), replacement=False),
                                       mini_batch_size,
                                       drop_last=True)
            elif self.sampling_strategy == 'prioritised' or self.sampling_strategy == 'prioritised_clamp':
                priorities = self.priorities.reshape(-1)
                sampler = BatchSampler(WeightedRandomSampler(priorities, int(batch_size), replacement=False),
                                       mini_batch_size,
                                       drop_last=True)
            else:
                raise NotImplementedError

            self.last_updated_indices = []  # keep track of the latest sampled data points in the buffer
            for indices in sampler:
                if self.prioritised:
                    self.last_updated_indices.append(indices)

                obs_batch = torch.FloatTensor(self.obs_store.float()).reshape(-1, *self.obs_size)[indices].to(
                    self.device)
                hidden_state_batch = torch.FloatTensor(self.hidden_states_store.float()).reshape(
                    -1, self.hidden_state_size).to(self.device)
                act_batch = torch.FloatTensor(self.act_store.float()).reshape(-1)[indices].to(self.device)
                mask_batch = torch.FloatTensor(self.mask_store.float()).reshape(-1)[indices].to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_store.float()).reshape(-1)[indices].to(
                    self.device)
                val_batch = torch.FloatTensor(self.value_store.float()).reshape(-1)[indices].to(self.device)
                returns_batch = torch.FloatTensor(self.returns_store.float()).reshape(-1)[indices].to(self.device)

                if self.prioritised:
                    # importance sampling weights for Prioritised Experience Replay
                    weights_batch = self.get_per_weights().reshape(-1)[indices].to(self.device)
                else:
                    weights_batch = mask_batch

                if mode == 'il':
                    yield obs_batch, hidden_state_batch, act_batch, returns_batch, mask_batch, weights_batch
                elif mode == 'hippo':
                    adv_batch = torch.FloatTensor(self.adv_store.float()).reshape(-1)[indices].to(self.device)
                    yield obs_batch, hidden_state_batch, act_batch, returns_batch, \
                          mask_batch, log_prob_act_batch, val_batch, adv_batch
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

    def update_priorities(self, updates):
        """Update priorities for Prioritised Experience Replay"""
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
        """Directly compute the priorities in HIPPO rather than using the TD-error sampled from the buffer"""
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
        """Compute the Prioritised Experience Replay importance sampling weights. Used for IL"""
        n = self.get_n_valid_transitions()
        per_weights = ((1 / n) * (1 / self.priorities)) ** self.beta
        max_weight = torch.max(per_weights)
        per_weights = per_weights / max_weight

        return per_weights

    def _normalise_priorities(self):
        """Normalise the priorities"""
        summed = torch.sum(self.priorities)
        self.priorities = self.priorities / summed
        self.max_priority = torch.max(self.priorities)
        print(self.max_priority)


