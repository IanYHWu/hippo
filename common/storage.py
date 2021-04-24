import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler
import numpy as np
from collections import deque


class Storage:

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
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.hidden_states_batch[-1] = torch.from_numpy(last_hidden_state.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        rew_batch = self.rew_batch
        if use_gae:
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
            G = self.value_batch[-1]
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]

                G = rew + gamma * G * (1 - done)
                self.return_batch[i] = G

        if normalize_adv:
            self.adv_batch = (self.adv_batch - torch.mean(self.adv_batch)) / (torch.std(self.adv_batch) + 1e-8)

    def fetch_train_generator(self, mini_batch_size=None, recurrent=False):
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
        # If agent's policy is recurrent, data should be sampled along the time-horizon
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

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.obs_store = []
        self.hidden_states_store = []
        self.act_store = []
        self.rew_store = []
        self.returns_store = None
        self.trajectory_length = 0

    def store(self, obs, hidden_state, act, rew):
        self.obs_store.append(torch.from_numpy(obs.copy()))
        self.hidden_states_store.append(torch.from_numpy(hidden_state.copy()))
        self.act_store.append(torch.from_numpy(act.copy()))
        self.rew_store.append(torch.from_numpy(rew.copy()))

        self.trajectory_length += 1

    def store_last(self, last_obs, last_hidden_state):
        self.obs_store.append(torch.from_numpy(last_obs.copy()))
        self.hidden_states_store.append(torch.from_numpy(last_hidden_state.copy()))

    @staticmethod
    def _list_to_tensor(list_of_tensors):
        big_tensor = torch.stack(list_of_tensors)
        return big_tensor

    def _stores_to_tensors(self):
        self.obs_store = self._list_to_tensor(self.obs_store)
        self.hidden_states_store = self._list_to_tensor(self.hidden_states_store)
        self.act_store = self._list_to_tensor(self.act_store)
        self.rew_store = self._list_to_tensor(self.rew_store)

    def compute_returns(self, gamma=0.99):
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
        self.return_batch = torch.zeros(self.num_envs, self.num_steps)
        self.step = 0

    def store(self, obs, hidden_state, act, rew, done):
        self.obs_batch[:, self.step] = torch.from_numpy(obs.copy())
        self.hidden_states_batch[:, self.step] = torch.from_numpy(hidden_state.copy())
        self.act_batch[:, self.step] = torch.from_numpy(act.copy())
        self.rew_batch[:, self.step] = torch.from_numpy(rew.copy())
        self.done_batch[:, self.step] = torch.from_numpy(done.copy())

        self.step = (self.step + 1) % self.num_steps

    @staticmethod
    def _remove_cliffhangers_helper(input_tensor, mask, non_zero_rows):
        non_zero_tensor = input_tensor[non_zero_rows]
        shapes = len(non_zero_tensor.shape[2:])
        for i in range(0, shapes):
            mask = mask.unsqueeze(-1)

        return non_zero_tensor * mask

    def _remove_cliffhangers(self):
        mask, non_zero_rows = self._generate_masks()
        self.obs_batch = self._remove_cliffhangers_helper(self.obs_batch, mask, non_zero_rows)
        self.hidden_states_batch = self._remove_cliffhangers_helper(self.hidden_states_batch, mask, non_zero_rows)
        self.act_batch = self._remove_cliffhangers_helper(self.act_batch, mask, non_zero_rows)
        self.rew_batch = self._remove_cliffhangers_helper(self.rew_batch, mask, non_zero_rows)

    def _generate_masks(self):
        non_zero_rows = torch.abs(self.done_batch).sum(dim=1) > 0
        dones = self.done_batch[non_zero_rows]
        dones = dones.cpu()
        done_arr = dones.numpy()
        one_dones = np.argwhere(done_arr == 1)
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

    def _reshape_batches(self):
        self.obs_batch = torch.transpose(self.obs_batch, 1, 0)
        self.hidden_states_batch = torch.transpose(self.hidden_states_batch, 1, 0)
        self.act_batch = torch.transpose(self.act_batch, 1, 0)
        self.rew_batch = torch.transpose(self.rew_batch, 1, 0)
        self.done_batch = torch.transpose(self.done_batch, 1, 0)

    def compute_returns(self, gamma=0.99):
        self._remove_cliffhangers()
        G = 0
        for i in reversed(range(self.num_steps)):
            rew = self.rew_batch[:, i]
            G = rew + gamma * G
            self.return_batch[:, i] = G


class DemoReplayBuffer:

    def __init__(self, obs_size, hidden_state_size, device, max_samples):
        self.obs_size = obs_size
        self.hidden_state_size = hidden_state_size
        self.max_len = 0
        self.device = device
        self.max_samples = max_samples

    def store(self, demo_store):

        if isinstance(demo_store, DemoStorage):
            obs = demo_store.obs_store
            hidden_states = demo_store.hidden_states_store
            actions = demo_store.act_store
            rewards = demo_store.rew_store
            trajectory_len = demo_store.trajectory_length
            returns = demo_store.returns_store.reshape(trajectory_len, 1)
            demo_store.reset()

            mask = torch.ones(trajectory_len).reshape(trajectory_len, 1)

            if self.max_len == 0:
                self.obs_store = obs.unsqueeze(0)
                self.hidden_states_store = hidden_states.unsqueeze(0)
                self.act_store = actions.unsqueeze(0)
                self.rew_store = rewards.unsqueeze(0)
                self.returns_store = returns.unsqueeze(0)
                self.mask_store = mask.unsqueeze(0)
                self.max_len = trajectory_len
            else:
                if trajectory_len < self.max_len:
                    obs = self._pad_tensor(obs, self.max_len, pad_trajectory=True)
                    hidden_states = self._pad_tensor(hidden_states, self.max_len, pad_trajectory=True)
                    actions = self._pad_tensor(actions, self.max_len, pad_trajectory=True)
                    returns = self._pad_tensor(returns, self.max_len, pad_trajectory=True)
                    rewards = self._pad_tensor(rewards, self.max_len, pad_trajectory=True)
                    mask = self._pad_tensor(mask, self.max_len, pad_trajectory=True)

                elif trajectory_len > self.max_len:
                    self.obs_store = self._pad_tensor(self.obs_store, trajectory_len, pad_trajectory=False)
                    self.hidden_states_store = self._pad_tensor(self.hidden_states_store, trajectory_len,
                                                                pad_trajectory=False)
                    self.act_store = self._pad_tensor(self.act_store, trajectory_len, pad_trajectory=False)
                    self.returns_store = self._pad_tensor(self.returns_store, trajectory_len, pad_trajectory=False)
                    self.rew_store = self._pad_tensor(self.rew_store, trajectory_len, pad_trajectory=False)
                    self.mask_store = self._pad_tensor(self.mask_store, trajectory_len, pad_trajectory=False)
                    self.max_len = trajectory_len

                self.obs_store = self._add_to_buffer(obs, self.obs_store)
                self.hidden_states_store = self._add_to_buffer(hidden_states, self.hidden_states_store)
                self.act_store = self._add_to_buffer(actions, self.act_store)
                self.returns_store = self._add_to_buffer(returns, self.returns_store)
                self.rew_store = self._add_to_buffer(rewards, self.rew_store)
                self.mask_store = self._add_to_buffer(mask, self.mask_store)

            n_samples = self.get_buffer_n_samples()
            if n_samples > self.max_samples:
                self.obs_store = self.obs_store[:-1, :]
                self.hidden_states_store = self.hidden_states_store[:-1, :]
                self.act_store = self.act_store[:-1, :]
                self.rew_store = self.rew_store[:-1, :]
                self.returns_store = self.returns_store[:-1, :]
                self.mask_store = self.mask_store[:-1, :]

        else:
            self.obs_store = demo_store.obs_batch
            self.hidden_states_store = demo_store.hidden_states_batch
            self.act_store = demo_store.act_batch
            self.rew_store = demo_store.rew_batch
            self.returns_store = demo_store.return_batch
            self.mask_store = demo_store.mask_batch
            self.max_len = self.mask_store.shape[1]

    @staticmethod
    def _pad_tensor(input_tensor, new_len, pad_trajectory=False):
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
        new_buffer = torch.cat((trajectory.unsqueeze(0), buffer), dim=0)
        return new_buffer

    @staticmethod
    def _list_to_tensor(list_of_tensors):
        big_tensor = torch.stack(list_of_tensors)
        return big_tensor

    def il_demo_generator(self, batch_size, mini_batch_size, sample_method='uniform', recurrent=False):
        if not recurrent:
            if sample_method == 'uniform':
                sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                                       mini_batch_size, drop_last=True)
                for indices in sampler:
                    obs_batch = torch.FloatTensor(self.obs_store.float()).reshape(-1, *self.obs_size)[indices].to(
                        self.device)
                    hidden_state_batch = torch.FloatTensor(self.hidden_states_store.float()).reshape(
                        -1, self.hidden_state_size).to(self.device)
                    act_batch = torch.FloatTensor(self.act_store.float()).reshape(-1)[indices].to(self.device)
                    mask_batch = torch.FloatTensor(self.mask_store.float()).reshape(-1)[indices].to(self.device)
                    returns_batch = torch.FloatTensor(self.returns_store.float()).reshape(-1)[indices].to(self.device)
                    yield obs_batch, hidden_state_batch, act_batch, mask_batch, returns_batch
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def get_n_valid_transitions(self):
        valid_samples = torch.count_nonzero(self.mask_store)
        return valid_samples

    def get_buffer_n_samples(self):
        return len(self.act_store)

    def _compute_pi_v(self, actor_critic):
        log_prob_act_list = []
        val_list = []
        with torch.no_grad():
            for i in range(0, len(self.obs_store)):
                dist_batch, val_batch, _ = actor_critic(self.obs_store[i].squeeze(1).float().to(self.device),
                                                        self.hidden_states_store[i].float().to(self.device),
                                                        self.mask_store[i].float().to(self.device))
                log_prob_act_batch = dist_batch.log_prob(self.act_store[i].to(self.device))
                val_list.append(val_batch)
                log_prob_act_list.append(log_prob_act_batch)
            self.value_store = self._list_to_tensor(val_list).cpu()
            self.value_store *= self.mask_store.squeeze(-1)
            self.log_prob_act_store = self._list_to_tensor(log_prob_act_list).cpu()

    def compute_imp_samp_advantages(self, actor_critic, gamma=0.99, lmbda=0.95, normalise_adv=True):
        self._compute_pi_v(actor_critic)
        adv_list = []
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
            adv_mean = torch.sum(self.adv_store) / (self.get_n_valid_transitions() + 1e-8)
            mean_sq = torch.sum(self.adv_store ** 2) / (self.get_n_valid_transitions() + 1e-8)
            sq_mean = adv_mean ** 2
            adv_std = torch.sqrt(mean_sq - sq_mean + 1e-8)
            self.adv_store = (self.adv_store - adv_mean) / adv_std

    def imp_samp_demo_generator(self, batch_size, mini_batch_size, recurrent=False, sample_method='uniform'):
        if not recurrent:
            mask_weights = self.mask_store.squeeze(-1).reshape(-1)
            if sample_method == 'uniform':
                weights = mask_weights
                sampler = BatchSampler(WeightedRandomSampler(weights, batch_size, replacement=False), mini_batch_size,
                                       drop_last=True)
            elif sample_method == 'prioritised':
                weights = mask_weights * torch.abs(self.returns_store.reshape(-1) - self.value_store.reshape(-1))
                sampler = BatchSampler(WeightedRandomSampler(weights, batch_size, replacement=False), mini_batch_size,
                                       drop_last=True)
            elif sample_method == 'prioritised_clamp':
                weights = mask_weights * torch.clamp(self.returns_store.reshape(-1) - self.value_store.reshape(-1),
                                                     min=0)
                sampler = BatchSampler(WeightedRandomSampler(weights, batch_size, replacement=False), mini_batch_size,
                                       drop_last=True)
            else:
                raise NotImplementedError

            for indices in sampler:
                obs_batch = torch.FloatTensor(self.obs_store.float()).reshape(-1, *self.obs_size)[indices].to(
                    self.device)
                hidden_state_batch = torch.FloatTensor(self.hidden_states_store.float()).reshape(
                    -1, self.hidden_state_size).to(self.device)
                act_batch = torch.FloatTensor(self.act_store.float()).reshape(-1)[indices].to(self.device)
                mask_batch = torch.FloatTensor(self.mask_store.float()).reshape(-1)[indices].to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_store.float()).reshape(-1)[indices].to(
                    self.device)
                val_batch = torch.FloatTensor(self.value_store.float()).reshape(-1)[indices].to(self.device)
                adv_batch = torch.FloatTensor(self.adv_store.float()).reshape(-1)[indices].to(self.device)
                returns_batch = torch.FloatTensor(self.returns_store.float()).reshape(-1)[indices].to(self.device)
                yield obs_batch, hidden_state_batch, act_batch, returns_batch, \
                      mask_batch, log_prob_act_batch, val_batch, adv_batch

        else:
            raise NotImplementedError


if __name__ == '__main__':
    rb = DemoReplayBuffer(obs_size=(4, 4), hidden_state_size=4, device='cpu', max_samples=2)
    ds = DemoStorage(device='cpu')
    for j in range(0, 3):
        act = np.array([np.random.randint(3)], dtype=float)
        rew = np.array([np.random.randint(10)], dtype=float)
        obs = np.random.rand(4, 4)
        obs = obs.astype(float)
        hidden = np.random.rand(4, 4)
        hidden = hidden.astype(float)
        ds.store(obs, hidden, act, rew)

    ds.compute_returns()
    rb.store(ds)
    print(rb.act_store)
    for j in range(0, 5):
        act = np.array([np.random.randint(3)], dtype=float)
        rew = np.array([np.random.randint(10)], dtype=float)
        obs = np.random.rand(4, 4)
        obs = obs.astype(float)
        hidden = np.random.rand(4, 4)
        hidden = hidden.astype(float)
        ds.store(obs, hidden, act, rew)

    ds.compute_returns()
    rb.store(ds)
    print(rb.act_store)
    for j in range(0, 2):
        act = np.array([np.random.randint(3)], dtype=float)
        rew = np.array([np.random.randint(10)], dtype=float)
        obs = np.random.rand(4, 4)
        obs = obs.astype(float)
        hidden = np.random.rand(4, 4)
        hidden = hidden.astype(float)
        ds.store(obs, hidden, act, rew)
    ds.compute_returns()
    rb.store(ds)
    print("final store:")
    print(rb.act_store)

    generator = rb.il_demo_generator(batch_size=6, mini_batch_size=3, sample_method='uniform', recurrent=False)
    for sample in generator:
        obs_batch, hidden_state_batch, act_batch, mask_batch, returns_batch = sample
        print(act_batch)
        # print(obs_batch)
        # print(act_batch)
        # print(hidden_state_batch)
        # print(act_batch)
        # print(mask_batch)
