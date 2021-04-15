import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
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

    def _list_to_tensor(self, list_of_tensors):
        big_tensor = torch.stack(list_of_tensors)
        return big_tensor

    def stores_to_tensors(self):
        self.obs_store = self._list_to_tensor(self.obs_store)
        self.hidden_states_store = self._list_to_tensor(self.hidden_states_store)
        self.act_store = self._list_to_tensor(self.act_store)
        self.rew_store = self._list_to_tensor(self.rew_store)

    def compute_returns(self, gamma=0.99):
        self.returns_store = torch.zeros(self.trajectory_length)
        G = 0
        for i in reversed(range(self.trajectory_length)):
            rew = self.rew_store[i]
            G = rew + gamma * G
            self.returns_store[i] = G


class DemoReplayBuffer:

    def __init__(self, obs_size, hidden_state_size, device):
        self.obs_size = obs_size
        self.hidden_state_size = hidden_state_size
        self.max_len = 0
        self.device = device

    def store(self, obs, hidden_states, actions, returns, trajectory_len):
        mask = torch.ones(trajectory_len)

        if self.max_len == 0:
            self.obs_store = obs.unsqueeze(0)
            self.hidden_states_store = hidden_states.unsqueeze(0)
            self.act_store = actions.unsqueeze(0)
            self.returns_store = returns.unsqueeze(0)
            self.mask_store = mask.unsqueeze(0)
            self.max_len = trajectory_len
        else:
            if trajectory_len < self.max_len:
                obs = self._pad_tensor(obs, self.max_len, pad_trajectory=True)
                hidden_states = self._pad_tensor(hidden_states, self.max_len, pad_trajectory=True)
                actions = self._pad_tensor(actions, self.max_len, pad_trajectory=True)
                returns = self._pad_tensor(returns, self.max_len, pad_trajectory=True)
                mask = self._pad_tensor(mask, self.max_len, pad_trajectory=True)

            elif trajectory_len > self.max_len:
                self.obs_store = self._pad_tensor(self.obs_store, self.max_len, pad_trajectory=False)
                self.hidden_states_store = self._pad_tensor(self.hidden_states_store, self.max_len, pad_trajectory=False)
                self.act_store = self._pad_tensor(self.act_store, self.max_len, pad_trajectory=False)
                self.returns_store = self._pad_tensor(self.returns_store, self.max_len, pad_trajectory=False)
                self.mask_store = self._pad_tensor(self.mask_store, self.max_len, pad_trajectory=False)
                self.max_len = trajectory_len

            self._add_to_buffer(obs, self.obs_store)
            self._add_to_buffer(hidden_states, self.hidden_states_store)
            self._add_to_buffer(actions, self.act_store)
            self._add_to_buffer(returns, self.returns_store)
            self._add_to_buffer(mask, self.mask_store)

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

    def fetch_demo_generator(self, mini_batch_size, sample_method='uniform', recurrent=False):
        buffer_size = self.max_len * len(self.obs_store)
        if not recurrent:
            if sample_method == 'uniform':
                sampler = BatchSampler(SubsetRandomSampler(range(buffer_size)),
                                       mini_batch_size, drop_last=True)
                for indices in sampler:
                    obs_batch = torch.FloatTensor(self.obs_store.float()).reshape(-1, *self.obs_size)[indices].to(self.device)
                    hidden_state_batch = torch.FloatTensor(self.hidden_states_store.float()).reshape(
                        -1, *self.hidden_state_size).to(self.device)
                    act_batch = torch.FloatTensor(self.act_store.float()).reshape(-1)[indices].to(self.device)
                    mask_batch = torch.FloatTensor(self.mask_store.float()).reshape(-1)[indices].to(self.device)
                    returns_batch = torch.FloatTensor(self.returns_store.float()).reshape(-1)[indices].to(self.device)
                    yield obs_batch, hidden_state_batch, act_batch, mask_batch, returns_batch
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def get_buffer_capacity(self):
        return len(self.obs_store) * self.max_len


if __name__ == '__main__':
    rb = DemoReplayBuffer(obs_size=(4, 4), hidden_state_size=(4, 4), device='cpu')
    for i in range(0, 3):
        ds = DemoStorage(device='cpu')
        for j in range(0, 3):
            act = np.array([np.random.randint(3)], dtype=float)
            rew = np.array([np.random.randint(10)], dtype=float)
            obs = np.random.rand(4, 4)
            obs = obs.astype(float)
            hidden = np.random.rand(4, 4)
            hidden = hidden.astype(float)
            ds.store(obs, hidden, act, rew)
        for j in range(0, 5):
            act = np.array([np.random.randint(3)], dtype=float)
            rew = np.array([np.random.randint(10)], dtype=float)
            obs = np.random.rand(4, 4)
            obs = obs.astype(float)
            hidden = np.random.rand(4, 4)
            hidden = hidden.astype(float)
            ds.store(obs, hidden, act, rew)

        ds.stores_to_tensors()
        ds.compute_returns()

        rb.store(ds.obs_store, ds.hidden_states_store, ds.act_store, ds.returns_store, ds.trajectory_length)

    generator = rb.fetch_demo_generator(mini_batch_size=3, sample_method='uniform', recurrent=False)
    for sample in generator:
        obs_batch, hidden_state_batch, act_batch, mask_batch, returns_batch = sample
        print(act_batch)
        # print(obs_batch)
        # print(act_batch)
        # print(hidden_state_batch)
        # print(act_batch)
        # print(mask_batch)








