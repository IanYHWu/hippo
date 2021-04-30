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
    """Demonstration replay buffer. Used for both demo_multi = True and demo_multi = False
    CURRENTLY BEING REWRITTEN"""

    def __init__(self):
        pass


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
