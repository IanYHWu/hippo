"""
This module implements different controllers, which are objects that decide (1) when to query a demonstration (2) when
to learn from the demonstration buffer and (3) what the demonstration seeds should be
"""

import numpy as np
from collections import deque
import torch


class BaseController:
    """Base class for the controller"""

    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.pre_load = params.pre_load
        self.pre_load_seed_sampling = params.pre_load_seed_sampling
        self.num_levels = args.num_levels

    def initialise(self):
        """Initialise scores used by the controller"""
        pass

    def learn_from_env(self):
        """Learn from the environment"""
        pass

    def learn_from_demos(self, curr_timestep):
        """Learn from demonstrations"""
        pass

    def get_new_seeds(self):
        """Get seeds to sample new demos of"""
        pass

    def get_preload_seeds(self):
        """Get seeds for pre-loading"""
        if self.pre_load_seed_sampling == 'random':
            # sample randomly from the training seeds
            seeds = np.random.randint(0, self.num_levels, self.pre_load)
            return seeds.tolist()
        elif self.pre_load_seed_sampling == 'fixed':
            # sample seeds from 0 to pre_load
            if self.pre_load > self.num_levels:
                print("Warning: evaluation seeds used for pre-loading")
                print("Consider reducing the number of pre-load trajectories")
            seeds = [i for i in range(0, self.pre_load)]
            return seeds
        else:
            raise NotImplementedError

    def get_learn_indices(self):
        """Get the demo storage indices of demos to learn from"""
        pass

    def update(self):
        """Update the controller"""
        pass

    def get_stats(self):
        """Return demo learning statistics"""
        pass


class DemoScheduler(BaseController):
    """Demonstration Scheduler - used for predefined schedules for querying and learning

    Attributes:
        num_timesteps: total number of training timesteps
        num_demos:  total number of demos to query
        demo_schedule: demo schedule type
        demo_learn_ratio: r, ratio of env-learning steps to demo-learning steps
        pre_load: number of demos to pre-load
        rollout: env rollout object
        n_envs: number of envs per rollout
        n_steps: number of env-learning steps per rollout
        demo_sampling: seed sampling strategy
        pre_load_seed_sampling: hot start seed sampling strategy
        num_demo_seeds: number of seeds per demo to sample
        num_levels: number of training levels
        demo_learn_count: number of demo learning steps done so far
    """

    def __init__(self, args, params, rollout, schedule='linear', demo_storage=None):
        super().__init__(args, params)

        self.controller_type = "simple_schedule"
        self.num_timesteps = args.num_timesteps
        self.demo_schedule = schedule
        self.demo_learn_ratio = params.demo_learn_ratio
        self.rollout = rollout  # environment rollout
        self.demo_storage = demo_storage
        self.n_envs = params.n_envs
        self.n_steps = params.n_steps
        self.demo_sampling = params.demo_sampling
        self.num_demo_seeds = params.num_learn_demos
        self.demo_limit = params.demo_limit
        self.demo_levels = params.demo_levels
        self.demo_learn_count = 0
        self.replace_count = 0
        self.replace = params.replace

    def learn_from_demos(self, curr_timestep):
        """Learn from the replay buffer"""
        if self.demo_limit:
            if curr_timestep > self.demo_limit:
                return False
        learn_every = (1 / self.demo_learn_ratio) * self.n_envs * self.n_steps
        if curr_timestep > ((self.demo_learn_count + 1) * learn_every):
            self.demo_learn_count += 1
            return True
        else:
            return False

    def learn_from_env(self):
        """Learn from the environment, always True"""
        return True

    def get_new_seeds(self, replace_mode=False):
        """Sample a list of seeds - used for gathering trajectories"""
        if replace_mode:
            n = self.demo_storage.get_n_samples()
            if self.replace > n:
                replace_num = n
            else:
                replace_num = self.replace
            replace_indices = np.random.choice(n, replace_num).tolist()
            num_seeds = self.replace
        else:
            replace_indices = []
            num_seeds = self.num_demo_seeds

        if self.demo_sampling == 'latest':
            # sample seeds based on the seeds used in the latest env rollout
            envs = np.random.randint(0, self.n_envs, num_seeds)
            seeds = []
            for env in envs:
                seed = self.rollout.info_batch[-1][env]['level_seed']
                seeds.append(seed)
            return seeds, replace_indices
        elif self.demo_sampling == 'random':
            # sample seeds randomly from the permitted levels
            if not self.demo_levels:
                max_demo_seed = 2147483648
            else:
                max_demo_seed = self.demo_levels
            seeds = np.random.randint(0, max_demo_seed, num_seeds)
            return seeds.tolist(), replace_indices
        else:
            raise NotImplementedError

    def get_learn_indices(self):
        """Get a list of storage indices - used for sampling demonstrations from demo_storage"""
        n = self.demo_storage.get_n_samples()
        if self.demo_sampling == 'random':
            indices = np.random.randint(0, n, self.num_demo_seeds)
            return indices.tolist()
        elif self.demo_sampling == 'fixed':
            envs = np.random.randint(0, self.n_envs, self.num_demo_seeds)
            seeds = []
            indices = []
            for env in envs:
                seed = self.rollout.info_batch[-1][env]['level_seed']
                seeds.append(seed)
            for seed in seeds:
                index_list = self.demo_storage.seed_to_ind[seed]
                ind = np.random.choice(index_list)
                indices.append(ind)
            return indices

    def get_stats(self):
        """Get the latest demo stats"""
        return self.demo_learn_count


class BanditController(BaseController):
    """Bandit controller - uses a bandit to decide when to do an env learning and when to do a demo learning step.
    Also decides which demos to sample given a demo learning step is selected

    Attributes:
        controller_type: specifies what type of controller this is
        bandit: bandit object to use
        rollout: env rollout
        max_samples: max storage samples allowed
        value_losses: list of values losses associated with each index of the storage
        last_demo_indices: indices of last demos learned from
        demo_storage: demo_storage object
        demo_buffer: demo_buffer object
        actor_critic: policy to learn
        learn_from: which arm was last used (0/1)
        scoring_method: method used to assign weights to individual demos
        temperature: temperature coefficient to control weighting of scores
        num_learn_demos: number of samples learned from per demo learning step
        rho: coefficient controlling weight of demo feedback relative to env feedback
        num_store_demos: current number of demos in demo_store
        env_val_loss_window: sliding window of env value losses
        demo_val_loss_window: sliding window of demo value losses
        env_learn_count: number of env learning steps performed
        demo_learn_count: number of demo learning steps performed
    """

    def __init__(self, args, params, rollout, demo_storage, demo_buffer, actor_critic):
        super().__init__(args, params)

        self.controller_type = "bandit"
        self.bandit = EXP3()
        self.rollout = rollout  # env rollout
        self.max_samples = params.demo_store_max_samples
        self.value_losses = None
        self.staleness = None
        self.last_demo_indices = None
        self.demo_storage = demo_storage
        self.demo_buffer = demo_buffer
        self.actor_critic = actor_critic
        self.learn_from = None
        self.scoring_method = params.scoring_method
        self.temperature = params.temperature
        self.num_learn_demos = params.num_learn_demos
        self.demo_sampling_replace = params.demo_sampling_replace
        self.rho = params.rho
        self.mu = params.mu

        self.num_store_demos = 0
        self.env_val_loss_window = deque(maxlen=5)
        self.demo_val_loss_window = deque(maxlen=5)
        self.env_learn_count = 0
        self.demo_learn_count = 0

    def initialise(self):
        """Initialise the value loss scores of individual demos"""
        self.num_store_demos = self.demo_storage.get_n_samples()
        print("Number of valid trajectories in store: {}".format(self.num_store_demos))
        self.value_losses = np.zeros(self.num_store_demos)
        self.staleness = np.zeros(self.num_store_demos)
        if self.num_learn_demos > self.num_store_demos:
            self.num_learn_demos = self.num_store_demos
            print("Warning - number of valid demonstrations is less than the demonstration learning number")
        i = 0
        demo_rewards = self.demo_storage.env_rewards
        print("Demonstration rewards: {}".format(demo_rewards))
        while i < self.num_store_demos:
            start_index = i
            j = 0
            while j < self.demo_buffer.max_samples and i < self.num_store_demos:
                demo_obs_t, demo_hidden_state_t, demo_act_t, demo_rew_t, demo_done_t = self.demo_storage.get_demo_trajectory(
                    store_index=i)
                self.demo_buffer.store(demo_obs_t, demo_hidden_state_t, demo_act_t, demo_rew_t, demo_done_t)
                j += 1
                i += 1
            self.demo_buffer.compute_pi_v(self.actor_critic)
            self.demo_buffer.compute_estimates(self.actor_critic)
            value_losses = self.demo_buffer.compute_value_losses().numpy()
            self.value_losses[start_index: i] = value_losses
            self.demo_buffer.reset()

    def learn_from_env(self):
        """Learn from the environment"""
        self._sample_bandit()
        if self.learn_from == 0:
            self.env_learn_count += 1
            return True
        else:
            return False

    def learn_from_demos(self, curr_timestep):
        """Learn from demonstrations"""
        if self.learn_from == 1:
            self.demo_learn_count += 1
            return True
        else:
            return False

    def _sample_bandit(self):
        """Sample the bandit"""
        learn_from = self.bandit.sample()
        self.learn_from = learn_from

    def _update_bandit(self):
        """Update the bandit with feedback - helper method"""
        if self.learn_from == 0:
            feedback = torch.mean(torch.abs(self.rollout.adv_batch)).item()
            self.env_val_loss_window.append(feedback)
        else:
            feedback = self.rho * np.mean(self.demo_buffer.compute_value_losses().numpy())
            self.demo_val_loss_window.append(feedback)
        self.bandit.update(feedback)

    def _update_value_losses(self):
        """Update the value loss scores"""
        latest_value_losses = self.demo_buffer.compute_value_losses().numpy()
        for index, i in enumerate(self.last_demo_indices):
            self.value_losses[i] = latest_value_losses[index]
            self.staleness[i] = self.demo_learn_count

    def update(self):
        """Update the bandit"""
        self._update_bandit()
        if self.learn_from == 1:
            self._update_value_losses()

    def get_learn_indices(self):
        """Decide which demos to learn from"""
        if self.scoring_method == 'rank':  # rank prioritisation
            ranking = np.argsort(self.value_losses * -1)
            val_scores = (1 / (np.arange(0, len(ranking)) + 1)) ** self.temperature
            val_scores /= np.sum(val_scores)
            ranking, val_p = zip(*sorted(zip(ranking, val_scores)))
            c = self.demo_learn_count
            stale_score = c - self.staleness
            stale_p = stale_score / np.sum(stale_score)
            P = (1 - self.rho) * np.array(list(val_p)) + self.rho * stale_p
            if self.demo_sampling_replace:
                indices = np.random.choice(self.num_store_demos, self.num_learn_demos, replace=True, p=P)
            else:
                indices = np.random.choice(self.num_store_demos, self.num_learn_demos, replace=False, p=P)
            self.last_demo_indices = indices
            return indices.tolist()
        else:
            raise NotImplementedError

    def get_new_seeds(self, replace_mode=True):
        """Sample a list of seeds - used for gathering trajectories"""
        return [], []

    def get_stats(self):
        """Get the demo stats"""
        choice_ratio_list = self.bandit.choice_ratio_list
        demo_ratio = np.sum(choice_ratio_list) / len(choice_ratio_list)
        stats = {"env learning steps": self.env_learn_count, "demo learning steps": self.demo_learn_count,
                 "env value loss window": 0.0 if len(self.env_val_loss_window) == 0 else np.round(
                     np.mean(self.env_val_loss_window), 3),
                 "demo value loss window": 0.0 if len(self.demo_val_loss_window) == 0 else np.round(
                     np.mean(self.demo_val_loss_window), 3),
                 "ratio (env/demo)": np.round(demo_ratio, 3),
                 "stale median": np.median(self.demo_learn_count - self.staleness),
                 "stale max": np.max(self.demo_learn_count - self.staleness),
                 "stale min": np.min(self.demo_learn_count - self.staleness)}
        return stats


class ValueLossScheduler(BanditController):
    """Scheduler that prioritises demonstrations based on their latest value losses. Halfway between the linear scheduler
    and the bandit controller. Inherits from BanditController"""

    def __init__(self, args, params, rollout, demo_storage, demo_buffer, actor_critic, schedule='linear'):
        super().__init__(args, params, rollout, demo_storage, demo_buffer, actor_critic)

        self.controller_type = "value_loss_schedule"
        self.num_timesteps = args.num_timesteps
        self.demo_schedule = schedule
        self.demo_learn_ratio = params.demo_learn_ratio
        self.rollout = rollout  # environment rollout
        self.demo_storage = demo_storage
        self.n_envs = params.n_envs
        self.n_steps = params.n_steps
        self.demo_sampling = params.demo_sampling
        self.num_demo_seeds = params.num_learn_demos
        self.demo_limit = params.demo_limit
        self.demo_levels = params.demo_levels
        self.demo_learn_count = 0
        self.replace_count = 0
        self.replace = params.replace

    def learn_from_demos(self, curr_timestep):
        """Learn from the replay buffer"""
        if self.demo_limit:
            if curr_timestep > self.demo_limit:
                return False
        learn_every = (1 / self.demo_learn_ratio) * self.n_envs * self.n_steps
        if curr_timestep > ((self.demo_learn_count + 1) * learn_every):
            self.demo_learn_count += 1
            self.learn_from = 1
            return True
        else:
            return False

    def learn_from_env(self):
        """Learn from the environment, always True"""
        self.learn_from = 0
        return True

    def update(self):
        """Update the value losses"""
        if self.learn_from == 1:
            self._update_value_losses()
            self.demo_val_loss_window.append(np.mean(self.demo_buffer.compute_value_losses().numpy()))

    def get_stats(self):
        stats = {"demo learning steps": self.demo_learn_count,
                 "demo value loss window": 0.0 if len(self.demo_val_loss_window) == 0 else np.mean(self.demo_val_loss_window),
                 "stale median": np.median(self.staleness),
                 "stale max": np.max(self.staleness),
                 "stale min": np.min(self.staleness)}
        return stats


class EXP3:
    """Implementation of the Exp3 Adverserial bandit

    Attributes:
        arms: tuple of indices associated with each distinct action
        gamma: weighting parameter - gamma = 1 implies uniform random
        demean_window: window for mean tracking
        norm_window: window for normalisation
        rand_init: random initialisation steps
        store_choices: store the choices made
        track_choice_ratios: track the choice ratios - tracks based on recency
    """

    def __init__(self,
                 arms=(0, 1),  # default - two arms
                 gamma=0.3,
                 demean_window=5,  # subtract the mean
                 norm_window=5,  # how many to normalize by [0,1]
                 rand_init=5,
                 store_choices=False,
                 track_choice_ratios=True):

        self.arms = arms
        self.arm = 0
        self.w = {i: 1 for i in range(len(self.arms))}
        self.gamma = gamma
        self.norm_window = norm_window
        self.demean_window = demean_window
        self.rand_init = rand_init
        self.probs = [1 / len(arms)] * len(arms)
        self.raw_scores = deque(maxlen=demean_window)
        self.feedback = deque(maxlen=norm_window)
        self.num_pulls = 0
        self.store_choices = store_choices
        if store_choices:
            self.choices = []  # store history of choices
        self.track_choice_ratios = track_choice_ratios
        if track_choice_ratios:
            self.choice_ratio_list = deque(maxlen=30)

    def sample(self):
        """Sample from the bandit"""
        self.probs = [(1 - self.gamma) * x / (sum(self.w.values()) + 1e-6) + self.gamma / len(self.arms) for x in
                      self.w.values()]  # self.w are the bandit weights
        self.probs /= np.sum(self.probs)  # normalisation, in case they don't sum to one
        if len(self.feedback) < self.rand_init:
            # cycle between arms during random initialisation phase
            if self.arm == 0:
                self.arm = 1
            else:
                self.arm = 0
        else:
            self.arm = np.random.choice(range(0, len(self.probs)), p=self.probs)
        self.num_pulls += 1
        if self.store_choices:
            self.choices.append(self.arm)
        if self.track_choice_ratios:
            self.choice_ratio_list.append(self.arm)

        return self.arm

    def update(self, feedback):
        """Update the bandit using feedback"""
        # need to normalize score
        # since this is non-stationary, subtract the mean of the previous window
        self.raw_scores.append(feedback)
        if self.demean_window > 0:
            feedback -= np.mean(self.raw_scores)  # subtract trailing mean
        self.feedback.append(feedback)
        normalised = (feedback - np.min(self.feedback)) / \
                     (np.max(self.feedback) - np.min(self.feedback) + 1e-6)  # normalized to [0,1]
        for idx, arm in enumerate(self.arms):
            if arm == self.arm:
                x = normalised / self.probs[idx]
            else:
                x = 0
            self.w[arm] *= np.exp((self.gamma * x) / len(self.arms))  # weight update
        sum_w = sum(self.w.values())
        for i, _ in self.w.items():
            self.w[i] = self.w[i] / sum_w
