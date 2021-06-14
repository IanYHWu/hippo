"""
This module implements different controllers, which are objects that decide (1) when to query a demonstration (2) when
to learn from the demonstration buffer and (3) what the demonstration seeds should be
"""

import numpy as np


class DemoScheduler:
    """Demonstration Scheduler - used for predefined schedules for querying and learning

    Attributes:
        num_timesteps: total number of training timesteps
        num_demos:  total number of demos to query
        demo_schedule: demo schedule type
        demo_learn_ratio: r, ratio of env-learning steps to demo-learning steps
        hot_start: number of demos to pre-load
        rollout: env rollout object
        n_envs: number of envs per rollout
        n_steps: number of env-learning steps per rollout
        seed_sampling: seed sampling strategy
        hot_start_seed_sampling: hot start seed sampling strategy
        num_demo_seeds: number of seeds per demo to sample
        replay: flag to indicate Variant I (False) or II (True)
        num_levels: number of training levels
        query_count: number of queries made so far
        demo_learn_count: number of demo learning steps done so far
    """

    def __init__(self, args, params, rollout, schedule='linear'):
        self.num_timesteps = args.num_timesteps
        self.num_demos = params.num_demo_queries
        self.demo_schedule = schedule
        self.demo_learn_ratio = params.demo_learn_ratio
        self.hot_start = params.hot_start
        self.rollout = rollout
        self.n_envs = params.n_envs
        self.n_steps = params.n_steps
        self.seed_sampling = params.seed_sampling
        self.hot_start_seed_sampling = params.hot_start_seed_sampling
        self.num_demo_seeds = params.num_demo_seeds
        self.replay = params.use_replay
        self.num_levels = args.num_levels
        self.demo_limit = params.demo_limit

        self.query_count = 0
        self.demo_learn_count = 0
        if self.hot_start:
            self.buffer_empty = False
        else:
            self.buffer_empty = True

    def query_demonstrator(self, curr_timestep):
        """Get a trajectory from the demonstrator"""
        if self.demo_schedule == 'linear':
            return self._linear_schedule(curr_timestep)
        else:
            raise NotImplementedError

    def learn_from_demos(self, curr_timestep, always_learn=False):
        """Learn from the replay buffer"""
        if always_learn:
            return True
        if self.demo_limit:
            if curr_timestep > self.demo_limit:
                print("limit")
                return False
        learn_every = (1 / self.demo_learn_ratio) * self.n_envs * self.n_steps
        if self.replay and self.buffer_empty:
            return False
        else:
            if curr_timestep > ((self.demo_learn_count + 1) * learn_every):
                self.demo_learn_count += 1
                if not self.replay:
                    self.query_count += 1
                return True
            else:
                return False

    def _linear_schedule(self, curr_timestep):
        """Linear Scheduler"""
        demo_every = self.num_timesteps // self.num_demos
        if curr_timestep > ((self.query_count + 1) * demo_every):
            self.buffer_empty = False
            self.query_count += 1
            return True
        else:
            return False

    def get_seeds(self, hot_start_mode=False):
        """Sample a list of seeds"""
        if hot_start_mode:  # sample seeds for hot-start
            if self.hot_start_seed_sampling == 'random':
                # sample randomly from the training seeds
                seeds = np.random.randint(0, self.num_levels, self.hot_start)
                return seeds.tolist()
            elif self.hot_start_seed_sampling == 'fixed':
                # sample seeds from 0 to hot_start
                if self.hot_start > self.num_levels:
                    print("Warning: evaluation seeds used for hot start")
                    print("Consider reducing the number of hot start trajectories")
                seeds = [i for i in range(0, self.hot_start)]
                return seeds
            else:
                raise NotImplementedError
        else:
            if self.seed_sampling == 'latest':
                # sample seeds based on the seeds used in the latest env rollout
                envs = np.random.randint(0, self.n_envs, self.num_demo_seeds)
                seeds = []
                for env in envs:
                    seed = self.rollout.info_batch[-1][env]['level_seed']
                    seeds.append(seed)
                return seeds
            elif self.seed_sampling == 'random':
                # sample seeds randomly from the training levels
                seeds = np.random.randint(0, self.num_levels, self.num_demo_seeds)
                return seeds.tolist()
            else:
                raise NotImplementedError

    def get_stats(self):
        """Get the latest demo stats"""
        return self.query_count, self.demo_learn_count, 0.0


class GAEController:
    """Controller based on a running average of the average absolute GAE

    Attributes:
        rollout: env rollout object
        args: argparse object
        n_envs: number of envs in env rollout
        n_steps: number of steps per env rollout
        adv_tracker: tracks the average advantage for each env (online)
        count_tracker: tracks the number of samples in each env (online)
        running_avg: running average of the average absolute GAE
        weighting_coef: recency weighting coefficient for the running average
        rho: threshold coefficient, for demonstration requesting
        t: tracks number of rollouts performed
        demo_seeds: list of demo seeds
        demo_learn_ratio: r, ratio of env-learning steps to demo-learning steps
        num_timesteps: total number of training timesteps
        num_demos:  total number of demos to query
        demo_learn_ratio: r, ratio of env-learning steps to demo-learning steps
        hot_start: number of demos to pre-load
        rollout: env rollout object
        query_count: number of queries made so far
        demo_learn_count: number of demo learning steps done so far
    """

    def __init__(self, args, params, rollout):
        self.rollout = rollout
        self.args = args
        self.n_envs = params.n_envs
        self.n_steps = params.n_steps
        self.adv_tracker = np.zeros(self.n_envs)
        self.count_tracker = np.zeros(self.n_envs)
        self.running_avg = 0.0
        self.weighting_coef = params.weighting_coef
        self.rho = params.rho
        self.t = 0
        self.demo_seeds = None

        self.demo_learn_ratio = params.demo_learn_ratio
        self.num_timesteps = args.num_timesteps
        self.num_demos = params.num_demo_queries
        self.hot_start = params.hot_start
        self.query_count = 0
        self.demo_learn_count = 0
        if self.hot_start:
            self.buffer_empty = False
        else:
            self.buffer_empty = True

    def _compute_avg_adv(self):
        """Compute the average advantage of each trajectory in the rollout in an online manner. By doing this online,
        we can track the average advantages for trajectories spanning multiple rollouts"""
        # extract the relevant trajectories from the rollout
        adv_batch = self.rollout.adv_batch
        done_batch = self.rollout.done_batch
        info_batch = self.rollout.info_batch
        adv_list = []
        seed_list = []
        for i in range(self.n_envs):
            for j in range(self.n_steps):
                if not done_batch[j][i]:
                    # update the trackers in an online manner
                    self.count_tracker[i] += 1
                    self.adv_tracker[i] += (1 / self.count_tracker[i]) * (abs(adv_batch[j][i]) - self.adv_tracker[i])
                else:
                    # once a trajectory is done, store the average absolute GAE and the seed, then reset the trackers
                    seed = info_batch[j][i]['level_seed']
                    adv_list.append(self.adv_tracker[i])
                    seed_list.append(seed)
                    self.adv_tracker[i] = 0
                    self.count_tracker[i] = 0

        return adv_list, seed_list

    def _update_running_avg(self, adv_list):
        """Update the running average of the average absolute GAE"""
        if adv_list:
            mean_adv_t = np.mean(adv_list)
            if self.t == 0:
                self.running_avg = mean_adv_t
                self.t += 1
            else:
                self.running_avg = self.weighting_coef * mean_adv_t + (1 - self.weighting_coef) * self.running_avg
                self.t += 1

    def _generate_demo_seeds(self):
        """Generate a list of seeds to extract demonstrations for"""
        demo_seeds = []
        adv_list, seed_list = self._compute_avg_adv()  # get the average absolute GAEs
        self._update_running_avg(adv_list)  # update the running average
        for adv, seed in zip(adv_list, seed_list):
            # if the average absolute GAE of an env trajectory is rho higher than the running average, include that seed
            if adv > self.rho * self.running_avg:
                demo_seeds.append(seed)

        self.demo_seeds = demo_seeds

    def query_demonstrator(self, curr_timestep):
        """Get a trajectory from the demonstrator"""
        self._generate_demo_seeds()
        # if we have a seed to learn from, return True
        if self.demo_seeds:
            self.query_count += len(self.demo_seeds)
            return True
        else:
            return False

    def learn_from_demos(self, curr_timestep, always_learn=False):
        """Learn from the replay buffer"""
        if always_learn:
            return True
        learn_every = (1 / self.demo_learn_ratio) * self.n_envs * self.n_steps
        if self.buffer_empty:
            return False
        else:
            if curr_timestep > ((self.demo_learn_count + 1) * learn_every):
                self.demo_learn_count += 1
                return True
            else:
                return False

    def get_seeds(self):
        """Get the list of seeds to learn from"""
        return self.demo_seeds

    def get_stats(self):
        """Get the latest demo statistics"""
        return self.query_count, self.demo_learn_count, self.running_avg


























