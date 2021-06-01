import random
import numpy as np


class DemoScheduler:
    """Demonstration Scheduler - used for predefined schedules for querying and learning"""

    def __init__(self, args, params, rollout, schedule='linear'):
        self.num_timesteps = args.num_timesteps
        self.num_demos = params.num_demo_queries
        self.demo_schedule = schedule
        self.demo_learn_ratio = params.demo_learn_ratio
        self.hot_start = params.hot_start
        self.rollout = rollout
        self.n_envs = params.n_envs
        self.n_steps = params.n_steps
        self.multi = params.demo_multi

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
        learn_every = (1 / self.demo_learn_ratio) * self.n_envs * self.n_steps
        if not self.multi and self.buffer_empty:
            return False
        else:
            if curr_timestep > ((self.demo_learn_count + 1) * learn_every):
                self.demo_learn_count += 1
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

    def get_seeds(self, demos_per_step=2):
        envs = np.random.randint(0, self.n_envs, demos_per_step)
        seeds = []
        for env in envs:
            seed = self.rollout.info_batch[-1][env]['level_seed']
            seeds.append(seed)
        return seeds

    def get_stats(self):
        return self.query_count, self.demo_learn_count






