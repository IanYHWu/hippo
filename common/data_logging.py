import numpy as np
import pandas as pd
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import os
import torch
import json
import yaml


class Logger(object):

    def __init__(self, args, params):
        self.start_time = time.time()
        self.params = params
        self.args = args
        self.root = self.args.log_dir
        self.name = self.args.name
        self.root_path = None
        self.checkpoint_path = None
        self.log_path = None
        self.n_envs = params.n_envs

        self._make_dirs()
        self._initialise_writer()

        self.episode_rewards = []
        for _ in range(params.n_envs):
            self.episode_rewards.append([])
        self.episode_len_buffer = deque(maxlen=40)
        self.episode_reward_buffer = deque(maxlen=40)

        self.log = pd.DataFrame(columns=['timesteps', 'wall_time', 'num_episodes',
                                         'max_episode_rewards', 'mean_episode_rewards', 'min_episode_rewards',
                                         'max_episode_len', 'mean_episode_len', 'min_episode_len'])
        self.writer = None
        self.timesteps = 0
        self.num_episodes = 0

    def _make_dirs(self):
        root_path = self.root + '/' + self.name
        if not os.path.isdir(root_path):
            os.makedirs(root_path)
        checkpoint_path = root_path + '/checkpoint'
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.root_path = root_path
        self.checkpoint_path = checkpoint_path + '/checkpoint'
        self.log_path = self.root_path + '/' + self.name + '.csv'

    def save_model(self, model):
        torch.save({
            'model_state_dict': model.state_dict(),
        }, self.checkpoint_path)

    def save_args(self):
        with open(self.root_path + '/input_args.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

    def load_checkpoint(self, model):
        checkpoint = torch.load(self.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def feed(self, rew_batch, done_batch):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if done_batch[i][j]:
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1
        self.timesteps += (self.n_envs * steps)

    def write_summary(self, summary):
        for key, value in summary.items():
            self.writer.add_scalar(key, value, self.timesteps)

    def _initialise_writer(self):
        self.writer = SummaryWriter(self.log_path)

    def dump(self):
        wall_time = time.time() - self.start_time
        if self.num_episodes > 0:
            episode_statistics = self._get_episode_statistics()
            episode_statistics_list = list(episode_statistics.values())
            for key, value in episode_statistics.items():
                self.writer.add_scalar(key, value, self.timesteps)
        else:
            episode_statistics_list = [None] * 6
        log = [self.timesteps] + [wall_time] + [self.num_episodes] + episode_statistics_list
        self.log.loc[len(self.log)] = log

        with open(self.log_path, 'w') as f:
            self.log.to_csv(f, index=False)
        print(self.log.loc[len(self.log) - 1])

    def _get_episode_statistics(self):
        episode_statistics = {'Rewards/max_episodes': np.max(self.episode_reward_buffer),
                              'Rewards/mean_episodes': np.mean(self.episode_reward_buffer),
                              'Rewards/min_episodes': np.min(self.episode_reward_buffer),
                              'Len/max_episodes': np.max(self.episode_len_buffer),
                              'Len/mean_episodes': np.mean(self.episode_len_buffer),
                              'Len/min_episodes': np.min(self.episode_len_buffer)}
        return episode_statistics


class ParamLoader:
    def __init__(self, args):
        with open('hyperparams/config.yml', 'r') as f:
            params_dict = yaml.safe_load(f)[args.param_set]
        self._generate_loader(params_dict)

    def _generate_loader(self, params_dict):
        for key, val in params_dict.items():
            setattr(self, key, val)


def load_args(root_path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(root_path + '/input_args.txt', 'r') as f:
        args.__dict__ = json.load(f)

    return args
