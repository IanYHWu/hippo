import numpy as np
import gym
import torch.nn as nn
import torch
import math


def set_global_log_levels(level):
    gym.logger.set_level(level)


def set_global_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_n_params(model):
    return str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3)) + ' M params'


def extract_seeds(info):
    seeds_list = [d['level_seed'] for d in info]
    return seeds_list


class DemoLRScheduler:

    def __init__(self, args, params):
        self.schedule_type = params.demo_lr_schedule
        self.demo_learn_ratio = params.demo_learn_ratio
        self.num_timesteps = args.num_timesteps
        self.params = params
        self.args = args

        self.i = 0

        if not self.schedule_type:
            self.schedule = None
        elif self.schedule_type == 'linear_inc':
            self._generate_linear_schedule(0.0002, 0.0008)
        elif self.schedule_type == 'linear_dec':
            self._generate_linear_schedule(0.0008, 0.0002)
        else:
            raise NotImplementedError

    def _generate_linear_schedule(self, start_rate, end_rate):
        learn_every = self.params.n_envs * self.params.n_steps * (1 / self.demo_learn_ratio)
        num_learn_steps = math.ceil(self.num_timesteps / learn_every)
        self.schedule = np.linspace(start_rate, end_rate, num_learn_steps)

    def get_lr(self):
        if self.schedule is None:
            return None
        else:
            lr = self.schedule[self.i]
            self.i += 1
            return lr







