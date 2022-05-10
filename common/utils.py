"""Module for utility functions and objects"""

import numpy as np
import gym
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def set_global_log_levels(level):
    """Set the logging level"""
    gym.logger.set_level(level)


def set_global_seeds(seed):
    """Set the global seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init(module, weight_init, bias_init, gain=1):
    """Initialise network parameters"""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    """Orthogonal initialisation for network parameters, useful for recurrent networks"""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    """Xavier initialisation for network parameters"""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    """Adjust the learning rate - simple LR schedule"""
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_n_params(model):
    """Get the number of parameters"""
    return str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3)) + ' M params'


def extract_seeds(info):
    """Extract the seeds given an info dict"""
    seeds_list = [d['level_seed'] for d in info]
    return seeds_list


def visualise(arr):
    arr = np.transpose(arr, (1, 2, 0))
    plt.imshow(arr)
    plt.show()


def animate(arr):
    arr = arr.numpy()
    fig = plt.figure()
    i = 0
    im = plt.imshow(arr[0].transpose((1, 2, 0)), animated=True)

    def update(i):
        if i < len(arr):
            i += 1
        else:
            i = 0
        im.set_array(arr[i].transpose((1, 2, 0)))
        return im,

    a = animation.FuncAnimation(fig, update, blit=True)
    plt.show()











