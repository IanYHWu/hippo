import os
from functools import partial

import gym
import torch
from gym.spaces.box import Box
from procgen import ProcgenEnv
from envs.procgen_wrappers import VecEnvWrapper, VecExtractDictObs, VecNormalize
import numpy as np


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImageProcgen(TransposeObs):
    def __init__(self, env=None, op=[0, 3, 2, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImageProcgen, self).__init__(env)
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[2], obs_shape[1], obs_shape[0]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        if ob.shape[0] == 1:
            ob = ob[0]
        return ob.transpose(self.op[0], self.op[1], self.op[2], self.op[3])


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, seeds, device):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device
        self.level_sampler = LevelSampler(seeds)
        self.seeds = seeds

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype)

    @property
    def raw_venv(self):
        rvenv = self.venv
        while hasattr(rvenv, 'venv'):
            rvenv = rvenv.venv
        return rvenv

    def reset(self):
        if self.seeds is not None:
            for e in range(self.venv.num_envs):
                seed = self.level_sampler.sample()
                self.venv.seed(seed, e)

        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        # obs = torch.from_numpy(obs).float().to(self.device) / 255.

        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        # actions = actions.numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        # print(f"stepping {info[0]['level_seed']}, done: {done}")

        # reset environment here
        if self.seeds is not None:
            for e in done.nonzero()[0]:
                seed = self.level_sampler.sample()
                self.venv.seed(seed, e)  # seed resets the corresponding level

            # NB: This reset call propagates upwards through all VecEnvWrappers
            obs = self.raw_venv.observe()['rgb']
            # Note reset does not reset game instances, but only returns latest observations
        #
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        # obs = torch.from_numpy(obs).float().to(self.device) / 255.
        # reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, reward, done, info


class LevelSampler:

    def __init__(self, seeds):
        self.seeds = seeds

    def sample(self):
        return np.random.choice(self.seeds)

