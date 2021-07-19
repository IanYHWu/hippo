import numpy as np
import torch
import yaml


class ParamLoader:
    """ParamLoader object - collects all the parameters specified in config.yml into a class that can be
    conveniently accessed. The attributes of this class comprise of all the default values for all possible
    parameters"""

    def __init__(self, args):
        """Set default values"""
        self.num_demos = 20
        self.reward_filter = False
        self.hidden_size = 256
        self.demo_max_steps = 999
        self.algo = "bc"
        self.epochs = 5
        self.mini_batch_size = 512
        self.learning_rate = 0.0005
        self.l2_coef = 0.001
        self.entropy_coef = 0.001
        self.demo_seed_sampling = "fixed"
        self.normalise_reward = True
        self.architecture = 'ResNet'
        self.recurrent = False

        # read in yaml config file and overwrite the appropriate defaults
        with open('imitation/hyperparams/config.yml', 'r') as f:
            params_dict = yaml.safe_load(f)[args.param_set]
        self._generate_loader(params_dict)

    def _generate_loader(self, params_dict):
        """Create the loader object, overwriting defaults when needed"""
        for key, val in params_dict.items():
            setattr(self, key, val)


