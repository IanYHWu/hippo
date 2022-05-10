"""Module for loading functions - used to load a variety of different objects"""

from procgen import ProcgenEnv
from envs.procgen_wrappers import *
from common.model import *
from common.actor_critic import CategoricalAC
from agents.ppo import PPO, get_args
import yaml


def load_env(args, params, evaluate=False, eval_seed=None):
    """Load the Procgen environment
        args: argparse object
        params: ParamLoader object
        eval: flag for evaluation mode
        demo: flag for demonstration mode
        demo_level_seed: seed for demonstration mode
        eval_seed: seed for evaluation mode
    """
    if not evaluate:
        env = ProcgenEnv(num_envs=params.n_envs,
                         env_name=args.env_name,
                         start_level=args.start_level,
                         num_levels=args.num_levels,
                         distribution_mode=args.distribution_mode)
    else:
        # for evaluation mode
        env = ProcgenEnv(num_envs=1,
                         env_name=args.env_name,
                         start_level=eval_seed,
                         num_levels=0,
                         distribution_mode=args.distribution_mode)

    normalize_rew = params.normalise_reward
    env = VecExtractDictObs(env, "rgb")
    if normalize_rew:
        env = VecNormalize(env, ob=False)  # normalizing returns, but not the image frames
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    return env


def load_model(params, env, device):
    """Load an actor critic policy"""
    observation_shape = env.observation_space.shape
    action_size = env.action_space.n
    architecture = params.architecture
    recurrent = params.recurrent

    # Model architecture
    if len(observation_shape) == 3:
        if architecture == 'Small':
            print('Using SmallNet Base')
            base = SmallNetBase(observation_shape[0], input_h=observation_shape[1],
                                input_w=observation_shape[2], recurrent=recurrent,
                                hidden_size=params.hidden_size)
        else:
            print('Using ResNet Base')
            base = ResNetBase(observation_shape[0], input_h=observation_shape[1],
                              input_w=observation_shape[2], recurrent=recurrent,
                              hidden_size=params.hidden_size)
    elif len(observation_shape) == 1:
        print('Using MLP Base')
        base = MLPBase(observation_shape, recurrent=recurrent, hidden_size=params.hidden_size)
    else:
        raise NotImplementedError

    # Discrete action space
    actor_critic = CategoricalAC(base, recurrent, action_size)
    actor_critic.to(device)

    return actor_critic


def load_agent(env, actor_critic, storage, device, params):
    """Load an RL Agent"""
    if params.algo == "ppo" or params.algo == "bc":
        params_dict = get_args(params)
        agent = PPO(env, actor_critic, storage, device, **params_dict)
    else:
        raise NotImplementedError

    return agent


class ParamLoader:
    """ParamLoader object - collects all the parameters specified in config.yml into a class that
       can be conveniently accessed. The attributes of this class comprise of all the default values
       for all possible parameters"""

    def __init__(self, args):
        """Set default values"""
        # basic ppo parameters
        self.n_envs = 2
        self.n_steps = 16
        self.n_checkpoints = 2
        self.epoch = 3
        self.mini_batch_per_epoch = 8
        self.mini_batch_size = 2048
        self.gamma = 0.999
        self.lmbda = 0.95
        self.learning_rate = 0.0005
        self.grad_clip_norm = 0.5
        self.eps_clip = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.normalise_adv = True
        self.normalise_reward = True
        self.use_gae = True
        self.algo = 'hippo'
        self.architecture = 'ResNet'
        self.recurrent = False
        self.hidden_size = 256

        # read in yaml config file and overwrite the appropriate defaults
        with open('hyperparams/config.yml', 'r') as f:
            params_dict = yaml.safe_load(f)[args.param_set]
        self._generate_loader(params_dict)
        self.wandb_id = None

    def _generate_loader(self, params_dict):
        """Create the loader object, overwriting defaults when needed"""
        for key, val in params_dict.items():
            setattr(self, key, val)
