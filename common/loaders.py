"""Module for loading functions - used to load a variety of different objects"""

from procgen import ProcgenEnv
from envs.procgen_wrappers import *
from common.model import *
from common.actor_critic import CategoricalAC
from agents.ppo import PPO, get_args
from agents.hippo import HIPPO, get_args_hippo
from agents.sil import SIL, get_args_sil
from imitation.kickstarting import Kickstarter, get_args_kickstarting
import yaml

from envs.seeded_procgen import VecPyTorchProcgen


def load_env(args, params, eval=False, demo=False, demo_level_seed=None, eval_seed=None):
    """Load the Procgen environment
        args: argparse object
        params: ParamLoader object
        eval: flag for evaluation mode
        demo: flag for demonstration mode
        demo_level_seed: seed for demonstration mode
        eval_seed: seed for evaluation mode
    """
    if not demo and not eval:
        # regular environment-learning mode
        env = ProcgenEnv(num_envs=params.n_envs,
                         env_name=args.env_name,
                         start_level=args.start_level,
                         num_levels=args.num_levels,
                         distribution_mode=args.distribution_mode)
    elif demo and not eval:
        # demonstration-collecting mode
        if demo_level_seed is None:
            demo_level_seed = 0
        demo_level_seed = np.array([demo_level_seed], dtype='int32')  # it seems to bug out if we don't convert to numpy
        env = ProcgenEnv(num_envs=1,
                         env_name=args.env_name,
                         start_level=demo_level_seed,
                         num_levels=1,
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


def load_seeded_env(args, params, seeds, device):

    venv = ProcgenEnv(num_envs=params.n_envs, env_name=args.env_name, \
        num_levels=args.num_levels, start_level=args.start_level, \
        distribution_mode=args.distribution_mode)
    normalize_rew = params.normalise_reward
    venv = VecExtractDictObs(venv, "rgb")
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)
    # venv = TransposeFrame(venv)
    # venv = ScaledFloatFrame(venv)

    envs = VecPyTorchProcgen(venv, seeds, device)

    return envs


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
                                input_w=observation_shape[2], recurrent=recurrent, hidden_size=params.hidden_size)
        else:
            print('Using ResNet Base')
            base = ResNetBase(observation_shape[0], input_h=observation_shape[1],
                              input_w=observation_shape[2], recurrent=recurrent, hidden_size=params.hidden_size)
    elif len(observation_shape) == 1:
        print('Using MLP Base')
        base = MLPBase(observation_shape, recurrent=recurrent, hidden_size=params.hidden_size)
    else:
        raise NotImplementedError

    # Discrete action space
    actor_critic = CategoricalAC(base, recurrent, action_size)
    actor_critic.to(device)

    return actor_critic


def load_agent(env, actor_critic, storage, device, params, demo_buffer=None, num_timesteps=None, pretrained_policy=None):
    """Load an RL Agent"""
    if params.algo == "ppo":
        params_dict = get_args(params)
        agent = PPO(env, actor_critic, storage, device, **params_dict)
    elif params.algo == 'hippo':
        params_dict = get_args_hippo(params)
        agent = HIPPO(env, actor_critic, storage, demo_buffer, device, **params_dict)
    elif params.algo == 'sil':
        params_dict = get_args_sil(params)
        agent = SIL(env, actor_critic, storage, demo_buffer, device, **params_dict)
    elif params.algo == 'kickstarting':
        params_dict = get_args_kickstarting(params)
        agent = Kickstarter(env, actor_critic, storage, device, pretrained_policy, num_timesteps, **params_dict)
    else:
        raise NotImplementedError

    return agent


class ParamLoader:
    """ParamLoader object - collects all the parameters specified in config.yml into a class that can be
    conveniently accessed. The attributes of this class comprise of all the default values for all possible
    parameters"""

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
        # basic hippo parameters
        self.demo_coef = 0.5
        self.demo_learning_rate = 0.0005
        self.demo_batch_size = 64
        self.demo_mini_batch_size = 32
        self.demo_epochs = 10
        self.demo_controller = 'linear_schedule'  # controller type
        self.demo_sampling_strategy = 'uniform'
        self.demo_entropy_coef = 0.01
        self.demo_value_coef = 0.005
        self.demo_max_steps = 999  # max acceptable length of demo trajectories
        self.demo_normalise_adv = False  # normalise demo advantages
        self.demo_lr_schedule = False  # learning rate scheduler for demo learning steps
        self.buffer_max_samples = 100  # max capacity of demo buffer
        self.reward_filter = None
        # schedule controller
        self.pre_load = 0  # pre-loaded demonstrations
        self.pre_load_seed_sampling = 'random'  # determines how we sample seeds for hot start trajectories
        self.demo_learn_ratio = 0.1  # ratio of demo-learning steps to env steps
        self.demo_levels = 200  # number of seeds we permit demonstrations of
        self.demo_sampling = 'random'
        self.demo_transition_sampling = 'uniform'  # sampling strategy for demo buffer transitions
        self.num_learn_demos = 64  # number of demo trajectories sampled per demo learning step
        self.demo_limit = None  # no. of timesteps beyond which no demo learning is permitted
        self.store_mode = True  # store demos and re-use
        self.demo_store_max_samples = 200  # maximum sample capacity of demo storage. Used with store_mode
        self.replace = 0  # no. of demos in the storage to replace per demo learn step. Used with store_mode
        # Bandit controller
        self.scoring_method = 'rank'
        self.demo_sampling_replace = False  # during demo sampling, sample with replacement
        self.alpha = 1  # EMA coefficient for tracking env value losses. 1 => account for only the latest reading
        self.temperature = 0.5
        self.rho = 0.3  # staleness coefficient
        self.mu = 0.5  # demo score scaling - downweights the demo feedback
        self.eta = 0  # weighting of environment val losses relative to demo val losses when computing val loss score
        self.nu = 1  # priortised sampling weighting

        # read in yaml config file and overwrite the appropriate defaults
        with open('hyperparams/config.yml', 'r') as f:
            params_dict = yaml.safe_load(f)[args.param_set]
        self._generate_loader(params_dict)
        self.wandb_id = None

    def _generate_loader(self, params_dict):
        """Create the loader object, overwriting defaults when needed"""
        for key, val in params_dict.items():
            setattr(self, key, val)
