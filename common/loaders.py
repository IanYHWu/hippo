from procgen import ProcgenEnv
from envs.procgen_wrappers import *
from common.model import *
from common.actor_critic import CategoricalAC
from agents.ppo import PPO, get_args_ppo


def load_env(args, params):
    env = ProcgenEnv(num_envs=params.n_envs,
                     env_name=args.env_name,
                     start_level=args.start_level,
                     num_levels=args.num_levels,
                     distribution_mode=args.distribution_mode)
    normalize_rew = params.normalise_reward
    env = VecExtractDictObs(env, "rgb")
    if normalize_rew:
        env = VecNormalize(env, ob=False)  # normalizing returns, but not the img frames.
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    return env


def load_model(params, env, device):
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = params.architecture
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Model architecture
    if architecture == 'small':
        embedder_model = SmallModel(in_channels=in_channels)
    else:
        embedder_model = ImpalaModel(in_channels=in_channels)

    # Discrete action space
    recurrent = params.recurrent
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        actor_critic = CategoricalAC(embedder_model, recurrent, action_size)
    else:
        raise NotImplementedError
    actor_critic.to(device)

    return embedder_model, actor_critic


def load_agent(env, actor_critic, logger, storage, device, params):

    if params.algo == "ppo":
        params_dict = get_args_ppo(params)
        agent = PPO(env, actor_critic, logger, storage, device, **params_dict)
    else:
        raise NotImplementedError

    return agent


