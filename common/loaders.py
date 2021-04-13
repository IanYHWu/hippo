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
        env = VecNormalize(env, ob=False)  # normalizing returns, but not the image frames
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    return env


def load_model(params, env, device):
    observation_shape = env.observation_space.shape
    architecture = params.architecture
    recurrent = params.recurrent

    # Model architecture
    if len(observation_shape) == 3:
        if architecture == 'Small':
            base = SmallNetBase(observation_shape[0], input_h=observation_shape[1],
                                input_w=observation_shape[2], recurrent=recurrent, hidden_size=params.hidden_size)
        else:
            base = ResNetBase(observation_shape[0], input_h=observation_shape[1],
                              input_w=observation_shape[2], recurrent=recurrent, hidden_size=params.hidden_size)
    elif len(observation_shape) == 1:
        base = MLPBase(observation_shape, recurrent=recurrent, hidden_size=params.hidden_size)
    else:
        raise NotImplementedError

    # Discrete action space
    actor_critic = CategoricalAC(base, recurrent)
    actor_critic.to(device)

    return actor_critic


def load_agent(env, actor_critic, logger, storage, device, params):

    if params.algo == "ppo":
        params_dict = get_args_ppo(params)
        agent = PPO(env, actor_critic, logger, storage, device, **params_dict)
    else:
        raise NotImplementedError

    return agent


