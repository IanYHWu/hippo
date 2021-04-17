from procgen import ProcgenEnv
from envs.procgen_wrappers import *
from common.model import *
from common.actor_critic import CategoricalAC
from agents.ppo import PPO, get_args_ppo
from agents.ppo_demo import PPODemo, get_args_ppo_demo


def load_env(args, params, demo=False, demo_level_seed=None):
    if not demo:
        env = ProcgenEnv(num_envs=params.n_envs,
                         env_name=args.env_name,
                         start_level=args.start_level,
                         num_levels=args.num_levels,
                         distribution_mode=args.distribution_mode)
    else:
        demo_level_seed = np.array([demo_level_seed], dtype='int32')
        assert demo_level_seed is not None
        env = ProcgenEnv(num_envs=1,
                         env_name=args.env_name,
                         start_level=demo_level_seed,
                         num_levels=1,
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


def load_agent(env, actor_critic, storage, device, params, demo_buffer=None):
    if params.algo == "ppo":
        params_dict = get_args_ppo(params)
        agent = PPO(env, actor_critic, storage, device, **params_dict)
    elif params.algo == "ppo_demo":
        params_dict = get_args_ppo_demo(params)
        agent = PPODemo(env, actor_critic, storage, demo_buffer, device, **params_dict)
    else:
        raise NotImplementedError

    return agent



