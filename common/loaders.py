from procgen import ProcgenEnv
from envs.procgen_wrappers import *
from common.model import *
from common.actor_critic import CategoricalAC
from agents.ppo import PPO, get_args
from agents.ppo_demo_il import PPODemoIL, get_args_demo_il
from agents.ppo_demo_hippo import PPODemoHIPPO, get_args_demo_hippo


def load_env(args, params, eval=False, demo=False, multi_demo=False, demo_level_seed=None, eval_seed=None):
    if not demo and not eval:
        env = ProcgenEnv(num_envs=params.n_envs,
                         env_name=args.env_name,
                         start_level=args.start_level,
                         num_levels=args.num_levels,
                         distribution_mode=args.distribution_mode)
    elif demo and not multi_demo and not eval:
        demo_level_seed = np.array([demo_level_seed], dtype='int32')
        env = ProcgenEnv(num_envs=1,
                         env_name=args.env_name,
                         start_level=demo_level_seed,
                         num_levels=1,
                         distribution_mode=args.distribution_mode)
    elif demo and multi_demo and not eval:
        demo_level_seed = np.array(demo_level_seed, dtype='int32')
        env = ProcgenEnv(num_envs=params.n_envs,
                         env_name=args.env_name,
                         start_level=demo_level_seed,
                         num_levels=1,
                         distribution_mode=args.distribution_mode)
    else:
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
        params_dict = get_args(params)
        agent = PPO(env, actor_critic, storage, device, **params_dict)
    elif params.algo == "ppo_demo_il":
        params_dict = get_args_demo_il(params)
        agent = PPODemoIL(env, actor_critic, storage, demo_buffer, device, **params_dict)
    elif params.algo == 'ppo_demo_hippo':
        params_dict = get_args_demo_hippo(params)
        agent = PPODemoHIPPO(env, actor_critic, storage, demo_buffer, device, **params_dict)
    else:
        raise NotImplementedError

    return agent



