import numpy as np
from train import gather_demo
import torch
import os
from imitation.loaders import ParamLoader
from common.utils import set_global_log_levels, set_global_seeds
from common.loaders import load_model, load_env
from common.rollout import DemoStorage, DemoRollout
from agents.demonstrator import SyntheticDemonstrator
from imitation.bc import BC
from imitation.arguments import parser
from test import Evaluator


def train(agent, actor_critic, demo_rollout, demo_storage, demonstrator, evaluator, params):

    save_path = args.policy_save_path + '/' + args.name
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    policy_save_path = save_path + '/' + 'checkpoint'

    print("Gathering {} Demonstrations".format(params.num_demos))
    # list of seeds to generate hot-start trajectories for
    if params.demo_seed_sampling == 'random':
        # sample randomly from the training seeds
        seeds = np.random.randint(0, args.num_levels, params.num_demos)
    elif params.demo_seed_sampling == 'fixed':
        # sample seeds from 0 to pre_load
        if params.num_demos > args.num_levels:
            print("Warning: evaluation seeds used for pre-loading")
            print("Consider reducing the number of pre-load trajectories")
        seeds = [i for i in range(0, params.num_demos)]
    else:
        raise NotImplementedError

    for seed in seeds:
        # gather demo trajectories by seed and store them
        gather_demo(args, seed, demonstrator, demo_rollout=demo_rollout, params=params, demo_storage=demo_storage, demo_buffer=None,
                    store_mode=True, reward_filter=params.reward_filter)

    agent.train()
    if args.evaluate:
        print("Evaluating pre-trained model...")
        eval_rewards, eval_len = evaluator.evaluate(actor_critic)
        print("Mean Test Rewards: {}".format(eval_rewards))
        print("Mean Test Length: {}".format(eval_len))

    torch.save({'model_state_dict': actor_critic.state_dict()}, policy_save_path)


def main(args):
    params = ParamLoader(args)
    set_global_seeds(args.seed)

    if args.device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device: {}".format(device))
    torch.set_num_threads(4)

    env = load_env(args, params, demo=True)
    actor_critic = load_model(params, env, device)
    observation_shape = env.observation_space.shape

    demo_storage = DemoStorage(observation_shape, params.hidden_size, params.num_demos,
                                       params.demo_max_steps, device)
    demo_rollout = DemoRollout(observation_shape, params.hidden_size, params.demo_max_steps, device)

    demonstrator = SyntheticDemonstrator(args.demonstrator_path, actor_critic, device)

    if args.evaluate:
        print("Initialising evaluator...")
        evaluator = Evaluator(args, params, device)
    else:
        evaluator = None

    if params.algo == "bc":
        print("Training Behavioural Cloning model...")
        agent = BC(actor_critic, demo_storage, params.epochs, params.mini_batch_size, params.learning_rate,
                 params.l2_coef, params.entropy_coef, device)
    else:
        raise NotImplementedError

    train(agent, actor_critic, demo_rollout, demo_storage, demonstrator, evaluator, params)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)