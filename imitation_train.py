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
from seed_selection import compute_seed_stats


def train(agent, actor_critic, demo_rollout, demo_storage, demonstrator, evaluator, params):

    save_path = args.policy_save_path + '/' + args.name
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    policy_save_path = save_path + '/' + 'checkpoint'

    print("Gathering {} Demonstrations".format(params.num_demos))
    # list of seeds to generate hot-start trajectories for
    if args.filter_demos:
        num_valid_demos = 0
        seed = 0
        training_seeds = []
        while num_valid_demos < params.num_demos:
            valid = gather_demo(args, seed, demonstrator, demo_rollout, demo_buffer=None, params=params,
                                demo_storage=demo_storage, store_mode=True,
                                reward_filter=params.reward_filter)
            if valid:
                num_valid_demos += 1
                training_seeds.append(seed)
            seed += 1
        print("Training seeds: {}".format(training_seeds))
    else:
        if params.pre_load_seed_sampling == 'random':
            # sample randomly from the training seeds
            seeds = np.random.randint(0, args.num_levels, params.pre_load).tolist()
        elif params.pre_load_seed_sampling == 'fixed':
            # sample seeds from 0 to pre_load
            if params.pre_load > args.num_levels:
                print("Warning: evaluation seeds used for pre-loading")
                print("Consider reducing the number of pre-load trajectories")
            seeds = [i for i in range(0, params.pre_load)]
        else:
            raise NotImplementedError
        for seed in seeds:
            # gather demo trajectories by seed and store them
            gather_demo(args, seed, demonstrator, demo_rollout, demo_buffer=None, params=params,
                        demo_storage=demo_storage,
                        store_mode=True,
                        reward_filter=params.reward_filter)

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
    demo_policy = load_model(params, env, device)
    demonstrator = SyntheticDemonstrator(args.demonstrator_path, demo_policy, device)

    if args.filter_demos:
        print("Computing seed statistics...")
        score_threshold, length_threshold = compute_seed_stats(args, params,
                                                               demonstrator=demonstrator)
        params.demo_max_steps = length_threshold
        params.reward_filter = score_threshold

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
