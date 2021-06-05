import torch
import numpy as np
import time
import random

from common.loaders import ParamLoader
from common.data_logging import load_args
from common.loaders import load_env, load_model, load_agent
from common.arguments import parser
from common.utils import set_global_log_levels, set_global_seeds, extract_seeds, DemoLRScheduler
from common.data_logging import Logger
from common.storage import Storage
from common.storage import DemoStorage, MultiDemoStorage
from common.storage import DemoReplayBuffer
from common.controller import DemoScheduler, GAEController
from test import Evaluator

from agents.demonstrator import Oracle


def train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params, evaluator=None,
          controller=None, demo_rollout=None, demo_buffer=None, demonstrator=None):
    """
    Main training loop
    """
    save_every = num_timesteps // params.n_checkpoints
    checkpoint_count = 0
    obs = env.reset()
    hidden_state = np.zeros((params.n_envs, rollout.hidden_state_size))
    done = np.zeros(params.n_envs)
    start_ = time.time()

    if params.algo == 'hippo':
        demo = True
        demo_lr_scheduler = DemoLRScheduler(args, params)
        if params.demo_multi:
            multi_demo = True
            print("Using Multiple Demonstrations")
        else:
            multi_demo = False
            print("Using Single Demonstrations")

        # if hot start, load hot start trajectories into the buffer
        if not multi_demo and params.hot_start:
            print("Hot Start - {} Demonstrations".format(params.hot_start))
            for i in range(0, params.hot_start):
                valid = False  # keeps track of whether the current trajectory is valid
                while not valid:
                    demo_level_seed = random.randint(0, int(2147483647))
                    step_count = 0
                    demo_env = load_env(args, params, demo=True, demo_level_seed=demo_level_seed)
                    demo_obs = demo_env.reset()
                    demo_hidden_state = np.zeros((1, rollout.hidden_state_size))
                    demo_done = np.zeros(1)
                    while demo_done[0] == 0:
                        demo_act, demo_next_hidden_state = demonstrator.predict(demo_obs, demo_hidden_state, demo_done)
                        demo_next_obs, demo_rew, demo_done, demo_info = demo_env.step(demo_act)
                        demo_rollout.store(demo_obs, demo_hidden_state, demo_act, demo_rew)
                        demo_obs = demo_next_obs
                        demo_hidden_state = demo_next_hidden_state
                        step_count += 1
                    if step_count < params.demo_max_steps:
                        # valid trajectories defined by whether they are shorter than demo_max_steps
                        demo_buffer.store(demo_rollout)
                        demo_env.close()
                        valid = True
                    else:
                        demo_rollout.reset()
                        demo_env.close()
    else:
        demo = False
        multi_demo = False
        demo_lr_scheduler = None

    print("Now training...")
    # main PPO training loop
    while curr_timestep < num_timesteps:
        actor_critic.eval()
        if demonstrator is not None:
            demonstrator.oracle.eval()
        for step in range(params.n_steps):
            act, log_prob_act, value, next_hidden_state = agent.predict(obs, hidden_state, done)
            next_obs, rew, done, info = env.step(act)
            rollout.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            hidden_state = next_hidden_state
        _, _, last_val, hidden_state = agent.predict(obs, hidden_state, done)
        rollout.store_last(obs, hidden_state, last_val)
        rollout.compute_estimates(params.gamma, params.lmbda, params.use_gae, params.normalise_adv)
        summary = agent.optimize()

        # learning from single demo trajectories - loop to gather trajectories
        if demo and not multi_demo:
            if controller.query_demonstrator(curr_timestep):
                # query the oracle for a single demonstration
                demo_level_seeds = controller.get_seeds()
                for demo_level_seed in demo_level_seeds:
                    tries = 0  # keeps track of how many times this level has been tried
                    valid = False
                    while not valid:
                        step_count = 0
                        demo_env = load_env(args, params, demo=True, demo_level_seed=demo_level_seed)
                        demo_obs = demo_env.reset()
                        demo_hidden_state = np.zeros((1, rollout.hidden_state_size))
                        demo_done = np.zeros(1)
                        while demo_done[0] == 0:
                            demo_act, demo_next_hidden_state = demonstrator.predict(demo_obs, demo_hidden_state, demo_done)
                            demo_next_obs, demo_rew, demo_done, demo_info = demo_env.step(demo_act)
                            # demo_rollout stores a single trajectory
                            demo_rollout.store(demo_obs, demo_hidden_state, demo_act, demo_rew)
                            demo_obs = demo_next_obs
                            demo_hidden_state = demo_next_hidden_state
                            step_count += 1
                        if step_count < params.demo_max_steps:
                            # if the trajectory is valid, compute returns and store it
                            demo_buffer.store(demo_rollout)  # store the trajectory in demo_buffer and reset demo_rollout
                            demo_env.close()
                            valid = True
                        else:
                            # else, reset the env and rollout, and then try again
                            demo_rollout.reset()
                            demo_env.close()
                            tries += 1
                            if tries == 10:
                                # if this level has yielded 10 bad trajectories, skip it
                                break

            demo_queries, demo_learning_count, demo_score = controller.get_stats()
            print("Demonstration Statistics: {} queries, {} demo learning steps, {} demo score".
                  format(demo_queries, demo_learning_count, demo_score))
            if args.log_demo_stats:
                logger.log_demo_stats(demo_queries, demo_learning_count, demo_score)

        # learning from single demo trajectories - optimise from the demonstrations
        if demo and not multi_demo:
            if controller.learn_from_demos(curr_timestep, always_learn=False):
                summary = agent.demo_optimize(demo_lr_scheduler)

        # learning from multiple demo trajectories - gather the trajectories and optimise
        if demo and multi_demo:
            controller.store_seeds()
            if controller.learn_from_demos(curr_timestep, always_learn=False):
                demo_level_seeds = controller.get_seeds()  # extract the current seeds of all n_envs environments
                for seed in demo_level_seeds:
                    demo_env = load_env(args, params, demo=True, demo_level_seed=seed)
                    demo_obs = demo_env.reset()
                    demo_hidden_state = np.zeros((1, rollout.hidden_state_size))
                    demo_done = np.zeros(1)
                    for step in range(params.demo_multi_steps):
                        demo_act, demo_next_hidden_state = demonstrator.predict(demo_obs, demo_hidden_state, demo_done)
                        demo_next_obs, demo_rew, demo_done, demo_info = demo_env.step(demo_act)
                        demo_rollout.store(demo_obs, demo_hidden_state, demo_act, demo_rew, demo_done)
                        demo_obs = demo_next_obs
                        demo_hidden_state = demo_next_hidden_state
                    demo_rollout.increment_env_counter()
                    demo_env.close()
                demo_buffer.store(demo_rollout)
                summary = agent.demo_optimize(demo_lr_scheduler)

        curr_timestep += params.n_steps * params.n_envs
        rew_batch, done_batch = rollout.fetch_log_data()  # fetch rewards and done data from the rollout
        if args.evaluate:
            # perform testing and log training and testing results
            eval_rewards, eval_len = evaluator.evaluate(actor_critic)
            logger.feed(rew_batch, done_batch, eval_rewards, eval_len)
        else:
            # log training results
            logger.feed(rew_batch, done_batch)
        logger.log_results()

        # Save the model
        if curr_timestep > ((checkpoint_count + 1) * save_every):
            print("Saving checkpoint: t = {}".format(curr_timestep))
            logger.save_checkpoint(actor_critic, curr_timestep)
            checkpoint_count += 1

    print("Training complete, saving final checkpoint")
    logger.save_checkpoint(actor_critic, curr_timestep)
    print("Wall time: {}".format(time.time() - start_))
    env.close()


def main(args):
    """
    Main function to execute
    """
    added_timesteps = args.add_timesteps  # used if extra epochs need to be trained
    load_checkpoint = args.load_checkpoint

    if load_checkpoint:
        args = load_args(args.log_dir + '/' + args.name)

    params = ParamLoader(args)

    set_global_seeds(args.seed)
    set_global_log_levels(args.log_level)  # controls how many results we store in our logging buffer
    print("Seed: {}".format(args.seed))

    num_timesteps = args.num_timesteps + added_timesteps
    print("Training for {} timesteps".format(num_timesteps))
    args.num_timesteps = num_timesteps

    print("Initialising logger...")
    logger = Logger(args, params, log_wandb=args.wandb)
    logger.save_args()

    if args.device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device: {}".format(device))
    torch.set_num_threads(4)

    print("Initialising environment...")
    env = load_env(args, params)

    print("Initialising model...")
    actor_critic = load_model(params, env, device)
    curr_timestep = 0
    if load_checkpoint:
        print("Loading checkpoint: {}".format(args.log_dir + '/' + args.name))
        actor_critic, curr_timestep = logger.load_checkpoint(actor_critic)
        print("Current timestep = {}".format(curr_timestep))

    observation_shape = env.observation_space.shape

    print("Initialising storage...")
    rollout = Storage(observation_shape, params.hidden_size, params.n_steps, params.n_envs, device)

    if args.evaluate:
        print("Initialising evaluator...")
        evaluator = Evaluator(args, params, device)
    else:
        evaluator = None

    if params.algo == 'hippo':
        algo = 'hippo'
        print("Using Agent - PPO Demo, HIPPO")
    elif params.algo == 'ppo':
        algo = 'ppo'
        print("Using Agent - Vanilla PPO")
    else:
        raise NotImplementedError

    if algo == 'hippo':
        print("Initialising demonstration storage and buffer...")
        if params.demo_multi:
            demo_rollout = MultiDemoStorage(observation_shape, params.hidden_size, params.demo_multi_steps, params.n_envs,
                                            device)
            demo_buffer = DemoReplayBuffer(observation_shape, params.hidden_size, device,
                                           max_samples=None,
                                           sampling_strategy=params.demo_sampling_strategy,
                                           mode=algo)
        else:
            demo_rollout = DemoStorage(device)
            demo_buffer = DemoReplayBuffer(observation_shape, params.hidden_size, device,
                                           max_samples=params.buffer_max_samples,
                                           sampling_strategy=params.demo_sampling_strategy,
                                           mode=algo)

        print("Initialising controller...")
        if params.demo_controller == 'linear_schedule':
            print("Using a linear schedule as the controller")
            controller = DemoScheduler(args, params, rollout, schedule='linear')
        elif params.demo_controller == 'gae':
            print('Using Average GAE Controller')
            controller = GAEController(args, params, rollout)
        else:
            raise NotImplementedError
        print("Initialising demonstrator...")
        demo_model = load_model(params, env, device)
        demonstrator = Oracle(args.oracle_path, demo_model, device)
        print("Initialising agent...")
        agent = load_agent(env, actor_critic, rollout, device, params=params, demo_buffer=demo_buffer)
        train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params, evaluator,
              controller, demo_rollout, demo_buffer, demonstrator)
    else:
        print("Initialising agent...")
        agent = load_agent(env, actor_critic, rollout, device, params)
        train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params, evaluator)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
