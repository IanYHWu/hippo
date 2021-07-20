import torch
import numpy as np
import time

from common.loaders import ParamLoader
from common.data_logging import load_args
from common.loaders import load_env, load_model, load_agent
from common.arguments import parser
from common.utils import set_global_log_levels, set_global_seeds, DemoLRScheduler
from common.data_logging import Logger
from common.rollout import Rollout
from common.rollout import DemoRollout
from common.rollout import DemoBuffer
from common.rollout import DemoStorage
from common.controller import DemoScheduler, BanditController, ValueLossScheduler
from test import Evaluator

from agents.demonstrator import SyntheticDemonstrator


def train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params, evaluator=None,
          controller=None, demo_rollout=None, demo_buffer=None, demo_storage=None, demonstrator=None):
    """
    Train the RL agent, representing the main training loop
        agent: RL agent
        actor_critic: policy network to train
        env: environment object
        rollout: environment-learning rollout
        logger: object to perform data logging
        curr_timestep: initial timestep - used for checkpointing
        num_timesteps: total number of timesteps to train
        params: ParamLoader object, to load parameters from config.yml
        evaluator: object to perform evaluation steps
        controller: object that decides when to query and learn from demos
        demo_rollout: demonstration-learning rollout. Collects transitions of a single trajectory
        demo_buffer: replay buffer for demonstration trajectories
        demo_storage: storage unit for demonstrations, matching seeds to single demo trajectories
        demonstrator: demonstrator agent
    """
    if params.algo == 'hippo':
        demo = True
        demo_lr_scheduler = DemoLRScheduler(args, params)

        # if hot start, load hot start trajectories into the buffer
        if params.store_mode and params.pre_load:
            print("Gathering {} Demonstrations".format(params.pre_load))
            # list of seeds to generate hot-start trajectories for
            demo_level_seeds = controller.get_preload_seeds()
            for seed in demo_level_seeds:
                # gather demo trajectories by seed and store them
                gather_demo(seed, demonstrator, demo_rollout, demo_buffer, params, demo_storage, store_mode=True,
                            reward_filter=params.reward_filter)
        controller.initialise()

    elif params.algo == 'ppo' or params.algo == 'kickstarting':
        demo = False
        demo_lr_scheduler = None
    else:
        raise NotImplementedError

    print("Now training...")
    # main PPO training loop
    save_every = num_timesteps // params.n_checkpoints
    checkpoint_count = 0
    obs = env.reset()
    hidden_state = np.zeros((params.n_envs, rollout.hidden_state_size))
    done = np.zeros(params.n_envs)
    start_ = time.time()

    while curr_timestep < num_timesteps:
        actor_critic.eval()
        if demonstrator is not None:
            demonstrator.demonstrator.eval()

        # environment-learning step
        if not demo or controller.learn_from_env():
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

        # demonstration-learning step
        if demo and controller.learn_from_demos(curr_timestep):
            if params.store_mode:
                demo_learn_indices = controller.get_learn_indices()
                for index in demo_learn_indices:
                    demo_obs_t, demo_hidden_state_t, demo_act_t, demo_rew_t, demo_done_t = demo_storage.get_demo_trajectory(
                        store_index=index)
                    demo_buffer.store(demo_obs_t, demo_hidden_state_t, demo_act_t, demo_rew_t, demo_done_t)
                demo_gather_seeds, demo_gather_indices = controller.get_new_seeds(replace_mode=True)
                for seed, index in zip(demo_gather_seeds, demo_gather_indices):
                    gather_demo(args, seed, demonstrator, demo_rollout, demo_buffer, params, demo_storage, store_mode=True,
                                store_index=index, reward_filter=params.reward_filter)
            else:
                # non-store model only works with schedule-type controller
                assert controller.controller_type == "simple_schedule"
                demo_gather_seeds, _ = controller.get_new_seeds(replace_mode=False)
                for seed in demo_gather_seeds:
                    gather_demo(args, seed, demonstrator, demo_rollout, demo_buffer, params, demo_storage=None,
                                store_mode=False, reward_filter=params.reward_filter)

            # perform a demo-learning step
            summary = agent.demo_optimize(demo_lr_scheduler)
            # reset the replay buffer after a learning step

        if demo:
            if args.log_demo_stats:
                stats_dict = controller.get_stats()
                logger.log_demo_stats(stats_dict)
            controller.update()
            demo_buffer.reset()

    print("Training complete, saving final checkpoint")
    logger.save_checkpoint(actor_critic, curr_timestep)
    print("Wall time: {}".format(time.time() - start_))
    env.close()


def gather_demo(args, seed, demonstrator, demo_rollout, demo_buffer, params, demo_storage=None, store_mode=False,
                store_index=None, reward_filter=False):
    """Gather demonstration trajectories by seed"""
    # if the seed is not in the demo storage, or we aren't using demo_storage, get a demo and store it
    tries = 0  # keeps track of how many times this level has been tried
    valid = False
    while not valid:
        step_count = 0
        demo_env = load_env(args, params, demo=True, demo_level_seed=seed)
        demo_obs = demo_env.reset()
        demo_hidden_state = np.zeros((1, demo_rollout.hidden_state_size))
        demo_done = np.zeros(1)
        demo_info = None
        # collect a trajectory of at most demo_max_steps steps
        # ensures we only collect good trajectories
        while demo_done[0] == 0 and step_count < params.demo_max_steps:
            demo_act, demo_next_hidden_state = demonstrator.predict(demo_obs, demo_hidden_state,
                                                                    demo_done)
            demo_next_obs, demo_rew, demo_done, demo_info = demo_env.step(demo_act)
            # demo_rollout stores a single trajectory
            demo_rollout.store(demo_obs, demo_hidden_state, demo_act, demo_rew, demo_done, demo_info)
            demo_obs = demo_next_obs
            demo_hidden_state = demo_next_hidden_state
            step_count += 1
        final_env_reward = demo_info[0]['env_reward']
        if step_count < params.demo_max_steps:
            if (not reward_filter) or (reward_filter and final_env_reward > 0):
                valid = True
        if valid:
            # if the trajectory is valid, compute returns and store it
            demo_obs_t, demo_hidden_state_t, demo_act_t, demo_rew_t, demo_done_t, demo_env_rew = demo_rollout.get_demo_trajectory()
            if store_mode:
                if store_index is None:
                    demo_storage.update_guides(seed)
                    demo_storage.store(demo_obs_t, demo_hidden_state_t, demo_act_t, demo_rew_t,
                                       demo_done_t, demo_env_rew)
                else:
                    demo_storage.update_guides(seed, store_index=store_index)
                    demo_storage.store(demo_obs_t, demo_hidden_state_t, demo_act_t, demo_rew_t,
                                       demo_done_t, store_index=store_index)
            else:
                demo_buffer.store(demo_obs_t, demo_hidden_state_t, demo_act_t, demo_rew_t,
                                  demo_done_t)
            demo_rollout.reset()  # after storing the trajectory, reset the rollout
            demo_env.close()
        else:
            # else, reset the env and rollout, and then try again
            demo_rollout.reset()
            demo_env.close()
            tries += 1
            if tries == 100:
                # if this level has yielded 10 bad trajectories, skip it
                break


def main(args):
    added_timesteps = args.add_timesteps  # used if extra epochs need to be trained
    load_checkpoint = args.load_checkpoint
    pretrained_policy_path = args.pretrained_policy_path

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
    pretrained_actor_critic = None
    actor_critic = load_model(params, env, device)
    curr_timestep = 0
    if load_checkpoint:
        print("Loading checkpoint: {}".format(args.log_dir + '/' + args.name))
        actor_critic, curr_timestep = logger.load_checkpoint(actor_critic)
        print("Current timestep = {}".format(curr_timestep))
    if pretrained_policy_path:
        print("Loading pre-trained policy: {}".format(pretrained_policy_path))
        pretrained_actor_critic = logger.load_policy(actor_critic)
        if params.algo != "kickstarting":
            print("Training from pre-trained policy")
            actor_critic = pretrained_actor_critic
        else:
            print("Training using kickstarting")

    observation_shape = env.observation_space.shape

    print("Initialising rollout...")
    rollout = Rollout(observation_shape, params.hidden_size, params.n_steps, params.n_envs, device)

    if args.evaluate:
        print("Initialising evaluator...")
        evaluator = Evaluator(args, params, device)
    else:
        evaluator = None

    if params.algo == 'hippo':
        algo = 'hippo'
        print("Using Agent - HIPPO")
    elif params.algo == 'ppo':
        algo = 'ppo'
        print("Using Agent - Vanilla PPO")
    elif params.algo == 'kickstarting':
        algo = 'kickstarting'
    else:
        raise NotImplementedError

    if algo == 'hippo':
        print("Initialising demonstration rollout and buffer...")
        demo_rollout = DemoRollout(observation_shape, params.hidden_size, params.demo_max_steps, device)
        demo_buffer = DemoBuffer(observation_shape, params.hidden_size, params.buffer_max_samples,
                                 params.demo_max_steps, device)
        if params.store_mode:
            print("Initialising demonstration storage")
            demo_storage = DemoStorage(observation_shape, params.hidden_size, params.demo_store_max_samples,
                                       params.demo_max_steps, device)
        else:
            demo_storage = None

        print("Initialising controller...")
        if params.demo_controller == 'linear_schedule':
            print("Using a linear schedule as the controller")
            controller = DemoScheduler(args, params, rollout, schedule='linear', demo_storage=demo_storage)
        elif params.demo_controller == 'bandit':
            print('Using Bandit Controller')
            controller = BanditController(args, params, rollout, demo_storage, demo_buffer, actor_critic)
        elif params.demo_controller == 'value_loss_schedule':
            print('Using Value Loss Schedule')
            controller = ValueLossScheduler(args, params, rollout, demo_storage, demo_buffer, actor_critic)
        else:
            raise NotImplementedError
        print("Initialising demonstrator...")
        demo_model = load_model(params, env, device)
        demonstrator = SyntheticDemonstrator(args.demonstrator_path, demo_model, device)
        print("Initialising agent...")
        agent = load_agent(env, actor_critic, rollout, device, params=params, demo_buffer=demo_buffer)
        train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params, evaluator,
              controller, demo_rollout, demo_buffer, demo_storage, demonstrator)
    elif algo == 'kickstarting':
        print("Initialising agent...")
        agent = load_agent(env, actor_critic, rollout, device, params, pretrained_policy=pretrained_actor_critic, num_timesteps=args.num_timesteps)
        train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params,
              evaluator)
    else:
        print("Initialising agent...")
        agent = load_agent(env, actor_critic, rollout, device, params)
        train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params, evaluator)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
