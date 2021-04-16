import torch
import numpy as np
import time

from common.data_logging import ParamLoader
from common.data_logging import load_args
from common.loaders import load_env, load_model, load_agent
from common.arguments import parser
from common.utils import set_global_log_levels, set_global_seeds
from common.data_logging import Logger
from common.storage import Storage
from common.storage import DemoStorage
from common.storage import DemoReplayBuffer
from common.controller import DemoScheduler

from agents.demonstrator import Oracle


def train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params,
          controller=None, demo_rollout=None, demo_buffer=None, demonstrator=None):
    save_every = num_timesteps // params.n_checkpoints
    checkpoint_count = 0
    obs = env.reset()
    hidden_state = np.zeros((params.n_envs, rollout.hidden_state_size))
    done = np.zeros(params.n_envs)
    start_ = time.time()

    if params.algo == 'ppo_demo':
        demo = True
        assert controller is not None
        assert demo_rollout is not None
        assert demonstrator is not None
        assert demo_buffer is not None
        print("Using Agent - PPO Demo")
        if params.hot_start:
            pass
    else:
        demo = False
        print("Using Agent - Vanilla PPO")

    print("Now training...")

    while curr_timestep < num_timesteps:
        actor_critic.eval()
        for step in range(params.n_steps):
            act, log_prob_act, value, next_hidden_state = agent.predict(obs, hidden_state, done)
            next_obs, rew, done, info = env.step(act)
            rollout.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            hidden_state = next_hidden_state
        _, _, last_val, hidden_state = agent.predict(obs, hidden_state, done)
        rollout.store_last(obs, hidden_state, last_val)
        # Compute advantage estimates
        rollout.compute_estimates(params.gamma, params.lmbda, params.use_gae, params.normalise_adv)
        summary = agent.optimize()

        if demo and controller.query_demonstrator(curr_timestep):
            demo_level_seed = info[0]["level_seed"]
            demo_env = load_env(args, params, demo=True, demo_level_seed=demo_level_seed)
            demo_obs = env.reset()
            demo_hidden_state = np.zeros((1, rollout.hidden_state_size))
            demo_done = np.zeros(1)
            while demo_done[0] == 0:
                demo_act, demo_next_hidden_state = demonstrator.predict(obs, hidden_state, done)
                demo_next_obs, demo_rew, demo_done, demo_info = demo_env.step(demo_act)
                demo_rollout.store(demo_obs, demo_hidden_state, demo_act, demo_rew)
                demo_obs = demo_next_obs
                demo_hidden_state = demo_next_hidden_state
            demo_rollout.compute_returns()
            demo_buffer.store(demo_rollout)

        if demo and controller.learn_from_demos(curr_timestep):
            summary = agent.demo_optimize()

        curr_timestep += params.n_steps * params.n_envs
        rew_batch, done_batch = rollout.fetch_log_data()
        logger.feed(rew_batch, done_batch)
        logger.log_results()

        # Save the model
        if curr_timestep > ((checkpoint_count + 1) * save_every):
            print("Saving checkpoint: t = {}".format(curr_timestep))
            logger.save_checkpoint(actor_critic, curr_timestep)
            checkpoint_count += 1
            if demo:
                demo_queries, demo_learning_count = controller.get_stats()
                print("Demonstration Statistics: {} queries, {} demo learning steps".format(
                    demo_queries, demo_learning_count))

    print("Training complete, saving final checkpoint")
    logger.save_checkpoint(actor_critic, curr_timestep)
    print("Wall time: {}".format(time.time() - start_))
    env.close()


def main(args):

    added_timesteps = args.add_timesteps
    load_checkpoint = args.load_checkpoint

    if load_checkpoint:
        args = load_args(args.log_dir + '/' + args.name)

    params = ParamLoader(args)

    set_global_seeds(args.seed)
    set_global_log_levels(args.log_level)
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

    if params.algo == 'ppo_demo':
        print("Initialising demonstration storage and buffer...")
        demo_rollout = DemoStorage(device)
        demo_buffer = DemoReplayBuffer(observation_shape, params.hidden_size, device)
        print("Initialising controller...")
        controller = DemoScheduler(args, params)
        print("Initialising demonstrator...")
        demo_model = load_model(params, env, device)
        demonstrator = Oracle.load_oracle(args.oracle_path, demo_model)
        print("Initialising agent...")
        agent = load_agent(env, actor_critic, rollout, device, params, demo_buffer=demo_buffer)
        train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params,
              controller, demo_rollout, demo_buffer, demonstrator)
    else:
        print("Initialising agent...")
        agent = load_agent(env, actor_critic, rollout, device, params)
        train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params)


if __name__ == "__main__":

    args = parser.parse_args()

    main(args)


