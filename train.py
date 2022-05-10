import torch
import numpy as np
import time

from common.loaders import ParamLoader
from common.data_logging import load_args
from common.loaders import load_env, load_model, load_agent
from common.arguments import parser
from common.utils import set_global_log_levels, set_global_seeds
from common.data_logging import Logger
from common.rollout import Rollout
from test import Evaluator


def train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params,
          evaluator=None):
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

        # environment-learning step
        for step in range(params.n_steps):
            act, log_prob_act, value, next_hidden_state = agent.predict(obs, hidden_state, done)
            next_obs, rew, done, info = env.step(act)
            rollout.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            hidden_state = next_hidden_state
        _, _, last_val, hidden_state = agent.predict(obs, hidden_state, done)
        rollout.store_last(obs, hidden_state, last_val)
        rollout.compute_estimates(params.gamma, params.lmbda, params.use_gae,
                                  params.normalise_adv)
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

    print("Training complete, saving final checkpoint")
    logger.save_checkpoint(actor_critic, curr_timestep)
    print("Wall time: {}".format(time.time() - start_))
    env.close()


def main(args):
    added_timesteps = args.add_timesteps  # used if extra epochs need to be trained
    load_checkpoint = args.load_checkpoint

    if load_checkpoint:
        args = load_args(args.log_dir + '/' + args.name)

    params = ParamLoader(args)

    set_global_seeds(args.seed)
    set_global_log_levels(
        args.log_level)  # controls how many results we store in our logging buffer
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

    print("Initialising rollout...")
    rollout = Rollout(observation_shape, params.hidden_size, params.n_steps, params.n_envs, device)

    if args.evaluate:
        print("Initialising evaluator...")
        evaluator = Evaluator(args, params, device)
    else:
        evaluator = None

    print("Initialising agent...")
    agent = load_agent(env, actor_critic, rollout, device, params)
    train(agent, actor_critic, env, rollout, logger, curr_timestep, num_timesteps, params,
          evaluator)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
