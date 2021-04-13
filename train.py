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
from common.utils import adjust_lr


def train(agent, actor_critic, env, rollout, logger, num_timesteps, params):
    save_every = num_timesteps // params.n_checkpoints
    checkpoint_count = 0
    obs = env.reset()
    hidden_state = np.zeros((params.n_envs, rollout.hidden_state_size))
    done = np.zeros(params.n_envs)
    start_ = time.time()
    t = 0
    print("Now training...")

    while t < num_timesteps:
        # Run actor_critic
        actor_critic.eval()
        for _ in range(params.n_steps):
            act, log_prob_act, value, next_hidden_state = agent.predict(obs, hidden_state, done)
            next_obs, rew, done, info = env.step(act)
            rollout.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            hidden_state = next_hidden_state
        _, _, last_val, hidden_state = agent.predict(obs, hidden_state, done)
        rollout.store_last(obs, hidden_state, last_val)
        # Compute advantage estimates
        rollout.compute_estimates(params.gamma, params.lmbda, params.use_gae, params.normalise_adv)

        # Optimize actor_critic & values
        summary = agent.optimize()
        t += params.n_steps * params.n_envs
        rew_batch, done_batch = rollout.fetch_log_data()
        logger.feed(rew_batch, done_batch)
        logger.write_summary(summary)
        logger.dump()
        agent.optimizer = adjust_lr(agent.optimizer, params.learning_rate, t, num_timesteps)
        # Save the model
        if t > ((checkpoint_count + 1) * save_every):
            print("Saving checkpoint: t = {}".format(t))
            logger.save_model(actor_critic)
            checkpoint_count += 1
    print("Training complete")
    logger.save_model(actor_critic)
    print("Wall time: {}".format(time.time() - start_))
    env.close()


def main(args, params):
    if args.load_checkpoint:
        args = load_args(args.log_dir)

    num_timesteps = args.num_timesteps
    set_global_seeds(args.seed)
    set_global_log_levels(args.log_level)
    print("Seed: {}".format(args.seed))
    if args.device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device: {}".format(device))
    torch.set_num_threads(4)

    print("Initialising environment...")
    env = load_env(args, params)
    print("Initialising logger...")
    logger = Logger(args, params)
    logger.save_args()
    print("Initialising model...")
    actor_critic = load_model(params, env, device)
    if args.load_checkpoint:
        print("Loading checkpoint: {}".format(args.log_dir))
        actor_critic = logger.load_checkpoint(actor_critic)

    observation_shape = env.observation_space.shape

    print("Initialising storage...")
    rollout = Storage(observation_shape, params.hidden_size, params.n_steps, params.n_envs, device)

    print("Initialising agent...")
    agent = load_agent(env, actor_critic, logger, rollout, device, params)
    train(agent, actor_critic, env, rollout, logger, num_timesteps, params)


if __name__ == "__main__":

    args = parser.parse_args()

    params = ParamLoader(args)

    main(args, params)


