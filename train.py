import torch
import os
from common.data_logging import ParamLoader
from common.data_logging import load_args
from common.loaders import load_env, load_model, load_agent
from common.arguments import parser
from common.utils import set_global_log_levels, set_global_seeds
from common.data_logging import Logger
from common.storage import Storage


def train(args, params):

    if args.load_checkpoint:
        args = load_args(args.log_dir)

    set_global_seeds(args.seed)
    set_global_log_levels(args.log_level)
    if args.device == 'gpu':
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
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
    embedding_model, actor_critic = load_model(params, env, device)
    if args.load_checkpoint:
        print("Loading checkpoint: {}".format(args.log_dir))
        actor_critic = logger.load_checkpoint(actor_critic)

    observation_shape = env.observation_space.shape
    hidden_state_size = embedding_model.output_dim

    print("Initialising storage...")
    storage = Storage(observation_shape, hidden_state_size, params.n_steps, params.n_envs, device)

    print("Initialising agent...")
    agent = load_agent(env, actor_critic, logger, storage, device, params)
    agent.train(args.num_timesteps)


if __name__ == "__main__":

    args = parser.parse_args()

    params = ParamLoader(args)

    train(args, params)


