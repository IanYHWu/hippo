import procgen
import torch
import numpy as np
import matplotlib.pyplot as plt
from common.loaders import load_env, load_seeded_env


def compute_seed_stats(args, params, demonstrator):
    """Gather demonstration trajectories by seed"""
    # if the seed is not in the demo storage, or we aren't using demo_storage, get a demo and store it
    n = 10
    trajectory_lengths = np.zeros(n)
    trajectory_rewards = np.zeros(n)
    for i in range(n):
        step_count = 0
        total_reward = 0
        demo_env = load_env(args, params, demo=True, demo_level_seed=i)
        demo_obs = demo_env.reset()
        demo_hidden_state = np.zeros((1, params.hidden_size))
        demo_done = np.zeros(1)
        # collect a trajectory of at most demo_max_steps steps
        # ensures we only collect good trajectories
        while demo_done[0] == 0:
            demo_act, demo_next_hidden_state = demonstrator.predict(demo_obs, demo_hidden_state,
                                                                    demo_done)
            demo_next_obs, demo_rew, demo_done, demo_info = demo_env.step(demo_act)
            # demo_rollout stores a single trajectory
            demo_obs = demo_next_obs
            demo_hidden_state = demo_next_hidden_state
            step_count += 1
            total_reward += demo_info[0]['env_reward']

        trajectory_lengths[i] = step_count
        trajectory_rewards[i] = total_reward

        demo_env.close()

    if args.plot_seed_stats:
        plot_seed_stats(args, trajectory_lengths, trajectory_rewards)

    score_threshold = np.percentile(trajectory_rewards, 50)
    accepted_lengths = []
    for i, j in zip(trajectory_rewards, trajectory_lengths):
        if i > score_threshold:
            accepted_lengths.append(j)
    length_threshold = int(np.percentile(accepted_lengths, 90))

    print("Score Threshold: {}".format(score_threshold))
    print("Length Threshold: {}".format(length_threshold))

    return score_threshold, length_threshold


def plot_seed_stats(args, traj_lengths, traj_rewards):
    plt.scatter(traj_lengths, traj_rewards)
    plt.xlabel("Trajectory Length")
    plt.ylabel("Trajectory Reward")
    plt.savefig(args.log_dir + '/' + args.name + '/' + "seed_stats_plot")

