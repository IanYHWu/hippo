"""
Arguments for Argparse
"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--name', type=str, required=True, help='Experiment name'
)
parser.add_argument(
    '--load_checkpoint', action='store_true', help='Continue training from a checkpoint'
)
parser.add_argument(
    '--pretrained_policy_path', type=str, default=None, help='Path to pre-trained policy'
)
parser.add_argument(
    '--log_dir', type=str, default='..', help='Logging directory'
)
parser.add_argument(
    '--env_name', type=str, default='starpilot', help='Environment name'
)
parser.add_argument(
    '--start_level', type=int, default=int(0), help='Start-level for environment'
)
parser.add_argument(
    '--num_levels', type=int, default=int(0), help='Number of training levels for environment'
)
parser.add_argument(
    '--distribution_mode', type=str, default='easy', help='Distribution mode for environment'
)
parser.add_argument(
    '--param_set', type=str, default='easy', help='Parameter set (config.yml)'
)
parser.add_argument(
    '--device', type=str, default='gpu', required=False, help='Device to use'
)
parser.add_argument(
    '--num_timesteps', type=int, default=int(25000000), help='Number of training timesteps'
)
parser.add_argument(
    '--seed', type=int, default=int(28), help='Random generator seed'
)
parser.add_argument(
    '--log_level', type=int, default=int(40), help='Number of levels per log - {10,20,30,40}'
)
parser.add_argument(
    '--num_checkpoints', type=int, default=int(1), help='Number of checkpoints to store'
)
parser.add_argument(
    '--add_timesteps', type=int, default=int(0), help='Additional timesteps to train - used for checkpoint training'
)

# evaluation args
parser.add_argument(
    '--evaluate', action='store_true', help='Perform evaluation after every rollout'
)
parser.add_argument(
    '--num_test_episodes', type=int, default=10, help='Number of test episodes to perform per evaluation run'
)
parser.add_argument(
    '--evaluate_policy_demo_kl', action='store_true', help='Log KL stats between policy and demonstrator'
)

# wandb args
parser.add_argument(
    '--wandb', action='store_true', help='Log results on wandb'
)
parser.add_argument(
    '--wandb_project_name', type=str, default='msc_2021', help='Project name for wandb'
)
parser.add_argument(
    '--wandb_name', type=str, default='test_run', help='Run name for wandb'
)

# ppo_demo args
parser.add_argument(
    '--demonstrator_path', type=str, default='', help='Path to synthetic demonstrator'
)
parser.add_argument(
    '--log_demo_stats', action='store_true', help='Log demonstration statistics'
)


