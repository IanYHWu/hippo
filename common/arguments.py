"""
Arguments for Argparse
"""

import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument(
    '--name', type=str, required=True, help='Experiment name'
)
parser.add_argument(
    '--load_checkpoint', type=bool, default=False, help='Continue training from a checkpoint'
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