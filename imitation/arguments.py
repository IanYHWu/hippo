import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--name', type=str, required=True, help='Experiment name'
)

parser.add_argument(
    '--policy_save_path', type=str, default='..', help='Policy save path'
)

parser.add_argument(
    '--num_levels', type=int, default=200, help='Number of training seeds'
)

parser.add_argument(
    '--seed', type=int, default=28, help='Seed'
)

parser.add_argument(
    '--demonstrator_path', type=str, default='..', help='Demonstrator path'
)

parser.add_argument(
    '--param_set', type=str, default=None, help='Parameter set to use'
)

parser.add_argument(
    '--device', type=str, default='gpu', help='Device to use'
)

parser.add_argument(
    '--env_name', type=str, default='starpilot', help='Procgen environment to use'
)

parser.add_argument(
    '--distribution_mode', type=str, default='easy', help='Procgen distribution mode'
)

parser.add_argument(
    '--evaluate', action="store_true", help='Evaluate pre-trained model'
)

parser.add_argument(
    '--num_test_episodes', type=int, default=20, help='Number of test episodes to use'
)

parser.add_argument(
    '--filter_demos', action="store_true", help="Filter demonstrations by reward"
)

parser.add_argument(
    '--plot_seed_stats', action="store_true", help="Plot seed statistics"
)
