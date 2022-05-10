"""
This module implements different controllers, which are objects that decide (1) when to query a demonstration (2) when
to learn from the demonstration buffer and (3) what the demonstration seeds should be
"""

class BaseController:
    """Base class for the controller"""

    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.pre_load = params.pre_load
        self.pre_load_seed_sampling = params.pre_load_seed_sampling
        self.num_levels = args.num_levels

    def initialise(self):
        """Initialise scores used by the controller"""
        pass

    def learn_from_env(self):
        """Learn from the environment"""
        pass

    def learn_from_demos(self, curr_timestep):
        """Learn from demonstrations"""
        pass

    def get_new_seeds(self):
        """Get seeds to sample new demos of"""
        pass

    def get_learn_indices(self):
        """Get the demo storage indices of demos to learn from"""
        pass

    def update(self):
        """Update the controller"""
        pass

    def get_stats(self):
        """Return demo learning statistics"""
        pass
