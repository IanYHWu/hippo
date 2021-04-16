
class DemoScheduler:

    def __init__(self, args, params):
        self.num_timesteps = args.num_timesteps
        self.num_demos = params.num_demos
        self.demo_schedule = params.demo_schedule
        self.demo_learn_ratio = params.demo_learn_ratio
        self.hot_start = params.hot_start

        self.query_count = 0
        self.demo_learn_count = 0
        if self.hot_start:
            self.buffer_empty = False
        else:
            self.buffer_empty = True

    def query_demonstrator(self, curr_timestep):
        if self.demo_schedule == 'linear':
            return self._linear_schedule(curr_timestep)
        else:
            raise NotImplementedError

    def learn_from_demos(self, curr_timestep):
        learn_every = 1 / self.demo_learn_ratio
        if self.buffer_empty:
            return False
        else:
            if curr_timestep > ((self.demo_learn_count + 1) * learn_every):
                return True
            else:
                return False

    def _linear_schedule(self, curr_timestep):
        demo_every = self.num_timesteps // self.num_demos
        if curr_timestep > ((self.query_count + 1) * demo_every):
            self.buffer_empty = False
            return True
        else:
            return False

    def get_stats(self):
        return self.query_count, self.demo_learn_count






