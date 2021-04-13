

class BaseAgent(object):
    """
    Class for the basic agent objects.
    """

    def __init__(self,
                 env,
                 actor_critic,
                 logger,
                 storage,
                 device):
        """
        env: (gym.Env) environment following the openAI Gym API
        """
        self.env = env
        self.actor_critic = actor_critic
        self.logger = logger
        self.storage = storage
        self.device = device

        self.t = 0

    def predict(self, obs, hidden_state, done):
        """
        Predict the action with the given input
        """
        pass

    def update_policy(self):
        """
        Train the neural network model
        """
        pass

    def train(self, num_timesteps):
        """
        Train the agent with the collected trajectories
        """
        pass

    def evaluate(self):
        """
        Evaluate the agent
        """
        pass
