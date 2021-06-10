"""Module for the base agent class"""

class BaseAgent(object):
    """
    Class for the basic agent objects.
    """

    def __init__(self,
                 env,
                 actor_critic,
                 storage,
                 device):
        """
        env: (gym.Env) environment following the openAI Gym API
        """
        self.env = env
        self.actor_critic = actor_critic
        self.storage = storage
        self.device = device

        self.t = 0

    def predict(self, obs, hidden_state, done):
        """
        Predict the action with the given input
        """
        pass

    def optimize(self):
        """
        Train the neural network model
        """
        pass

