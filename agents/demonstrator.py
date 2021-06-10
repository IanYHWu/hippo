"""Module for the demonstrator class"""

import torch


class Oracle:
    """Synthetic Demonstrator class - loads a policy to use as the synthetic demonstrator

    Attributes:
        device: cpu/gpu
        path: path to the synthetic demonstrator checkpoint
        oracle: the synthetic demonstrator object
    """

    def __init__(self, path, model, device):
        self.device = device
        self.path = path
        self.oracle = None

        self.load_oracle(model)

    def predict(self, obs, hidden_state, done):
        """Predict the next step using the synthetic demonstrator"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.oracle(obs, hidden_state, mask)
            act = dist.sample().reshape(-1)

        return act.cpu().numpy(), hidden_state.cpu().numpy()

    def load_oracle(self, model):
        """Load the synthetic demonstrator from checkpoint"""
        checkpoint = torch.load(self.path, map_location='cpu')
        #checkpoint = torch.load(self.path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.oracle = model
