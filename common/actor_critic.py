import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CategoricalAC(nn.Module):
    def __init__(self, base, recurrent):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation - probably a ConvNet
        action_size: number of the categorical actions
        """
        super().__init__()
        self.base = base
        self.recurrent = recurrent

    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hx, masks):
        value, actor_features, rnn_hxs = self.base(x, hx, masks)
        log_probs = F.log_softmax(actor_features, dim=1)
        p = Categorical(logits=log_probs)
        value = value.reshape(-1)

        return p, value, rnn_hxs