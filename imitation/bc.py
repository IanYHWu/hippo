import torch
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler
from torch import optim
import numpy as np


class BC:

    def __init__(self, actor_critic, demo_storage, epochs, mini_batch_size, learning_rate,
                 l2_coef, entropy_coef, device):
        self.actor_critic = actor_critic
        self.demo_storage = demo_storage
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.l2_coef = l2_coef
        self.entropy_coef = entropy_coef
        self.device = device
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps=1e-5)
        self.batch_size = demo_storage.get_n_valid_transitions()

    def batch_generator(self):
        mask = 1 - self.demo_storage.done_store
        weights = mask.squeeze(-1).reshape(-1)
        sampler = BatchSampler(WeightedRandomSampler(weights, int(self.batch_size), replacement=False),
                               self.mini_batch_size, drop_last=True)
        for indices in sampler:
            obs_batch = torch.FloatTensor(self.demo_storage.obs_store.float()).reshape(-1,
                                                                                       *self.demo_storage.obs_shape)[
                indices].to(self.device)
            hidden_state_batch = torch.FloatTensor(self.demo_storage.hidden_states_store[:-1]).reshape(-1,
                                                                                          self.demo_storage.hidden_state_size).to(
                self.device)
            act_batch = torch.FloatTensor(self.demo_storage.act_store.float()).reshape(-1)[
                indices].to(
                self.device)
            done_batch = torch.FloatTensor(self.demo_storage.done_store).reshape(-1)[indices].to(self.device)

            yield obs_batch, hidden_state_batch, act_batch, done_batch

    def train(self):
        self.actor_critic.train()
        for e in range(self.epochs):
            loss_list, true_act_prob_list, entropy_loss_list, l2_loss_list = [], [], [], []
            generator = self.batch_generator()
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch = sample
                mask_batch = 1 - done_batch
                dist_batch, value_batch, _ = self.actor_critic(obs_batch, hidden_state_batch, mask_batch)
                entropy_batch = dist_batch.entropy()
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                true_act_prob = torch.exp(log_prob_act_batch).mean()
                l2_norm = [torch.sum(torch.square(w)) for w in self.actor_critic.parameters()]
                l2_norm = sum(l2_norm) / 2

                entropy_loss = -self.entropy_coef * entropy_batch.mean()
                neg_log_prob_act = -log_prob_act_batch.mean()
                l2_loss = self.l2_coef * l2_norm
                loss = neg_log_prob_act + entropy_loss + l2_loss

                loss_list.append(loss.item())
                true_act_prob_list.append(true_act_prob.item())
                entropy_loss_list.append(entropy_loss.item())
                l2_loss_list.append(l2_loss.item())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            summary = {"epoch": e + 1,
                       "loss": np.mean(loss_list),
                       "true_act_prob": np.mean(true_act_prob_list),
                       "entropy_loss": np.mean(entropy_loss_list),
                       "l2_loss": np.mean(l2_loss_list)}
            print("-----------------------")
            print(summary)
            print("-----------------------")

