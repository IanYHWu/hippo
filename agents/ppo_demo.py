from agents.ppo import PPO
import torch.optim as optim
import numpy as np
import torch


class PPODemo(PPO):

    def __init__(self,
                 env,
                 actor_critic,
                 storage,
                 demo_buffer,
                 device,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 learning_rate=5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.05,
                 entropy_coef=0.01,
                 demo_learning_rate=5e-4,
                 demo_batch_size=512,
                 demo_mini_batch_size=664,
                 demo_coef=0.5,
                 demo_epochs=3):

        super().__init__(env, actor_critic, storage, device)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.demo_buffer = demo_buffer
        self.demo_optimizer = optim.Adam(self.actor_critic.parameters(), lr=demo_learning_rate, eps=1e-5)
        self.demo_learning_rate = demo_learning_rate
        self.demo_coef = demo_coef
        self.demo_mini_batch_size = demo_mini_batch_size
        self.demo_epochs = demo_epochs
        self.demo_batch_size = demo_batch_size

    def demo_optimize(self):
        val_loss_list, pol_loss_list = [], []
        buffer_size = self.demo_buffer.get_buffer_capacity()
        batch_size = self.demo_batch_size
        mini_batch_size = self.demo_mini_batch_size
        if buffer_size < self.demo_batch_size:
            batch_size = buffer_size
        if self.demo_mini_batch_size > self.demo_batch_size:
            mini_batch_size = batch_size

        self.actor_critic.train()
        for e in range(self.demo_epochs):
            recurrent = self.actor_critic.is_recurrent()
            generator = self.demo_buffer.fetch_demo_generator(batch_size=batch_size,
                                                              mini_batch_size=mini_batch_size,
                                                              recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, mask_batch, returns_batch = sample
                dist_batch, value_batch, _ = self.actor_critic(obs_batch, hidden_state_batch, mask_batch)
                log_prob_act_batch = dist_batch.log_prob(act_batch)

                pol_loss = -log_prob_act_batch * torch.max(torch.zeros(returns_batch.shape),
                                                           (returns_batch - value_batch))
                pol_loss = pol_loss.mean()

                val_loss = 0.5 * (torch.max(torch.zeros(returns_batch.shape),
                                            (returns_batch - value_batch)))**2
                val_loss = val_loss.mean()

                loss = pol_loss + self.demo_coef * val_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.grad_clip_norm)
                self.demo_optimizer.step()
                self.demo_optimizer.zero_grad()
                val_loss_list.append(val_loss.item())
                pol_loss_list.append(pol_loss.item())

        summary = {'loss/pol': np.mean(pol_loss_list),
                   'loss/val': np.mean(val_loss_list)}

        return summary


def get_args_ppo_demo(params):
    param_dict = {'n_steps': params.n_steps,
                  'n_envs': params.n_envs,
                  'epoch': params.epoch,
                  'mini_batch_per_epoch': params.mini_batch_per_epoch,
                  'mini_batch_size': params.mini_batch_size,
                  'learning_rate': params.learning_rate,
                  'grad_clip_norm': params.grad_clip_norm,
                  'eps_clip': params.eps_clip,
                  'value_coef': params.value_coef,
                  'entropy_coef': params.entropy_coef,
                  'demo_learning_rate': params.demo_learning_rate,
                  'demo_mini_batch_size': params.demo_mini_batch_size,
                  'demo_coef': params.demo_coef,
                  'demo_epochs': params.demo_epochs}

    return param_dict
