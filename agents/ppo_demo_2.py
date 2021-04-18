from agents.base_agent import BaseAgent
import torch
import torch.optim as optim
import numpy as np


class PPODemo(BaseAgent):

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
                 mini_batch_size=32 * 8,
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

    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.actor_critic(obs, hidden_state, mask)
            act = dist.sample().reshape(-1)
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_count = 1

        self.actor_critic.train()
        for e in range(self.epoch):
            recurrent = self.actor_critic.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1 - done_batch)
                dist_batch, value_batch, _ = self.actor_critic(obs_batch, hidden_state_batch, mask_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # actor_critic Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                if grad_accumulation_count % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_count += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}

        return summary

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

                pol_loss = -log_prob_act_batch * torch.max(torch.zeros(returns_batch.shape).to(self.device),
                                                           (returns_batch - value_batch))
                pol_loss = pol_loss.mean()

                val_loss = 0.5 * (torch.max(torch.zeros(returns_batch.shape).to(self.device),
                                            (returns_batch - value_batch))).pow(2)
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
                  'demo_epochs': params.demo_epochs,
                  'demo_batch_size': params.demo_batch_size}

    return param_dict
