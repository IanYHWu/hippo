from agents.ppo import PPO
import torch
import torch.optim as optim
import numpy as np


class Kickstarter(PPO):

    def __init__(self,
                 env,
                 actor_critic,
                 storage,
                 device,
                 pre_trained_policy,
                 num_timesteps,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 learning_rate=5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01):

        super().__init__(env, actor_critic, storage, device)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.num_timesteps = num_timesteps
        self.epoch = epoch
        self.pre_trained_policy = pre_trained_policy
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.train_steps = 0

    def get_kickstarting_coef(self):
        timesteps_per_step = (self.n_envs * self.n_steps) * self.train_steps
        self.train_steps += 1
        return 0.1 * (1 - (timesteps_per_step / self.num_timesteps))

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list, kickstarting_loss_list = [], [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_count = 1

        kickstarting_coef = self.get_kickstarting_coef()

        self.actor_critic.train()
        self.pre_trained_policy.eval()
        for e in range(self.epoch):
            recurrent = self.actor_critic.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1 - done_batch)
                dist_batch, value_batch, _ = self.actor_critic(obs_batch, hidden_state_batch,
                                                               mask_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(
                    -self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # actor_critic Entropy
                entropy_loss = dist_batch.entropy().mean()

                kickstarting_dist_batch, _, _ = self.pre_trained_policy(obs_batch,
                                                                        hidden_state_batch,
                                                                        mask_batch)
                print("kickstart dist batch: {}".format(kickstarting_dist_batch))
                print("ce: {}".format(cross_entropy(kickstarting_dist_batch, dist_batch)))
                kickstarting_loss = cross_entropy(kickstarting_dist_batch, dist_batch).mean()
                print("kickstarting loss: {}".format(kickstarting_loss))

                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss + \
                       kickstarting_coef * kickstarting_loss
                loss.backward()

                if grad_accumulation_count % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                                   self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_count += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())
                kickstarting_loss_list.append(kickstarting_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list),
                   'Loss/kickstarting': np.mean(kickstarting_loss_list)}

        print(summary)

        return summary


def get_args_kickstarting(params):
    """Extract the relevant arguments for Vanilla PPO"""
    param_dict = {'n_steps': params.n_steps,
                  'n_envs': params.n_envs,
                  'epoch': params.epoch,
                  'mini_batch_per_epoch': params.mini_batch_per_epoch,
                  'mini_batch_size': params.mini_batch_size,
                  'learning_rate': params.learning_rate,
                  'grad_clip_norm': params.grad_clip_norm,
                  'eps_clip': params.eps_clip,
                  'value_coef': params.value_coef,
                  'entropy_coef': params.entropy_coef}

    return param_dict


def cross_entropy(p, q):
    p = p.probs
    q = q.probs
    print("p: {}".format(p))
    print("q: {}".format(q))
    return -torch.sum(p * torch.log(q), dim=1)