from agents.ppo import PPO, get_args_ppo
from common.utils import adjust_lr
import torch
import torch.optim as optim
import numpy as np
import time


class PPODemo(PPO):
    def __init__(self,
                 env,
                 actor_critic,
                 logger,
                 storage,
                 device,
                 n_checkpoints=10,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalise_adv=True,
                 normalise_reward=True,
                 use_gae=True,
                 dem_coef=0.2):

        super().__init__(env, actor_critic, logger, storage, device, n_checkpoints, n_steps,
                         epoch, mini_batch_per_epoch, mini_batch_size, gamma, lmbda, learning_rate, grad_clip_norm,
                         eps_clip, value_coef, entropy_coef, normalise_adv, normalise_reward, use_gae)
        self.dem_coef = dem_coef

    def demo_optimize(self):
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

                # Let model to handle the large batch-size with small gpu-memory
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

    def train(self, num_timesteps):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_count = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        start_ = time.time()
        print("Now training...")

        while self.t < num_timesteps:
            # Run actor_critic
            self.actor_critic.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalise_adv)

            # Optimize actor_critic & values
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            self.logger.feed(rew_batch, done_batch)
            self.logger.write_summary(summary)
            self.logger.dump()
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            # Save the model
            if self.t > ((checkpoint_count + 1) * save_every):
                print("Saving checkpoint: t = {}".format(self.t))
                self.logger.save_model(self.actor_critic)
                checkpoint_count += 1
        print("Training complete")
        print("Wall time: {}".format(start_ - time.time()))
        self.env.close()


def get_args_ppo(params):
    param_dict = {'n_checkpoints': params.n_checkpoints,
                  'n_steps': params.n_steps,
                  'n_envs': params.n_envs,
                  'epoch': params.epoch,
                  'mini_batch_per_epoch': params.mini_batch_per_epoch,
                  'mini_batch_size': params.mini_batch_size,
                  'gamma': params.gamma,
                  'lmbda': params.lmbda,
                  'learning_rate': params.learning_rate,
                  'grad_clip_norm': params.grad_clip_norm,
                  'eps_clip': params.eps_clip,
                  'value_coef': params.value_coef,
                  'entropy_coef': params.entropy_coef,
                  'normalise_adv': params.normalise_adv,
                  'normalise_reward': params.normalise_reward,
                  'use_gae': params.use_gae}

    return param_dict
