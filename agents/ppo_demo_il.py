from agents.ppo import PPO
import torch.optim as optim
import numpy as np
import torch


class PPODemoIL(PPO):

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
                 demo_batch_size=4096,
                 demo_mini_batch_size=512,
                 demo_value_coef=0.05,
                 demo_loss_coef=1,
                 demo_epochs=1,
                 demo_sampling_strategy='uniform'):

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
        self.demo_learning_rate = demo_learning_rate
        self.demo_value_coef = demo_value_coef
        self.demo_loss_coef = demo_loss_coef
        self.demo_mini_batch_size = demo_mini_batch_size
        self.demo_epochs = demo_epochs
        self.demo_batch_size = demo_batch_size
        self.demo_sampling_strategy = demo_sampling_strategy

    def demo_optimize(self, lr_schedule):
        val_loss_list, pol_loss_list = [], []

        n_valid_transitions = self.demo_buffer.get_n_valid_transitions()
        batch_size = self.demo_batch_size
        # the batch size must be <= than the number of non-padding transitions in the trajectory
        if n_valid_transitions <= batch_size:
            batch_size = n_valid_transitions
        mini_batch_size = self.demo_mini_batch_size
        # the mini-batch size must be <= the batch size
        if batch_size <= mini_batch_size:
            mini_batch_size = batch_size

        lr = lr_schedule.get_lr()
        if lr is None:
            lr = self.demo_learning_rate
        demo_optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)

        self.demo_buffer.compute_pi_v(self.actor_critic)
        self.actor_critic.train()
        for e in range(self.demo_epochs):
            recurrent = self.actor_critic.is_recurrent()
            generator = self.demo_buffer.demo_generator(batch_size=batch_size,
                                                        mini_batch_size=mini_batch_size,
                                                        recurrent=recurrent,
                                                        sample_method=self.demo_sampling_strategy,
                                                        mode='il')
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, returns_batch, mask_batch = sample

                dist_batch, value_batch, _ = self.actor_critic(obs_batch, hidden_state_batch, mask_batch)
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                pol_loss = -log_prob_act_batch * torch.clamp(returns_batch - value_batch, min=0)
                pol_loss = pol_loss.mean()

                val_loss = 0.5 * (torch.clamp(returns_batch - value_batch, min=0)).pow(2)
                val_loss = val_loss.mean()

                loss = self.demo_loss_coef * (pol_loss + self.demo_value_coef * val_loss)
                loss.backward()

                demo_optimizer.step()
                demo_optimizer.zero_grad()
                val_loss_list.append(val_loss.item())
                pol_loss_list.append(pol_loss.item())

        summary = {'loss/pol': np.mean(pol_loss_list),
                   'loss/val': np.mean(val_loss_list)}

        return summary


def get_args_demo_il(params):
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
                  'demo_value_coef': params.demo_value_coef,
                  'demo_epochs': params.demo_epochs,
                  'demo_batch_size': params.demo_batch_size,
                  'demo_loss_coef': params.demo_loss_coef,
                  'demo_sampling_strategy': params.demo_sampling_strategy}

    return param_dict
