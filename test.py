import torch
import random
import numpy as np

from common.loaders import load_env


class Evaluator:
    def __init__(self, args, params, device):
        self.args = args
        self.params = params
        self.device = device

        self.rel_demo_kl = 0
        self.rel_demo_prob = 0
        self.env_eval_kl = 0
        self.env_eval_prob = 0

    def evaluate(self, actor_critic, demonstrator=None):
        actor_critic.eval()
        eval_seed = random.randint(0, int(2147483647))
        env = load_env(self.args, self.params, eval=True, eval_seed=eval_seed)
        obs = env.reset()
        hidden_state = np.zeros((1, self.params.hidden_size))
        done = np.zeros(1)

        eps = 0
        episode_rewards_store = []
        episode_len_store = []
        episode_rewards = []

        env_eval_kl_list = []
        env_eval_prob_list = []

        while eps < self.args.num_test_episodes:
            action, next_hidden_state, dist = self._predict(actor_critic, obs, hidden_state, done)

            if demonstrator is not None:
                demo_act, _, demo_dist = self._predict(demonstrator.demonstrator, obs, hidden_state, done)
                env_eval_kl = torch.distributions.kl_divergence(dist, demo_dist).mean()
                env_eval_kl_list.append(env_eval_kl.item())
                act_prob = demo_dist.probs.reshape(-1)[action]
                env_eval_prob_list.append(act_prob)

            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            hidden_state = next_hidden_state
            episode_rewards.append(info[0]['env_reward'])

            if done[0] == 1:
                episode_rewards_store.append(np.sum(episode_rewards))
                episode_len_store.append(len(episode_rewards))
                episode_rewards = []
                eps += 1

        if demonstrator is not None:
            self.env_eval_kl = np.mean(env_eval_kl_list)
            self.env_eval_prob = np.mean(env_eval_prob_list)

        return np.mean(episode_rewards_store), np.mean(episode_len_store)

    def evaluate_demo_probs(self, actor_critic, demo_storage, demonstrator):
        act_kl_list = []
        act_prob_list = []
        with torch.no_grad():
            for i in range(demo_storage.get_n_samples()):
                demo_length = torch.nonzero(demo_storage.done_store[i])[0].item()
                mask_store = 1 - demo_storage.done_store[i, :demo_length]
                policy_dist, _, _ = actor_critic(demo_storage.obs_store[i, :demo_length].float().to(self.device),
                                                        demo_storage.hidden_states_store[i, :demo_length].float().to(
                                                            self.device),
                                                        mask_store.float().to(self.device))
                demo_dist, _, _ = demonstrator.demonstrator(demo_storage.obs_store[i, :demo_length].float().to(self.device),
                                                demo_storage.hidden_states_store[i, :demo_length].float().to(
                                                    self.device),
                                                mask_store.float().to(self.device))
                act_kl = torch.distributions.kl_divergence(policy_dist, demo_dist).mean()
                act_kl_list.append(act_kl.item())
                actions = demo_storage.act_store[i, :demo_length].long()
                act_prob = torch.gather(policy_dist.probs, 1, actions.unsqueeze(1)).squeeze(-1).mean()
                act_prob_list.append(act_prob.item())

        self.rel_demo_kl = np.mean(act_kl_list)
        self.rel_demo_prob = np.mean(act_prob_list)

    def _predict(self, actor_critic, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = actor_critic(obs, hidden_state, mask)
            act = dist.sample().reshape(-1)

        return act.cpu().numpy(), hidden_state.cpu().numpy(), dist





