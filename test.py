import torch
import random
import numpy as np

from common.loaders import load_env


class Evaluator:
    def __init__(self, args, params, device):
        self.args = args
        self.params = params
        self.device = device

    def evaluate(self, actor_critic):
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
        while eps < self.args.num_test_episodes:
            action, next_hidden_state = self._predict(actor_critic, obs, hidden_state, done)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            hidden_state = next_hidden_state
            episode_rewards.append(info[0]['env_reward'])

            if done[0] == 1:
                episode_rewards_store.append(np.sum(episode_rewards))
                episode_len_store.append(len(episode_rewards))
                episode_rewards = []
                eps += 1

        return np.mean(episode_rewards_store), np.mean(episode_len_store)

    def _predict(self, actor_critic, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = actor_critic(obs, hidden_state, mask)
            act = dist.sample().reshape(-1)

        return act.cpu().numpy(), hidden_state.cpu().numpy()





