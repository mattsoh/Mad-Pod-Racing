import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        mean = self.action_head(h)
        value = self.value_head(h).squeeze(-1)
        return mean, value


@dataclass
class RolloutBuffer:
    obs: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray

    @classmethod
    def create(cls, rollout_len, num_envs, obs_size, action_size):
        obs = np.zeros((rollout_len, num_envs, obs_size), dtype=np.float32)
        actions = np.zeros((rollout_len, num_envs, action_size), dtype=np.float32)
        logprobs = np.zeros((rollout_len, num_envs), dtype=np.float32)
        rewards = np.zeros((rollout_len, num_envs), dtype=np.float32)
        dones = np.zeros((rollout_len, num_envs), dtype=np.float32)
        values = np.zeros((rollout_len, num_envs), dtype=np.float32)
        returns = np.zeros((rollout_len, num_envs), dtype=np.float32)
        advantages = np.zeros((rollout_len, num_envs), dtype=np.float32)
        return cls(obs, actions, logprobs, rewards, dones, values, returns, advantages)

    def add(self, t, obs, actions, logprobs, rewards, dones, values):
        self.obs[t] = obs
        self.actions[t] = actions
        self.logprobs[t] = logprobs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values

    def compute_returns(self, last_value, gamma, gae_lambda):
        last_gae = np.zeros(self.rewards.shape[1], dtype=np.float32)
        for t in reversed(range(self.rewards.shape[0])):
            next_nonterminal = 1.0 - self.dones[t]
            next_values = last_value if t == self.rewards.shape[0] - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values


class PPOAgent:
    def __init__(self, obs_size, action_size, hidden_size, lr, device):
        self.device = device
        self.policy = PolicyNet(obs_size, action_size, hidden_size).to(device)
        self.log_std = nn.Parameter(torch.zeros(action_size, device=device))
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + [self.log_std], lr=lr
        )

    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mean, value = self.policy(obs_t)
        std = torch.exp(self.log_std)
        if deterministic:
            z = mean
        else:
            z = mean + std * torch.randn_like(mean)
        action = torch.tanh(z)
        logprob = self._logprob_from_z(z, mean, std, action)
        return action.detach().cpu().numpy(), logprob.detach().cpu().numpy(), value.detach().cpu().numpy()

    def evaluate_actions(self, obs, actions):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        mean, value = self.policy(obs_t)
        std = torch.exp(self.log_std)
        z = self._atanh(actions_t)
        logprob = self._logprob_from_z(z, mean, std, actions_t)
        entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + self.log_std).sum(-1)
        return logprob, entropy, value

    def update(self, buffer, clip, epochs, minibatch_size, vf_coef, ent_coef, max_grad_norm):
        obs = buffer.obs.reshape(-1, buffer.obs.shape[-1])
        actions = buffer.actions.reshape(-1, buffer.actions.shape[-1])
        logprobs = buffer.logprobs.reshape(-1)
        returns = buffer.returns.reshape(-1)
        advantages = buffer.advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_size = obs.shape[0]
        indices = np.arange(total_size)
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, total_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                new_logprob, entropy, value = self.evaluate_actions(obs[mb_idx], actions[mb_idx])
                ratio = torch.exp(new_logprob - torch.as_tensor(logprobs[mb_idx], device=self.device))
                adv = torch.as_tensor(advantages[mb_idx], device=self.device)

                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.mse_loss(value, torch.as_tensor(returns[mb_idx], device=self.device))
                entropy_loss = -entropy.mean()

                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

    def _logprob_from_z(self, z, mean, std, action):
        var = std * std
        log_scale = torch.log(std + 1e-8)
        logprob = -0.5 * (((z - mean) ** 2) / (var + 1e-8) + 2.0 * log_scale + math.log(2.0 * math.pi))
        logprob = logprob.sum(-1)
        correction = torch.log(1.0 - action * action + 1e-6).sum(-1)
        return logprob - correction

    @staticmethod
    def _atanh(x):
        x = torch.clamp(x, -0.999, 0.999)
        return 0.5 * torch.log((1.0 + x) / (1.0 - x))
