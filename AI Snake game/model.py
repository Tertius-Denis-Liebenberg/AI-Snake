import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class ActorCritic(nn.Module):
    def __init__(self, grid_size=40, channels=4):  # 4 channels
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out = self._get_conv_out((channels, grid_size, grid_size))
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(256, 3)
        self.critic_head = nn.Linear(256, 1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_features = self.conv(x)
        fc_features = self.fc(conv_features)
        logits = self.actor_head(fc_features)
        value = self.critic_head(fc_features)
        return logits, value

    def save(self, file_name='model.pth'):
        path = os.path.join('model', file_name)
        os.makedirs('model', exist_ok=True)
        checkpoint = {'actor_critic': self.state_dict()}
        torch.save(checkpoint, path)

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.returns = None
        self.advantages = None

    def size(self):
        return len(self.states)

    def add(self, state, action, logprob, value, reward, done):
        self.states.append(state.copy())  # np array copy
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, gamma, gae_lambda, next_value):
        size = self.size()
        self.values = np.asarray(self.values)
        self.rewards = np.asarray(self.rewards)
        self.dones = np.asarray(self.dones, dtype=float)

        advantages = np.zeros(size)
        gae = 0.0
        value_t = next_value

        for t in reversed(range(size)):
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * value_t * next_nonterminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[t] = gae
            value_t = self.values[t]

        self.returns = advantages + self.values
        self.advantages = advantages

class PPOTrainer:
    def __init__(self, ac_net, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=10,
                 gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5, batch_size=64, max_grad_norm=0.5):
        self.ac_net = ac_net
        self.optimizer = optim.Adam(ac_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

    def update(self, buffer, next_value):
        buffer.compute_gae(self.gamma, self.gae_lambda, next_value)

        # Normalize advantages
        adv_mean = np.mean(buffer.advantages)
        adv_std = np.std(buffer.advantages) + 1e-8
        buffer.advantages = (buffer.advantages - adv_mean) / adv_std

        old_logprobs = np.array(buffer.logprobs)
        n_steps = buffer.size()

        for _ in range(self.epochs):
            indices = np.random.permutation(n_steps)
            for start in range(0, n_steps, self.batch_size):
                end = start + self.batch_size
                mb_inds = indices[start:end]

                # Batch data
                mb_states = np.stack([buffer.states[i] for i in mb_inds])
                mb_states_t = torch.tensor(mb_states, dtype=torch.float)

                mb_actions = torch.tensor([buffer.actions[i] for i in mb_inds], dtype=torch.int64)
                mb_old_logprobs = torch.tensor(old_logprobs[mb_inds], dtype=torch.float)
                mb_advantages = torch.tensor(buffer.advantages[mb_inds], dtype=torch.float)
                mb_returns = torch.tensor(buffer.returns[mb_inds], dtype=torch.float)

                # Forward pass
                logits, newvals = self.ac_net(mb_states_t)
                dist = torch.distributions.Categorical(logits=logits)
                newlogprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Ratios
                ratios = torch.exp(newlogprobs - mb_old_logprobs)

                # Clipped surrogate
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                actor_loss = (-torch.min(surr1, surr2)).mean() - self.ent_coef * entropy

                # Value loss
                critic_loss = F.mse_loss(newvals.view(-1), mb_returns.view(-1))

                # Total loss
                loss = actor_loss + self.vf_coef * critic_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        buffer.clear()