import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gymnasium as gym
from collections import deque


# ==========================================
# 1. Neural Networks
# ==========================================
class ValueNetwork(nn.Module):
    def __init__(self, stateDim, hiddenDims, activation):
        super(ValueNetwork, self).__init__()
        self.activation = activation
        self.inputLayer = nn.Linear(stateDim, hiddenDims[0])
        self.hLayers = nn.ModuleList()
        for i in range(len(hiddenDims) - 1):
            hLayer = nn.Linear(hiddenDims[i], hiddenDims[i + 1])
            self.hLayers.append(hLayer)
        self.out = nn.Linear(hiddenDims[-1], 1)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            ss = torch.tensor(
                state, dtype=torch.float32, device=next(self.parameters()).device
            )
        else:
            ss = state
        if ss.dim() == 1:
            ss = ss.unsqueeze(0)

        l = self.activation(self.inputLayer(ss))
        for hLayer in self.hLayers:
            l = self.activation(hLayer(l))
        q = self.out(l)
        return q


class PolicyNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDims, activation):
        super(PolicyNetwork, self).__init__()
        self.activation = activation
        self.actionDim = actionDim

        self.inputLayer = nn.Linear(stateDim, hiddenDims[0])
        self.hLayers = nn.ModuleList()
        for i in range(len(hiddenDims) - 1):
            self.hLayers.append(nn.Linear(hiddenDims[i], hiddenDims[i + 1]))
        self.out = nn.Linear(hiddenDims[-1], actionDim)

    def forward(self, state, action=None):
        if not isinstance(state, torch.Tensor):
            ss = torch.tensor(
                state, dtype=torch.float32, device=next(self.parameters()).device
            )
        else:
            ss = state
        if ss.dim() == 1:
            ss = ss.unsqueeze(0)

        l = self.activation(self.inputLayer(ss))
        for hLayer in self.hLayers:
            l = self.activation(hLayer(l))
        logits = self.out(l)

        distrib = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = distrib.sample()

        logPs = distrib.log_prob(action)
        entropies = distrib.entropy()

        return action, logPs, entropies, logits

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            _, _, _, logits = self.forward(state)
            if deterministic:
                # FIX: Safe action shape indexing
                action = (
                    torch.argmax(logits, dim=-1)[0].item()
                    if logits.dim() > 1
                    else torch.argmax(logits, dim=-1).item()
                )
            else:
                distrib = torch.distributions.Categorical(logits=logits)
                action = (
                    distrib.sample()[0].item()
                    if logits.dim() > 1
                    else distrib.sample().item()
                )
        return action


# ==========================================
# 2. Single-Worker Rollout Buffer
# ==========================================
class RolloutBuffer:
    def __init__(self, buffer_size, state_dim):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.logprobs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.max_size = buffer_size

    def store(self, s, a, r, v, lp, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.values[self.ptr] = v
        self.logprobs[self.ptr] = lp
        self.dones[self.ptr] = d
        self.ptr += 1

    def compute_gae(self, last_value, last_done, gamma, lamda):
        advs = np.zeros(self.max_size, dtype=np.float32)
        last_gae = 0
        for t in reversed(range(self.max_size)):
            if t == self.max_size - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            advs[t] = last_gae = delta + gamma * lamda * next_non_terminal * last_gae

        returns = advs + self.values
        return advs, returns

    def clear(self):
        self.ptr = 0


# ==========================================
# 3. PPO Agent for CartPole
# ==========================================
class PPO_CartPole:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cpu")

        # Hyperparameters perfectly tuned for CartPole
        self.gamma = 0.99
        self.lamda = 0.95
        self.clip_range = 0.2
        self.beta = 0.001  # Very low entropy for balancing tasks
        self.max_grad_norm = 0.5
        self.steps_per_epoch = 1000
        self.opt_epochs = 10
        self.batchSize = 64
        self.max_steps = 500  # CartPole max

        self.actionDim = env.action_space.n
        self.stateDim = env.observation_space.shape[0]

        self.pNetwork = PolicyNetwork(
            self.stateDim, self.actionDim, [64, 64], F.relu
        ).to(self.device)
        self.vNetwork = ValueNetwork(self.stateDim, [64, 64], F.relu).to(self.device)

        self.policyOptimizer = optim.Adam(self.pNetwork.parameters(), lr=3e-4)
        self.valueOptimizer = optim.Adam(self.vNetwork.parameters(), lr=1e-3)

        self.buffer = RolloutBuffer(self.steps_per_epoch, self.stateDim)

    def trainAgent(self):
        s, _ = self.env.reset()
        ep_reward = 0
        ep_step = 0
        total_episodes = 0

        # Tracking moving average of last 10 episodes
        recent_scores = deque(maxlen=10)

        for epoch in range(1, 101):  # Max 100 epochs (~200-300 episodes)
            self.buffer.clear()

            # Collect Rollouts
            for t in range(self.steps_per_epoch):
                with torch.no_grad():
                    action_tensor, logp_tensor, _, _ = self.pNetwork(s)
                    val_tensor = self.vNetwork(s)

                    a = action_tensor.item()
                    logp = logp_tensor.item()
                    v = val_tensor.item()

                s_next, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated

                ep_reward += r
                ep_step += 1

                # CartPole Specific: The environment gives +1 per step.
                # If it terminates before 500, the agent failed. We do not bootstrap a failure.
                # If it truncates at 500, the agent succeeded, but ran out of time. We DO bootstrap.
                is_terminal_for_gae = 1 if terminated else 0

                self.buffer.store(s, a, r, v, logp, is_terminal_for_gae)
                s = s_next

                if done:
                    recent_scores.append(ep_reward)
                    total_episodes += 1
                    s, _ = self.env.reset()
                    ep_reward = 0
                    ep_step = 0

            # ==========================================
            # FIX 1: GAE BOUNDARY
            # ==========================================
            with torch.no_grad():
                if ep_step == 0:
                    last_val = 0.0
                    last_done = 1.0
                else:
                    last_val = self.vNetwork(s).item()
                    last_done = 0.0

            advs, returns = self.buffer.compute_gae(
                last_val, last_done, self.gamma, self.lamda
            )

            # Train the Networks
            self.trainNetwork(advs, returns)

            # Print Progress
            current_avg = np.mean(recent_scores) if len(recent_scores) > 0 else 0
            print(
                f"Epoch {epoch:03d} | Episodes: {total_episodes:03d} | Recent Avg Score: {current_avg:.1f} / 500.0"
            )

            if current_avg >= 495.0 and len(recent_scores) == 10:
                print(f"\n✅ SUCCESS! CartPole solved in {total_episodes} episodes!")
                print(
                    "Your PPO math is verified and bulletproof. Proceed to OBELIX Phase 2."
                )
                return

        print(
            "\n❌ FAILURE. The agent did not reach 500. There is a math bug in the PPO implementation."
        )

    def trainNetwork(self, advs_np, returns_np):
        b_states = torch.tensor(self.buffer.states, dtype=torch.float32).to(self.device)
        b_actions = torch.tensor(self.buffer.actions, dtype=torch.float32).to(
            self.device
        )
        b_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32).to(
            self.device
        )
        b_returns = torch.tensor(returns_np, dtype=torch.float32).to(self.device)
        b_advs = torch.tensor(advs_np, dtype=torch.float32).to(self.device)

        # FIX 3: Fetch the old values from the buffer for clipping
        b_values = torch.tensor(self.buffer.values, dtype=torch.float32).to(self.device)

        # Advantage Normalization
        b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        dataset_size = self.steps_per_epoch
        indices = np.arange(dataset_size)

        for _ in range(self.opt_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batchSize):
                end = start + self.batchSize
                mb_idx = indices[start:end]

                mb_states = b_states[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_logprobs = b_logprobs[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_advs = b_advs[mb_idx]
                mb_values_old = b_values[mb_idx]

                # Policy Loss
                _, new_logprobs, entropies, _ = self.pNetwork(mb_states, mb_actions)
                ratios = torch.exp(new_logprobs - mb_logprobs)

                surr1 = ratios * mb_advs
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * mb_advs
                )
                p_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.beta * entropies.mean()

                loss_pi = p_loss + entropy_loss

                self.policyOptimizer.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.pNetwork.parameters(), self.max_grad_norm
                )
                self.policyOptimizer.step()

                # ==========================================
                # FIX 3: VALUE CLIPPING
                # ==========================================
                new_values = self.vNetwork(mb_states).squeeze()
                v_clipped = mb_values_old + torch.clamp(
                    new_values - mb_values_old, -self.clip_range, self.clip_range
                )
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                self.valueOptimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.vNetwork.parameters(), self.max_grad_norm
                )
                self.valueOptimizer.step()


if __name__ == "__main__":
    # Ensure gym output matches expectations
    env = gym.make("CartPole-v1")
    agent = PPO_CartPole(env)
    print("Starting PPO Sanity Check on CartPole-v1...")
    agent.trainAgent()
