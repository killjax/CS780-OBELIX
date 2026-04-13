from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import tqdm
import argparse
import matplotlib.pyplot as plt
from itertools import count
from collections import deque

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class Lstm_Pbrs_Wrapper:
    def __init__(self, env, k=4, gamma=0.999):
        self.env = env
        self.k = k
        self.gamma = gamma
        self.frames = deque([], maxlen=k)
        self.current_potential = 0.0

    def _get_potential(self):
        bot_x = self.env.bot_center_x
        bot_y = self.env.bot_center_y
        box_x = self.env.box_center_x
        box_y = self.env.box_center_y

        scaling_factor = 0.1
        radar_range = 30 * self.env.scaling_factor  # 150 pixels for scaling_factor=5

        distance_to_box = np.sqrt((bot_x - box_x) ** 2 + (bot_y - box_y) ** 2)
        effective_distance = min(distance_to_box, radar_range)
        return -effective_distance * scaling_factor

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        self.current_potential = self._get_potential()
        return self._get_ob()

    def step(self, action, **kwargs):
        obs, base_reward, done = self.env.step(action, **kwargs)
        self.frames.append(obs)

        # --- PBRS MATH ---
        next_potential = self._get_potential()
        shaped_reward = (self.gamma * next_potential) - self.current_potential
        self.current_potential = next_potential
        total_reward = base_reward + shaped_reward
        return self._get_ob(), total_reward, done

    def _get_ob(self):
        return np.stack(list(self.frames), axis=0)


# 1. Neural Networks (LSTM)
class ValueNetwork(nn.Module):
    def __init__(self, stateDim, hiddenDims, activation):
        super(ValueNetwork, self).__init__()
        self.activation = activation

        self.lstm = nn.LSTM(
            input_size=stateDim, hidden_size=hiddenDims[0], batch_first=True
        )

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

        if ss.dim() == 2:
            ss = ss.unsqueeze(0)

        lstm_out, _ = self.lstm(ss)
        l = lstm_out[:, -1, :]

        for hLayer in self.hLayers:
            l = self.activation(hLayer(l))
        q = self.out(l)
        return q


class PolicyNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDims, activation):
        super(PolicyNetwork, self).__init__()
        self.activation = activation
        self.actionDim = actionDim

        self.lstm = nn.LSTM(
            input_size=stateDim, hidden_size=hiddenDims[0], batch_first=True
        )

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

        if ss.dim() == 2:
            ss = ss.unsqueeze(0)

        lstm_out, _ = self.lstm(ss)
        l = lstm_out[:, -1, :]

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


# 2. Single-Worker rBuffer
class rBuffer:
    def __init__(self, buffer_size, seq_len, feature_dim):
        self.states = np.zeros((buffer_size, seq_len, feature_dim), dtype=np.float32)
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


# 3. PPO Agent
class PPO:
    def __init__(self, env, args, activation=F.relu):
        self.env = env
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.actionDim = 5
        self.stateDim = 18
        self.seqLen = args.num_frames

        self.pNetwork = PolicyNetwork(
            self.stateDim, self.actionDim, args.hDim_p, activation
        ).to(self.device)
        self.vNetwork = ValueNetwork(self.stateDim, args.hDim_v, activation).to(
            self.device
        )

        # --- CURRICULUM LEARNING ---
        if args.load_weights is not None:
            print(f"Loading Policy weights from {args.load_weights}...")
            sd = torch.load(args.load_weights, map_location=self.device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            self.pNetwork.load_state_dict(sd, strict=True)
        # ---------------------------

        self.policyOptimizer = optim.Adam(self.pNetwork.parameters(), lr=args.policyLR)
        self.valueOptimizer = optim.Adam(self.vNetwork.parameters(), lr=args.valueLR)

        self.rbuffer = rBuffer(args.steps_per_epoch, self.seqLen, self.stateDim)

        self.trainRewardsList = []
        self.timeStepEpisode = []

    def trainAgent(self):
        s = self.env.reset(seed=self.args.seed)
        ep_reward = 0
        ep_step = 0

        epochs = self.args.episodes // (
            self.args.steps_per_epoch // self.args.max_steps + 1
        )
        recent_scores = deque(maxlen=20)
        pbar = tqdm.tqdm(range(epochs), desc="PPO Epochs")

        for epoch in pbar:
            self.rbuffer.clear()

            for t in range(self.args.steps_per_epoch):
                with torch.no_grad():
                    action_tensor, logp_tensor, _, _ = self.pNetwork(s)
                    val_tensor = self.vNetwork(s)

                    a = action_tensor.item()
                    logp = logp_tensor.item()
                    v = val_tensor.item()

                s_next, r, done = self.env.step(ACTIONS[a], render=self.args.render)
                ep_reward += r
                ep_step += 1

                self.rbuffer.store(s, a, r, v, logp, done)
                s = s_next

                if done or ep_step >= self.args.max_steps:
                    self.trainRewardsList.append(ep_reward)
                    self.timeStepEpisode.append(ep_step)
                    recent_scores.append(ep_reward)
                    s = self.env.reset()
                    ep_reward = 0
                    ep_step = 0
                else:
                    s = s_next

            current_avg = np.mean(recent_scores) if len(recent_scores) > 0 else 0
            pbar.set_postfix({"Avg Reward": f"{current_avg:.1f}"})

            with torch.no_grad():
                if ep_step == 0:
                    last_val = 0.0
                    last_done = 1.0
                else:
                    last_val = self.vNetwork(s).item()
                    last_done = 0.0

            advs, returns = self.rbuffer.compute_gae(
                last_val, last_done, self.args.gamma, self.args.lamda
            )

            self.trainNetwork(advs, returns)

        return self.trainRewardsList, self.timeStepEpisode

    def trainNetwork(self, advs_np, returns_np):
        b_states = torch.tensor(self.rbuffer.states, dtype=torch.float32).to(
            self.device
        )
        b_actions = torch.tensor(self.rbuffer.actions, dtype=torch.float32).to(
            self.device
        )
        b_logprobs = torch.tensor(self.rbuffer.logprobs, dtype=torch.float32).to(
            self.device
        )
        b_returns = torch.tensor(returns_np, dtype=torch.float32).to(self.device)
        b_advs = torch.tensor(advs_np, dtype=torch.float32).to(self.device)

        b_values = torch.tensor(self.rbuffer.values, dtype=torch.float32).to(
            self.device
        )

        b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        dataset_size = self.args.steps_per_epoch
        indices = np.arange(dataset_size)

        for _ in range(self.args.opt_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.args.batchSize):
                end = start + self.args.batchSize
                mb_idx = indices[start:end]

                mb_states = b_states[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_logprobs = b_logprobs[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_advs = b_advs[mb_idx]
                mb_values_old = b_values[mb_idx]

                _, new_logprobs, entropies, _ = self.pNetwork(mb_states, mb_actions)
                ratios = torch.exp(new_logprobs - mb_logprobs)

                surr1 = ratios * mb_advs
                surr2 = (
                    torch.clamp(
                        ratios, 1.0 - self.args.clip_range, 1.0 + self.args.clip_range
                    )
                    * mb_advs
                )
                p_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.args.beta * entropies.mean()

                loss_pi = p_loss + entropy_loss

                self.policyOptimizer.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.pNetwork.parameters(), self.args.max_grad_norm
                )
                self.policyOptimizer.step()

                new_values = self.vNetwork(mb_states).squeeze()
                v_clipped = mb_values_old + torch.clamp(
                    new_values - mb_values_old,
                    -self.args.clip_range,
                    self.args.clip_range,
                )

                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                self.valueOptimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.vNetwork.parameters(), self.args.max_grad_norm
                )
                self.valueOptimizer.step()

    def evaluateAgent(self):
        finalEvalRewardsList = []
        for e in range(10):  # Evaluate 10 times
            rs = 0
            s = self.env.reset()
            for c in count():
                a = self.pNetwork.get_action(s, deterministic=True)
                s, r, done = self.env.step(ACTIONS[a], render=self.args.render)
                rs += r
                if done or c >= self.args.max_steps:
                    finalEvalRewardsList.append(rs)
                    break
        return finalEvalRewardsList

    def save_weights(self, label):
        torch.save(self.pNetwork.state_dict(), label)
        print(f"PPO Policy weights saved to {label}")


def import_obelix(obelix_py: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_ppo_pbrs_lstm.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=400)

    # Environment Args
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--num_frames", type=int, default=4)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # PPO Specific Hyperparameters
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument(
        "--beta", type=float, default=0.01, help="Entropy coefficient for exploration"
    )
    ap.add_argument("--policyLR", type=float, default=3e-4)
    ap.add_argument("--valueLR", type=float, default=1e-3)
    ap.add_argument(
        "--steps_per_epoch",
        type=int,
        default=2000,
        help="Steps to collect before updating",
    )
    ap.add_argument(
        "--opt_epochs",
        type=int,
        default=10,
        help="Number of times to update network per epoch",
    )
    ap.add_argument("--batchSize", type=int, default=64)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)

    # Network Architecture
    ap.add_argument("--hDim_p", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--hDim_v", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--load_weights", type=str, default=None)

    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    base_env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    env = Lstm_Pbrs_Wrapper(base_env, k=args.num_frames, gamma=args.gamma)

    agent = PPO(env, args)

    try:
        trainRewardsList, totalSteps = agent.trainAgent()
    except KeyboardInterrupt:
        print("\n\n🚨 Training manually interrupted by user (Ctrl+C).")
        print("Safely halting and saving the current brain...")
        trainRewardsList = agent.trainRewardsList
        totalSteps = agent.timeStepEpisode

    agent.save_weights(args.out)
    print("Evaluating current policy...")

    eval_rewards = agent.evaluateAgent()
    finalEvalReward = np.mean(eval_rewards)
    print(f"OBELIX Final Eval Reward (Mean): {finalEvalReward}")

    plots_to_make = {
        "OBELIX - Train Rewards": trainRewardsList,
        "OBELIX - Episode Steps": totalSteps,
    }

    for title, data in plots_to_make.items():
        if len(data) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(data, label=title, color="blue")
            plt.title(title)
            plt.xlabel("Episodes")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    main()
