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
from collections import deque
from itertools import count

from value_based_deep_helper_functions import (
    FrameStackWrapper,
)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class ValueNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDims, activation):
        super(ValueNetwork, self).__init__()
        self.activation = activation
        # Input is ONLY state now
        self.inputLayer = nn.Linear(stateDim, hiddenDims[0])
        self.hLayers = nn.ModuleList()
        for i in range(len(hiddenDims) - 1):
            hLayer = nn.Linear(hiddenDims[i], hiddenDims[i + 1])
            self.hLayers.append(hLayer)
        # Output is Q-values for ALL actions
        self.out = nn.Linear(hiddenDims[-1], actionDim)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            s = torch.tensor(
                state, dtype=torch.float32, device=next(self.parameters()).device
            )
        else:
            s = state

        l = self.activation(self.inputLayer(s))
        for hLayer in self.hLayers:
            l = self.activation(hLayer(l))
        l = self.out(l)
        return l


class PolicyNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDims, activation):
        super(PolicyNetwork, self).__init__()
        self.activation = activation
        self.actionDim = actionDim

        self.inputLayer = nn.Linear(stateDim, hiddenDims[0])
        self.hLayers = nn.ModuleList()
        for i in range(len(hiddenDims) - 1):
            self.hLayers.append(nn.Linear(hiddenDims[i], hiddenDims[i + 1]))

        # Output layer for categorical logits
        self.out = nn.Linear(hiddenDims[-1], actionDim)
        self.logAlpha = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            s = torch.tensor(
                state, dtype=torch.float32, device=next(self.parameters()).device
            )
        else:
            s = state

        if s.dim() == 1:
            s = s.unsqueeze(0)

        l = self.activation(self.inputLayer(s))
        for hLayer in self.hLayers:
            l = self.activation(hLayer(l))
        logits = self.out(l)

        # Get probabilities using Softmax
        probs = F.softmax(logits, dim=-1)

        # Add epsilon to prevent log(0)
        z = probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(probs + z)

        return probs, log_probs

    def get_action(self, state, deterministic=False):
        probs, _ = self.forward(state)
        probs = probs[0]  # remove batch dim
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        return action


class ReplayBuffer:
    def __init__(self, bufferSize, seed):
        self.bufferSize = bufferSize
        self.seed = seed
        random.seed(self.seed)
        self.experience = deque(maxlen=bufferSize)

    def store(self, transition):
        self.experience.append(transition)

    def sample(self, batchSize):
        return random.sample(self.experience, batchSize)

    def splitExperiences(self, experiences):
        states, actions, rewards, nextStates, dones = zip(*experiences)
        return states, actions, rewards, nextStates, dones

    def length(self):
        return len(self.experience)


class SAC:
    def __init__(
        self,
        env,
        seed,
        gamma,
        tau,
        bufferSize,
        batch_size,
        updateFrequency,
        policyOptimizerFn,
        valueOptimizerFn_1,
        valueOptimizerFn_2,
        policyOptimizerLR,
        valueOptimizerLR,
        alphaOptimizerFn,
        MAX_TRAIN_EPISODES,
        MAX_EVAL_EPISODE,
        MAX_GRAD_NORM_P,
        MAX_GRAD_NORM_V,
        hDim_p,
        hDim_v,
        activation,
        minSamples,
        **kwargs,
    ):
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODE = MAX_EVAL_EPISODE
        self.MAX_GRAD_NORM_P = MAX_GRAD_NORM_P
        self.MAX_GRAD_NORM_V = MAX_GRAD_NORM_V
        self.updateFrequency = updateFrequency
        self.bufferSize = bufferSize
        self.batch_size = batch_size
        self.device = torch.device("cpu")

        self.initBookKeeping()

        self.render = kwargs.get("render", False)

        # DISCRETE spaces have 'n' instead of bounds/shapes
        self.action_dim = 5
        self.state_dim = 18 * kwargs.get("num_frames", 4)

        self.hDim_p = hDim_p
        self.hDim_v = hDim_v
        self.activation = activation

        # Target entropy heuristic for discrete spaces
        self.targetEntropy = -0.98 * np.log(1.0 / self.action_dim)

        self.targetValueNetwork_1 = ValueNetwork(
            self.state_dim, self.action_dim, self.hDim_v, activation=self.activation
        ).to(self.device)
        self.onlineValueNetwork_1 = ValueNetwork(
            self.state_dim, self.action_dim, self.hDim_v, activation=self.activation
        ).to(self.device)
        self.targetValueNetwork_2 = ValueNetwork(
            self.state_dim, self.action_dim, self.hDim_v, activation=self.activation
        ).to(self.device)
        self.onlineValueNetwork_2 = ValueNetwork(
            self.state_dim, self.action_dim, self.hDim_v, activation=self.activation
        ).to(self.device)

        self.policyNetwork = PolicyNetwork(
            self.state_dim, self.action_dim, self.hDim_p, activation=self.activation
        ).to(self.device)
        # --- CURRICULUM LEARNING: LOAD POLICY WEIGHTS ---
        load_path = kwargs.get("load_weights", None)
        if load_path is not None:
            print(f"Loading Policy weights from {load_path} for Phase 2...")
            sd = torch.load(load_path, map_location=self.device)
            # If saved as a dict with 'state_dict' (just in case)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            self.policyNetwork.load_state_dict(sd, strict=True)
        # ------------------------------------------------

        self.alphaOptimizerFn = alphaOptimizerFn
        self.alphaOptimizer = self.alphaOptimizerFn(
            [self.policyNetwork.logAlpha], lr=policyOptimizerLR
        )

        self.policyOptimizerFn = policyOptimizerFn
        self.policyOptimizerLR = policyOptimizerLR
        policy_params = [
            p for name, p in self.policyNetwork.named_parameters() if name != "logAlpha"
        ]
        self.policyOptimizer = self.policyOptimizerFn(
            policy_params, lr=self.policyOptimizerLR
        )

        self.valueOptimizerFn_1 = valueOptimizerFn_1
        self.valueOptimizerFn_2 = valueOptimizerFn_2
        self.valueOptimizerLR = valueOptimizerLR
        self.valueOptimizer_1 = self.valueOptimizerFn_1(
            self.onlineValueNetwork_1.parameters(), lr=self.valueOptimizerLR
        )
        self.valueOptimizer_2 = self.valueOptimizerFn_2(
            self.onlineValueNetwork_2.parameters(), lr=self.valueOptimizerLR
        )

        self.updateValueNetwork(
            self.onlineValueNetwork_1, self.targetValueNetwork_1, 1.0
        )
        self.updateValueNetwork(
            self.onlineValueNetwork_2, self.targetValueNetwork_2, 1.0
        )

        self.rBuffer = ReplayBuffer(self.bufferSize, self.seed)
        self.minSamples = minSamples

    def initBookKeeping(self):
        self.trainRewardsList = [0.0] * self.MAX_TRAIN_EPISODES
        self.finalEvalReward = 0
        self.timeStepEpisode = [0] * self.MAX_TRAIN_EPISODES

    def performBookKeeping(self, train=True):
        return

    def updateValueNetwork(self, onlineNet, targetNet, tau):
        with torch.no_grad():
            for online_param, target_param in zip(
                onlineNet.parameters(), targetNet.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + online_param.data * tau
                )

    def selectRandomAction(self):
        return np.random.randint(0, self.action_dim)

    def runSAC(self):
        self.initBookKeeping()
        trainRewardsList, timeStepEpisode = self.trainAgent()
        resultsEval = self.evaluateAgent()
        self.finalEvalReward = np.mean(resultsEval)
        return (
            trainRewardsList,
            self.finalEvalReward,
            timeStepEpisode,
        )

    def trainAgent(self):
        s = self.env.reset(seed=self.seed)

        for e in tqdm.tqdm(range(self.MAX_TRAIN_EPISODES), desc="Training Episodes"):
            total_time_steps = 0
            if e > 0:
                s = self.env.reset()
            done = False
            while not done:
                total_time_steps += 1

                if self.rBuffer.length() < self.minSamples:
                    a = self.selectRandomAction()
                else:
                    with torch.no_grad():
                        a = self.policyNetwork.get_action(s, deterministic=False)

                s_next, r, done = self.env.step(ACTIONS[a], render=self.render)

                # STORE TERMINATED, NOT DONE
                self.rBuffer.store((s, a, r, s_next, done))

                if self.rBuffer.length() > self.minSamples:
                    experiences = self.rBuffer.sample(self.batch_size)
                    self.trainNetwork(experiences)
                    self.performBookKeeping(train=True)
                self.trainRewardsList[e] += r

                if total_time_steps % self.updateFrequency == 0:
                    self.updateValueNetwork(
                        self.onlineValueNetwork_1, self.targetValueNetwork_1, self.tau
                    )
                    self.updateValueNetwork(
                        self.onlineValueNetwork_2, self.targetValueNetwork_2, self.tau
                    )

                s = s_next
            self.timeStepEpisode[e] = total_time_steps
            self.performBookKeeping(train=False)

        return (self.trainRewardsList, self.timeStepEpisode)

    def trainNetwork(self, experiences):
        ss, a_s, rs, sNexts, dones = self.rBuffer.splitExperiences(experiences)
        ss = torch.tensor(np.stack(ss), dtype=torch.float32, device=self.device)
        a_s = torch.tensor(
            np.array(a_s), dtype=torch.long, device=self.device
        ).unsqueeze(
            1
        )  # int/long for gather
        rs = torch.tensor(
            np.array(rs), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        sNexts = torch.tensor(np.stack(sNexts), dtype=torch.float32, device=self.device)
        dones = torch.tensor(
            np.array(dones), dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        with torch.no_grad():
            probs_next, logp_next = self.policyNetwork(sNexts)
            q_p_1 = self.targetValueNetwork_1(sNexts)
            q_p_2 = self.targetValueNetwork_2(sNexts)
            q_p = torch.min(q_p_1, q_p_2)

            alpha_val = torch.exp(self.policyNetwork.logAlpha).detach()

            # EXACT Expectation over all actions: V(s') = sum_a pi(a) * (Q(s', a) - alpha * log_pi(a))
            v_next = torch.sum(
                probs_next * (q_p - alpha_val * logp_next), dim=1, keepdim=True
            )
            target_q = rs + self.gamma * v_next * (1 - dones)

        # Get Q values for all actions, then gather the ones actually taken
        q_1_all = self.onlineValueNetwork_1(ss)
        q_2_all = self.onlineValueNetwork_2(ss)
        q_1 = q_1_all.gather(1, a_s)
        q_2 = q_2_all.gather(1, a_s)

        q_1_loss = torch.mean(0.5 * (q_1 - target_q) ** 2)
        q_2_loss = torch.mean(0.5 * (q_2 - target_q) ** 2)

        self.valueOptimizer_1.zero_grad()
        q_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.onlineValueNetwork_1.parameters(), self.MAX_GRAD_NORM_V
        )
        self.valueOptimizer_1.step()

        self.valueOptimizer_2.zero_grad()
        q_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.onlineValueNetwork_2.parameters(), self.MAX_GRAD_NORM_V
        )
        self.valueOptimizer_2.step()

        # Policy & Alpha Update
        probs, logp = self.policyNetwork(ss)

        # Freeze value nets
        for p in self.onlineValueNetwork_1.parameters():
            p.requires_grad = False
        for p in self.onlineValueNetwork_2.parameters():
            p.requires_grad = False

        q_current_1 = self.onlineValueNetwork_1(ss)
        q_current_2 = self.onlineValueNetwork_2(ss)
        q_current = torch.min(q_current_1, q_current_2)

        # Policy Loss: Expected advantage over all actions
        policy_loss_term = alpha_val * logp - q_current
        policyLoss = torch.sum(probs * policy_loss_term, dim=1).mean()

        self.policyOptimizer.zero_grad()
        policyLoss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policyNetwork.parameters(), self.MAX_GRAD_NORM_P
        )
        self.policyOptimizer.step()

        # Unfreeze value nets
        for p in self.onlineValueNetwork_1.parameters():
            p.requires_grad = True
        for p in self.onlineValueNetwork_2.parameters():
            p.requires_grad = True

        # Alpha Loss based on exact entropy calculation
        entropy = -torch.sum(probs * logp, dim=1, keepdim=True).detach()
        alphaLoss = torch.mean(
            self.policyNetwork.logAlpha * (entropy - self.targetEntropy)
        )

        self.alphaOptimizer.zero_grad()
        alphaLoss.backward()
        self.alphaOptimizer.step()

    def evaluateAgent(self):
        finalEvalRewardsList = []
        for e in range(self.MAX_EVAL_EPISODE):
            rs = 0
            s = self.env.reset()
            for c in count():
                with torch.no_grad():
                    a = self.policyNetwork.get_action(s, deterministic=True)
                s, r, done = self.env.step(ACTIONS[a], render=self.render)
                rs += r
                if done:
                    finalEvalRewardsList.append(rs)
                    break
        self.performBookKeeping(train=False)
        return finalEvalRewardsList

    def save_weights(self, label):
        # We only need to save the policy network for evaluation/submission
        torch.save(self.policyNetwork.state_dict(), label)
        print(f"SAC Policy weights saved to {label}")


def import_obelix(obelix_py: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    func_map = {
        "func_x": optim.Adam,
        "func_z": optim.Adam,
    }
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_sac_pbrs_frame.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--MAX_GRAD_NORM_P", type=float, default=1.0)
    ap.add_argument("--MAX_GRAD_NORM_V", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--bufferSize", type=int, default=500000)
    ap.add_argument("--batchSize", type=int, default=128)
    ap.add_argument("--minSamples", type=int, default=5000)
    ap.add_argument("--hDim_p", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--hDim_v", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--updateFrequency", type=int, default=1)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--num_frames", type=int, default=4)
    ap.add_argument(
        "--load_weights",
        type=str,
        default=None,
        help="Path to pre-trained weights for Curriculum Learning",
    )
    ap.add_argument(
        "--policyOptimizerFn", type=lambda k: func_map[k], default=optim.Adam
    )
    ap.add_argument("--policyOptimizerLR", type=float, default=0.0005)
    ap.add_argument(
        "--valueOptimizerFn_1", type=lambda k: func_map[k], default=optim.Adam
    )
    ap.add_argument(
        "--valueOptimizerFn_2", type=lambda k: func_map[k], default=optim.Adam
    )
    ap.add_argument("--valueOptimizerLR", type=float, default=0.0005)
    ap.add_argument(
        "--alphaOptimizerFn", type=lambda k: func_map[k], default=optim.Adam
    )

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
    env = FrameStackWrapper(base_env, k=args.num_frames, gamma=args.gamma)
    agent = SAC(
        env,
        args.seed,
        args.gamma,
        args.tau,
        args.bufferSize,
        args.batchSize,
        args.updateFrequency,
        args.policyOptimizerFn,
        args.valueOptimizerFn_1,
        args.valueOptimizerFn_2,
        args.policyOptimizerLR,
        args.valueOptimizerLR,
        args.alphaOptimizerFn,
        args.episodes,
        1,
        args.MAX_GRAD_NORM_P,
        args.MAX_GRAD_NORM_V,
        args.hDim_p,
        args.hDim_v,
        F.relu,
        args.minSamples,
        num_frames=args.num_frames,
        render=args.render,
        load_weights=args.load_weights,
    )
    trainRewardsList, finalEvalReward, totalSteps = agent.runSAC()

    print(f"OBELIX Final Eval Reward: {finalEvalReward}")

    plots_to_make = {
        "OBELIX - Train Rewards": trainRewardsList,
        "OBELIX - Total Steps": totalSteps,
    }

    for title, data in plots_to_make.items():
        plt.figure(figsize=(8, 4))
        plt.plot(data, label=title, color="blue")
        plt.title(title)
        plt.xlabel("Episodes")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.show()

    agent.save_weights(args.out)


if __name__ == "__main__":
    main()
