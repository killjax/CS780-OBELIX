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

from value_based_deep_helper_functions import (
    createDuelingNetwork,
    init_weights,
    selectDecayEpsilonGreedyAction,
    selectGreedyAction,
    FrameStackWrapper,
)
from value_buffer import ReplayBuffer

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class D3QN_PER:
    def __init__(
        self,
        env,
        seed,
        gamma,
        tau,
        alpha,
        beta,
        beta_rate,
        bufferSize,
        batchSize,
        optimizerFn,
        optimizerLR,
        MAX_TRAIN_EPISODES,
        MAX_EVAL_EPISODES,
        explorationStrategyTrainFn,
        explorationStrategyEvalFn,
        updateFrequency,
        **kwargs,
    ):
        # this NFQ method
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc.
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates Q-network using the createDuelingNetwork above
        # 4. creates and initializes (with network params) the optimizer function
        # 5. sets the explorationStartegy variables/functions for train and evaluation
        # 6. sets the batchSize for the number of experiences
        # 7. Creates the replayBuffer

        # Your code goes in here
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.beta_rate = beta_rate
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
        self.updateFrequency = updateFrequency
        self.bufferSize = bufferSize
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(self.device)

        self.initBookKeeping()

        self.inDim = 18 * kwargs.get("num_frames", 4)
        self.outDim = 5
        self.hDim = kwargs.get("hdim", [64, 64])
        self.activation = F.relu
        self.nnOnline = createDuelingNetwork(
            self.inDim, self.outDim, hDim=self.hDim, activation=self.activation
        ).to(self.device)
        if kwargs.get("std_init", True):
            self.nnOnline.apply(init_weights)
        self.nnTarget = createDuelingNetwork(
            self.inDim, self.outDim, hDim=self.hDim, activation=self.activation
        ).to(self.device)
        self.nnTarget.load_state_dict(self.nnOnline.state_dict())

        self.optimizerFn = optimizerFn
        self.optimizerLR = optimizerLR
        self.optimizer = self.optimizerFn(
            self.nnOnline.parameters(), lr=self.optimizerLR
        )

        self.explorationStrategyTrainFn = explorationStrategyTrainFn
        self.explorationStrategyEvalFn = explorationStrategyEvalFn

        self.batchSize = batchSize

        self.rBuffer = ReplayBuffer(
            bufferSize=self.bufferSize,
            bufferType="PER-D3QN",
            epsilon=kwargs.get("epsilon", 0.5),
            temp=kwargs.get("temp", 0.1),
            initial_epsilon_value=kwargs.get("initial_epsilon_value", 1.0),
            final_epsilon_value=kwargs.get("final_epsilon_value", 0.01),
            decay_type=kwargs.get("decay_type", "exponential"),
            decay_rate=kwargs.get("decay_rate", 0.99992),
            warmup_time_steps=kwargs.get("warmup_time_steps", 500),
            alpha=self.alpha,
            beta=self.beta,
            beta_rate=self.beta_rate,
        )
        self.render = kwargs.get("render", False)


class D3QN_PER(D3QN_PER):
    def initBookKeeping(self):
        # this method creates and intializes all the variables required for book-keeping values and it is called
        # init method

        # Your code goes in here
        self.trainRewardsList = np.zeros(self.MAX_TRAIN_EPISODES, dtype=float).tolist()
        self.finalEvalReward = 0
        self.timeStepEpisode = np.zeros(self.MAX_TRAIN_EPISODES, dtype=int).tolist()


class D3QN_PER(D3QN_PER):
    def runD3QN_PER(self):
        # this is the main method, it trains the agent

        self.initBookKeeping()
        (
            trainRewardsList,
            timeStepEpisode,
        ) = self.trainAgent()
        resultsEval = self.evaluateAgent()
        self.finalEvalReward = np.mean(resultsEval)
        finalEvalReward = self.finalEvalReward
        return (
            trainRewardsList,
            finalEvalReward,
            timeStepEpisode,
        )


class D3QN_PER(D3QN_PER):
    def trainAgent(self):
        # this method collects experiences and trains the NFQ agent and does BookKeeping while training.
        # this calls the trainNetwork() method internally, it also evaluates the agent per episode
        # it trains the agent for MAX_TRAIN_EPISODES

        # Your code goes in here

        self.updateNetwork(self.nnOnline, self.nnTarget)
        s = self.env.reset(seed=self.seed)
        total_time_steps = 0
        for e in tqdm.tqdm(range(self.MAX_TRAIN_EPISODES), desc="Training Episodes"):
            if e > 0:
                s = self.env.reset()
            done = False
            while not done:
                s, time_steps, total_reward, done = self.rBuffer.collectExperiences(
                    self.env,
                    s,
                    self.explorationStrategyTrainFn,
                    1,
                    net=self.nnOnline,
                    render=self.render,
                )
                if self.rBuffer.length() >= self.batchSize:
                    experiences = self.rBuffer.sample(self.batchSize)
                    self.trainNetwork(experiences, 1)
                self.timeStepEpisode[e] += time_steps
                total_time_steps += time_steps
                self.trainRewardsList[e] += total_reward
                if total_time_steps % self.updateFrequency == 0:
                    self.updateNetwork(self.nnOnline, self.nnTarget)

        return (
            self.trainRewardsList,
            self.timeStepEpisode,
        )


class D3QN_PER(D3QN_PER):
    def trainNetwork(self, experiences, epochs):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc.

        # Your code goes in here

        ss, a_s, rs, sNexts, dones, indices, weights = self.rBuffer.splitExperiences(
            experiences
        )
        ss = torch.tensor(np.array(ss), dtype=torch.float32, device=self.device)
        a_s = torch.tensor(
            np.array(a_s), dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rs = torch.tensor(
            np.array(rs), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        sNexts = torch.tensor(np.array(sNexts), dtype=torch.float32, device=self.device)
        dones = torch.tensor(
            np.array(dones), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        weights = torch.tensor(
            np.array(weights), dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        with torch.no_grad():
            argmax_a_qs = self.nnOnline.forward(sNexts).argmax(dim=1, keepdim=True)
            qs = self.nnTarget(sNexts)
            max_a_qs = qs.gather(1, argmax_a_qs)
            tdTargets = rs + self.gamma * max_a_qs * (1 - dones)

        qs = self.nnOnline.forward(ss).gather(1, a_s)
        loss = (weights * F.mse_loss(qs, tdTargets, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.nnOnline.parameters(), max_norm=1.0)
        self.optimizer.step()
        with torch.no_grad():
            priorities = (torch.abs(tdTargets - qs) + 1e-6).squeeze().tolist()
        self.rBuffer.update(indices, priorities)


class D3QN_PER(D3QN_PER):
    def updateNetwork(self, onlineNet, targetNet):
        # this function updates the onlineNetwork with the target network
        #
        # Your code goes in here
        #
        onlineNet_state_dict = onlineNet.state_dict()
        targetNet_state_dict = targetNet.state_dict()
        for key in onlineNet_state_dict:
            targetNet_state_dict[key] = onlineNet_state_dict[
                key
            ] * self.tau + targetNet_state_dict[key] * (1 - self.tau)
        targetNet.load_state_dict(targetNet_state_dict)


class D3QN_PER(D3QN_PER):
    def evaluateAgent(self):
        # this function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        # typcially MAX_EVAL_EPISODES = 1

        # Your code goes in here

        finalEvalRewardsList = []
        for e in range(self.MAX_EVAL_EPISODES):
            rs = 0
            s = self.env.reset()
            done = False
            while not done:
                a = self.explorationStrategyEvalFn(self.nnOnline, s)
                s, r, done = self.env.step(ACTIONS[a], render=self.render)
                rs += r
                if done:
                    finalEvalRewardsList.append(rs)

        return finalEvalRewardsList


class D3QN_PER(D3QN_PER):
    def save_weights(self, label):
        torch.save(self.nnOnline.state_dict(), label)
        print(f"weights-table saved to {label}")


def import_obelix(obelix_py: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    func_map = {
        "func_x": selectDecayEpsilonGreedyAction,
        "func_y": selectGreedyAction,
        "func_z": optim.RMSprop,
    }
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_D3qn_PER_frame_stack.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--initial_epsilon_value", type=float, default=1.0)
    ap.add_argument("--final_epsilon_value", type=float, default=0.01)
    ap.add_argument("--decay_rate", type=float, default=(0.05 / 1.0) ** (1 / 349999))
    ap.add_argument("--decay_type", type=str, default="exponential")

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--bufferSize", type=int, default=500000)
    ap.add_argument("--batchSize", type=int, default=128)
    ap.add_argument("--std_init", type=bool, default=False)
    ap.add_argument("--warmup_time_steps", type=int, default=5000)
    ap.add_argument("--hdim", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--updateFrequency", type=int, default=15)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--beta_rate", type=float, default=0.99999)
    ap.add_argument("--num_frames", type=int, default=4)

    ap.add_argument(
        "--explorationStrategyTrainFn",
        type=lambda k: func_map[k],
        default=selectDecayEpsilonGreedyAction,
    )
    ap.add_argument(
        "--explorationStrategyEvalFn",
        type=lambda k: func_map[k],
        default=selectGreedyAction,
    )
    ap.add_argument("--optimizerFn", type=lambda k: func_map[k], default=optim.RMSprop)
    ap.add_argument("--optimizerLR", type=float, default=0.0005)

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
    env = FrameStackWrapper(base_env, k=args.num_frames)
    agent = D3QN_PER(
        env,
        args.seed,
        args.gamma,
        args.tau,
        args.alpha,
        args.beta,
        args.beta_rate,
        bufferSize=args.bufferSize,
        batchSize=args.batchSize,
        optimizerFn=args.optimizerFn,
        optimizerLR=args.optimizerLR,
        MAX_TRAIN_EPISODES=args.episodes,
        MAX_EVAL_EPISODES=1,
        explorationStrategyTrainFn=args.explorationStrategyTrainFn,
        explorationStrategyEvalFn=args.explorationStrategyEvalFn,
        updateFrequency=args.updateFrequency,
        hdim=args.hdim,
        std_init=args.std_init,
        initial_epsilon_value=args.initial_epsilon_value,
        final_epsilon_value=args.final_epsilon_value,
        decay_type=args.decay_type,
        decay_rate=args.decay_rate,
        warmup_time_steps=args.warmup_time_steps,
        render=args.render,
        num_frames=args.num_frames,
    )
    trainRewardsList, finalEvalReward, totalSteps = agent.runD3QN_PER()

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
