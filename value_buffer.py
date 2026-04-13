from __future__ import annotations
from collections import deque
import random
from value_based_deep_helper_functions import (
    selectGreedyAction,
    selectDecayEpsilonGreedyAction,
    selectSoftMaxAction,
    selectEpsilonGreedyAction,
    create_tree,
    update,
    demonstrate_sampling,
)

alpha = 0.6
beta = 0.1
beta_rate = 0.99992
epsilon = 0.5
temp = 0.1
initial_epsilon_value = 1.0
final_epsilon_value = 0.01
decay_type = "exponential"
if decay_type == "exponential":
    decay_rate = (0.2 / 1.0) ** (1 / 19999)
else:
    decay_rate = (0.2 - 1.0) / 19999
warmup_time_steps = 500

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class ReplayBuffer:
    def __init__(self, bufferSize, bufferType="DQN", **kwargs):
        # this function creates the relevant data-structures, and intializes all relevant variables
        # it can take variable number of parameters like alpha, beta, beta_rate (required for PER)
        # here the bufferType variable can be used to maintain one class for all types of agents
        # using the bufferType parameter in the methods below, you can implement all possible functionalities
        # that could be used for different types of agents
        # permissible values for bufferType = NFQ, DQN, DDQN, D3QN and PER-D3QN

        # Your code goes in here
        self.bufferSize = bufferSize
        self.bufferType = bufferType
        if self.bufferType == "NFQ" or self.bufferType == "PER-D3QN":
            self.experience = []
            if self.bufferType == "PER-D3QN":
                self.alpha = kwargs.get("alpha", alpha)
                self.beta = kwargs.get("beta", beta)
                self.beta_rate = kwargs.get("beta_rate", beta_rate)
                self.alpha_power_sum_node, self.alpha_priorities = create_tree(
                    [0] * bufferSize
                )
                self.ok = False
                self.leaf_node_pointer = 0
                self.max_alpha_priority = 1
        else:
            self.experience = deque(maxlen=bufferSize)
        self.epsilon = kwargs.get("epsilon", epsilon)
        self.temp = kwargs.get("temp", temp)
        self.initial_epsilon_value = kwargs.get(
            "initial_epsilon_value", initial_epsilon_value
        )
        self.final_epsilon_value = kwargs.get(
            "final_epsilon_value", final_epsilon_value
        )
        self.decay_type = kwargs.get("decay_type", decay_type)
        self.decay_rate = kwargs.get("decay_rate", decay_rate)
        self.warmup_time_steps = kwargs.get("warmup_time_steps", warmup_time_steps)
        self.time_step_counter = 0
        return


class ReplayBuffer(ReplayBuffer):
    def store(self, experience):
        # stores the experiences, based on parameters in init it can assign priorities, etc.
        #
        # this function does not return anything
        #
        # Your code goes in here
        if self.bufferType == "PER-D3QN" and self.ok:
            self.experience[self.leaf_node_pointer] = experience
        else:
            self.experience.append(experience)
        if len(self.experience) == self.bufferSize:
            self.ok = True

        return


class ReplayBuffer(ReplayBuffer):
    def update(self, indices, priorities):
        # this is mainly used for PER-DDQN
        # otherwise just have a pass in this method
        #
        # this function does not return anything
        #
        # Your code goes in here
        for idx, i in enumerate(indices):
            self.max_alpha_priority = max(
                self.max_alpha_priority, priorities[idx] ** self.alpha
            )
            update(self.alpha_priorities[i], priorities[idx] ** self.alpha)

        return


class ReplayBuffer(ReplayBuffer):
    def collectExperiences(
        self, env, state, explorationStrategy, countExperiences, net=None, render=False
    ):
        # this method allows the agent to interact with the environment starting from a state and it collects
        # experiences during the interaction, it uses network to get the value function and uses exploration strategy
        # to select action. It collects countExperiences and in case the environment terminates before that it returns
        # the function calling this method needs to handle early termination accordingly.
        #
        # this function does not return anything
        #
        # Your code goes in here
        s = state
        time_steps = 0
        total_reward = 0
        for i in range(countExperiences):
            if explorationStrategy == selectGreedyAction:
                a = explorationStrategy(net, s)
            elif explorationStrategy == selectEpsilonGreedyAction:
                a = explorationStrategy(net, s, self.epsilon)
            elif explorationStrategy == selectSoftMaxAction:
                a = explorationStrategy(net, s, self.temp)
            elif explorationStrategy == selectDecayEpsilonGreedyAction:
                if self.length() > self.warmup_time_steps:
                    self.time_step_counter += 1
                a = explorationStrategy(
                    net,
                    s,
                    self.initial_epsilon_value,
                    self.final_epsilon_value,
                    self.time_step_counter,
                    self.decay_rate,
                    self.decay_type,
                )

            n_s, r, done = env.step(ACTIONS[a], render=render)
            if self.bufferType == "PER-D3QN":
                update(
                    self.alpha_priorities[self.leaf_node_pointer],
                    self.max_alpha_priority,
                )
                self.store((s, a, r, n_s, done))
                self.leaf_node_pointer = (self.leaf_node_pointer + 1) % self.bufferSize
                self.beta = 1 - ((1 - self.beta) * self.beta_rate)
            else:
                self.store((s, a, r, n_s, done))
            time_steps += 1
            total_reward += r
            if done:
                s = env.reset()
                break
            else:
                s = n_s
        return s, time_steps, total_reward, done


class ReplayBuffer(ReplayBuffer):
    def sample(self, batchSize, **kwargs):
        # this method returns batchSize number of experiences
        # based on extra arguments, it could do sampling or it could return the latest batchSize experiences or
        # via some other strategy
        #
        # in the case of Prioritized Experience Replay (PER) the sampling needs to take into account the priorities
        #
        # this function returns experiences samples
        #
        # Your code goes in here
        if self.bufferType == "PER-D3QN":
            experiencesList = []
            indices = demonstrate_sampling(self.alpha_power_sum_node, batchSize)
            weight_max = (
                self.length()
                * (
                    self.alpha_power_sum_node.min_value
                    / self.alpha_power_sum_node.value
                )
            ) ** (-self.beta)
            for i in indices:
                s, a, r, n_s, done = self.experience[i]
                weight = (
                    (
                        self.length()
                        * (
                            self.alpha_priorities[i].value
                            / self.alpha_power_sum_node.value
                        )
                    )
                    ** (-self.beta)
                ) / weight_max
                experiencesList.append((s, a, r, n_s, done, i, weight))
            return experiencesList
        experiencesList = random.sample(self.experience, batchSize)
        return experiencesList


class ReplayBuffer(ReplayBuffer):
    def splitExperiences(self, experiences):
        # it takes in experiences and gives the following:
        # states, actions, rewards, nextStates, dones
        #
        # Your code goes in here
        #
        states = [experiences[i][0] for i in range(len(experiences))]
        actions = [experiences[i][1] for i in range(len(experiences))]
        rewards = [experiences[i][2] for i in range(len(experiences))]
        nextStates = [experiences[i][3] for i in range(len(experiences))]
        dones = [experiences[i][4] for i in range(len(experiences))]
        if self.bufferType == "PER-D3QN":
            indices = [experiences[i][5] for i in range(len(experiences))]
            weights = [experiences[i][6] for i in range(len(experiences))]
            return states, actions, rewards, nextStates, dones, indices, weights
        return states, actions, rewards, nextStates, dones


class ReplayBuffer(ReplayBuffer):
    def length(self):
        # tells the number of experiences stored in the internal buffer
        #
        # Your code goes in here
        #
        return len(self.experience)
