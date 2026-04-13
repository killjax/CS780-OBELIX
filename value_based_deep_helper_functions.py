from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque


def selectGreedyAction(net, state):
    # this function gets q-values via the network and selects greedy action from q-values and returns it

    # Your code goes in here
    with torch.no_grad():
        s = torch.tensor(
            state, dtype=torch.float32, device=next(net.parameters()).device
        ).unsqueeze(0)
        Q = net(s)
        greedyAction = torch.argmax(Q).item()
    return greedyAction


def selectEpsilonGreedyAction(net, state, epsilon):
    # this function gets q-values via the network and selects an action from q-values using epsilon greedy strategy
    # and returns it
    # note this function can be used for decaying epsilon greedy strategy,
    # you would need to create a wrapper function that will handle decaying epsilon
    # you can create this wrapper in this helper function section
    # for the agents you would be implementing it would be nice to play with decaying parameter to get optimal results

    # Your code goes in here
    with torch.no_grad():
        s = torch.tensor(
            state, dtype=torch.float32, device=next(net.parameters()).device
        ).unsqueeze(0)
        rand_val = np.random.random()
        Q = net(s)
        if rand_val > epsilon:
            eGreedyAction = torch.argmax(Q).item()
        else:
            out_dim = Q.shape[-1]
            eGreedyAction = np.random.randint(0, out_dim)

    return eGreedyAction


def selectSoftMaxAction(net, state, temp):
    # this function gets q-values via the network and selects an action from q-values using softmax strategy
    # and returns it
    # note this function can be used for decaying temperature softmax strategy,
    # you would need to create a wrapper function that will handle decaying temperature
    # you can create this wrapper in this helper function section
    # for the agents you would be implementing it would be nice to play with decaying parameter to get optimal results

    # Your code goes in here
    with torch.no_grad():
        s = torch.tensor(
            state, dtype=torch.float32, device=next(net.parameters()).device
        ).unsqueeze(0)
        Q = net(s)
        probs = torch.softmax(Q / temp, dim=-1)
        softAction = torch.multinomial(probs, num_samples=1).item()

    return softAction


# Value Network
def createValueNetwork(inDim, outDim, hDim=[64, 64], activation=F.relu):
    # this creates a Feed Forward Neural Network class and instantiates it and returns the class
    # the class should be derived from torch nn.Module and it should have init and forward method at the very least
    # the forward function should return q-value for each possible action

    # Your code goes in here
    class valueNetwork(nn.Module):
        def __init__(self):
            super(valueNetwork, self).__init__()
            self.ffn = nn.ModuleList()
            self.ffn.append(nn.Linear(inDim, hDim[0]))
            for i in range(len(hDim)):
                if i == 0:
                    continue
                self.ffn.append(nn.Linear(hDim[i - 1], hDim[i]))
            self.ffn.append(nn.Linear(hDim[len(hDim) - 1], outDim))
            self.activation = activation

        def forward(self, x):
            for i in range(len(hDim)):
                x = self.activation(self.ffn[i](x))
            x = self.ffn[len(hDim)](x)
            return x

    return valueNetwork()


# in case you want to create any other helper function, the code goes in here
def selectDecayEpsilonGreedyAction(
    net, state, initial_value, final_value, episode, decay_rate, decay_type
):
    if decay_type == "exponential":
        epsilon = initial_value * (decay_rate**episode)
    else:
        epsilon = initial_value + decay_rate * episode

    epsilon = max(epsilon, final_value)

    return selectEpsilonGreedyAction(net, state, epsilon)


# in case you want to create any other helper function, the code goes in here
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.5, b=0.5)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-0.5, b=0.5)


# Dueling Network
def createDuelingNetwork(inDim, outDim, hDim=[32, 32], activation=F.relu):
    # this creates a Feed Forward Neural Network class and instantiates it and returns the class
    # the class should be derived from torch nn.Module and it should have init and forward method at the very least
    # the forward function should return q-value which is derived
    # internally from action-advantage function and v-function,
    # Note we center the advantage values, basically we subtract the mean from each state-action value

    # Your code goes in here
    class duelingNetwork(nn.Module):
        def __init__(self):
            super(duelingNetwork, self).__init__()
            self.ffn = nn.ModuleList()
            self.ffn.append(nn.Linear(inDim, hDim[0]))
            for i in range(len(hDim)):
                if i == 0:
                    continue
                self.ffn.append(nn.Linear(hDim[i - 1], hDim[i]))
            self.value_network = nn.Linear(hDim[len(hDim) - 1], 1)
            self.advantage_network = nn.Linear(hDim[len(hDim) - 1], outDim)
            self.activation = activation

        def forward(self, x):
            for i in range(len(hDim)):
                x = self.activation(self.ffn[i](x))
            V = self.value_network(x)
            A = self.advantage_network(x)
            Q = V + (A - A.mean(dim=1, keepdim=True))
            return Q

    return duelingNetwork()


class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
            self.min_value = min(self.left.min_value, self.right.min_value)
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        if value == 0:
            leaf.min_value = float("inf")
        else:
            leaf.min_value = value
        return leaf


def create_tree(input: list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        next_nodes = []
        for i in range(0, len(nodes) - 1, 2):
            next_nodes.append(Node(nodes[i], nodes[i + 1]))
        if len(nodes) % 2 == 1:
            next_nodes.append(nodes[-1])
        nodes = next_nodes

    return nodes[0], leaf_nodes


def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node

    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)


def update(node: Node, new_value: float):
    change = new_value - node.value

    node.value = new_value
    node.min_value = new_value
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    node.value += change
    node.min_value = min(node.left.min_value, node.right.min_value)

    if node.parent is not None:
        propagate_changes(change, node.parent)


def demonstrate_sampling(root_node: Node, batchSize):
    tree_total = root_node.value
    selected_indices = []
    for i in range(batchSize):
        rand_val = np.random.uniform(0, tree_total)
        selected_val = retrieve(rand_val, root_node).idx
        selected_indices.append(selected_val)

    return selected_indices


def selectSoftMaxAction(net, state, temp):
    # this function gets q-values via the network and selects an action from q-values using softmax strategy
    # and returns it
    # note this function can be used for decaying temperature softmax strategy,
    # you would need to create a wrapper function that will handle decaying temperature
    # you can create this wrapper in this helper function section
    # for the agents you would be implementing it would be nice to play with decaying parameter to get optimal results

    # Your code goes in here
    with torch.no_grad():
        s = torch.tensor(
            state, dtype=torch.float32, device=next(net.parameters()).device
        ).unsqueeze(0)
        Q = net(s)
        probs = torch.softmax(Q / temp, dim=-1)
        softAction = torch.multinomial(probs, num_samples=1).item()

    return softAction


# class FrameStackWrapper:
#     def __init__(self, env, k=4, gamma=0.999):
#         self.env = env
#         self.k = k
#         self.gamma = gamma
#         self.frames = deque([], maxlen=k)
#         self.current_potential = 0.0

#     def _get_potential(self):
#         # 1. Extract exact coordinates from the OBELIX environment instance
#         bot_x = self.env.bot_center_x
#         bot_y = self.env.bot_center_y
#         box_x = self.env.box_center_x
#         box_y = self.env.box_center_y

#         # 2. Calculate Euclidean Distance
#         distance = np.sqrt((bot_x - box_x) ** 2 + (bot_y - box_y) ** 2)

#         # 3. Calculate Potential
#         # Potential is negative distance. We scale it by 0.1 so that it provides
#         # a gentle "pull" without completely overpowering the base game rewards.
#         scaling_factor = 0.1
#         return -distance * scaling_factor

#     def reset(self, **kwargs):
#         obs = self.env.reset(**kwargs)
#         # Fill the frame stack
#         for _ in range(self.k):
#             self.frames.append(obs)

#         # Initialize the potential at the exact starting position
#         self.current_potential = self._get_potential()

#         return self._get_ob()

#     def step(self, action, **kwargs):
#         # Take a step in the actual OBELIX environment
#         obs, base_reward, done = self.env.step(action, **kwargs)
#         self.frames.append(obs)

#         # --- PBRS MATH ---
#         # 1. Get the new potential after moving
#         next_potential = self._get_potential()

#         # 2. Calculate shaped reward: F = (gamma * Next_Potential) - Current_Potential
#         shaped_reward = (self.gamma * next_potential) - self.current_potential

#         # 3. Save the new potential for the next frame
#         self.current_potential = next_potential

#         # 4. Combine the OBELIX base reward with our new shaped breadcrumb reward
#         total_reward = base_reward + shaped_reward
#         # -----------------

#         return self._get_ob(), total_reward, done

#     def _get_ob(self):
#         # Flattens the 4 frames into a 72-value array
#         return np.concatenate(list(self.frames), axis=0)

import numpy as np
from collections import deque


class FrameStackWrapper:
    def __init__(self, env, k=4, gamma=0.999):
        self.env = env
        self.k = k
        self.gamma = gamma
        self.frames = deque([], maxlen=k)
        self.current_potential = 0.0

    def _get_potential(self):
        # 1. Extract exact coordinates from the OBELIX environment instance
        bot_x = self.env.bot_center_x
        bot_y = self.env.bot_center_y
        box_x = self.env.box_center_x
        box_y = self.env.box_center_y

        scaling_factor = 0.1
        radar_range = 30 * self.env.scaling_factor  # 150 pixels for scaling_factor=5

        # ---------------------------------------------------------
        # PHASE C: DESTROY (The box is attached)
        # ---------------------------------------------------------
        if self.env.enable_push:
            # We want to pull the box to the nearest inner boundary (10 pixels from edge)
            max_x = self.env.frame_size[1] - 10
            max_y = self.env.frame_size[0] - 10
            min_x = 10
            min_y = 10

            dist_to_left = box_x - min_x
            dist_to_right = max_x - box_x
            dist_to_top = box_y - min_y
            dist_to_bottom = max_y - box_y

            distance_to_nearest_boundary = min(
                dist_to_left, dist_to_right, dist_to_top, dist_to_bottom
            )

            # The potential is strictly based on finishing the push
            return -distance_to_nearest_boundary * scaling_factor

        # ---------------------------------------------------------
        # PHASES A & B: SEARCH & DETECT
        # ---------------------------------------------------------
        else:
            # Calculate true mathematical distance
            distance_to_box = np.sqrt((bot_x - box_x) ** 2 + (bot_y - box_y) ** 2)

            # The Continuous Flat Floor:
            # If distance > radar_range, this equals radar_range (e.g., -15.0). The agent feels no pull.
            # If distance < radar_range, it equals the real distance. The agent feels the pull.
            effective_distance = min(distance_to_box, radar_range)

            return -effective_distance * scaling_factor

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Fill the frame stack
        for _ in range(self.k):
            self.frames.append(obs)

        # Initialize the potential at the exact starting position
        self.current_potential = self._get_potential()

        return self._get_ob()

    def step(self, action, **kwargs):
        # Take a step in the actual OBELIX environment
        obs, base_reward, done = self.env.step(action, **kwargs)
        self.frames.append(obs)

        # --- PBRS MATH ---
        # 1. Get the new potential after moving
        next_potential = self._get_potential()

        # 2. Calculate shaped reward: F = (gamma * Next_Potential) - Current_Potential
        shaped_reward = (self.gamma * next_potential) - self.current_potential

        # 3. Save the new potential for the next frame
        self.current_potential = next_potential

        # 4. Combine the OBELIX base reward with our new shaped breadcrumb reward
        total_reward = base_reward + shaped_reward
        # -----------------

        return self._get_ob(), total_reward, done

    def _get_ob(self):
        # Flattens the 4 frames into a 72-value array
        return np.concatenate(list(self.frames), axis=0)
