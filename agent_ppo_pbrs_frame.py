import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_LAST_ACTION = None
_REPEAT_COUNT = 0

_FRAME_QUEUE = deque([], maxlen=4)


def _load_once():
    """Load the trained PPO Policy model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights_ppo_phase1b.pth")

    class PolicyNetwork(nn.Module):
        def __init__(self, stateDim, actionDim, hiddenDims, activation=F.relu):
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

    model = PolicyNetwork(stateDim=72, actionDim=5, hiddenDims=[64, 64])

    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd, strict=True)
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_ACTION, _REPEAT_COUNT, _FRAME_QUEUE
    _load_once()

    if len(_FRAME_QUEUE) == 0:
        for _ in range(4):
            _FRAME_QUEUE.append(obs)
    else:
        _FRAME_QUEUE.append(obs)

    stacked_obs = np.concatenate(list(_FRAME_QUEUE), axis=0)
    x = torch.from_numpy(stacked_obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        action, _, _, logits = _MODEL(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    best = int(np.argmax(probs))

    _LAST_ACTION = best
    # return ACTIONS[best]
    return ACTIONS[action]
