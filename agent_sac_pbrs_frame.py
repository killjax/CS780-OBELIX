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
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    # wpath = os.path.join(submission_dir, "weights_D3qn_PER_pbrs_frame.pth")
    wpath = os.path.join(submission_dir, "weights_phase1_sac.pth")

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

            probs = F.softmax(logits, dim=-1)

            z = probs == 0.0
            z = z.float() * 1e-8
            log_probs = torch.log(probs + z)

            return probs, log_probs

        def get_action(self, state, deterministic=False):
            probs, _ = self.forward(state)
            probs = probs[0]
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            return action

    model = PolicyNetwork(
        stateDim=72, actionDim=5, hiddenDims=[64, 64], activation=F.relu
    )

    # Load weights safely
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd, strict=True)
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action with smoothing."""
    global _LAST_ACTION, _REPEAT_COUNT, _FRAME_QUEUE
    _load_once()

    # MANAGE THE STACK
    if len(_FRAME_QUEUE) == 0:
        for _ in range(4):
            _FRAME_QUEUE.append(obs)
    else:
        _FRAME_QUEUE.append(obs)
    stacked_obs = np.concatenate(list(_FRAME_QUEUE), axis=0)
    x = torch.from_numpy(stacked_obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        action_idx = _MODEL.get_action(x, deterministic=True)

    return ACTIONS[action_idx]
