import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_FRAME_QUEUE = deque([], maxlen=4)
_LAST_ACTION = "FW"


def _load_once():
    """Load the trained PPO LSTM model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights_ppo_lstm_phase2a.pth")

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

        def forward(self, state):
            if not isinstance(state, torch.Tensor):
                s = torch.tensor(
                    state, dtype=torch.float32, device=next(self.parameters()).device
                )
            else:
                s = state

            if s.dim() == 2:
                s = s.unsqueeze(0)

            lstm_out, _ = self.lstm(s)
            l = lstm_out[:, -1, :]

            for hLayer in self.hLayers:
                l = self.activation(hLayer(l))

            logits = self.out(l)
            probs = F.softmax(logits, dim=-1)
            return probs

    model = PolicyNetwork(
        stateDim=18, actionDim=5, hiddenDims=[64, 64], activation=F.relu
    )

    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd, strict=True)
    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _FRAME_QUEUE, _LAST_ACTION
    _load_once()

    obs_fixed = obs.copy()

    if _LAST_ACTION != "FW":
        obs_fixed[17] = 0

    if len(_FRAME_QUEUE) == 0:
        for _ in range(4):
            _FRAME_QUEUE.append(obs_fixed)
    else:
        _FRAME_QUEUE.append(obs_fixed)

    stacked_obs = np.stack(list(_FRAME_QUEUE), axis=0)
    x = torch.from_numpy(stacked_obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        probs = _MODEL(x)
        probs = probs[0].numpy()

        if obs_fixed[17] == 1:
            probs[2] = 0.0

        action_idx = int(np.argmax(probs))

    _LAST_ACTION = ACTIONS[action_idx]
    return _LAST_ACTION
