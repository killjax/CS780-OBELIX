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

_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05
_FRAME_QUEUE = deque([], maxlen=4)


def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights_D3qn_PER_pbrs_frame.pth")
    # wpath = os.path.join(submission_dir, "weights_phase1.pth")

    class duelingNetwork(nn.Module):
        def __init__(self, inDim=72, outDim=5, hDim=[64, 64], activation=F.relu):
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
            for i in range(len(self.ffn)):
                x = self.activation(self.ffn[i](x))
            V = self.value_network(x)
            A = self.advantage_network(x)
            Q = V + (A - A.mean(dim=1, keepdim=True))
            return Q

    model = duelingNetwork(inDim=72, outDim=5, hDim=[64, 64])

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
        q = _MODEL(x).squeeze(0).numpy()

    best = int(np.argmax(q))

    # Smoothing: if top-2 Qs are close, avoid flip-flopping
    if _LAST_ACTION is not None:
        order = np.argsort(-q)
        best_q, second_q = float(q[order[0]]), float(q[order[1]])

        # Ensure _LAST_ACTION is competitive before repeating it
        if (best_q - second_q) < _CLOSE_Q_DELTA and _LAST_ACTION in (
            order[0],
            order[1],
        ):
            if _REPEAT_COUNT < _MAX_REPEAT:
                best = _LAST_ACTION
                _REPEAT_COUNT += 1
            else:
                _REPEAT_COUNT = 0
        else:
            _REPEAT_COUNT = 0
    else:
        _REPEAT_COUNT = 0

    _LAST_ACTION = best
    return ACTIONS[best]
