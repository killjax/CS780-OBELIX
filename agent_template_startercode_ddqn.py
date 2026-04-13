"""Better DQN-style agent scaffold for OBELIX (CPU).

This agent is *evaluation-only*: it loads pretrained weights from a file
placed next to agent.py inside the submission zip (weights.pth).

Why your STD is huge:
- if the policy is stochastic (epsilon > 0) during evaluation, scores vary a lot.
Fix:
- greedy action selection (epsilon=0), model.eval(), torch.no_grad().
- optional action smoothing to reduce oscillation when Q-values are close.

Submission ZIP structure:
  submission.zip
    agent.py
    weights.pth
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_LAST_ACTION = None
_REPEAT_COUNT = 0

_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05


def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(18, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
            )

        def forward(self, x):
            return self.net(x)

    model = Net()

    # Load weights safely
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd, strict=True)
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action with smoothing."""
    global _LAST_ACTION, _REPEAT_COUNT
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q = _MODEL(x).squeeze(0).numpy()

    best = int(np.argmax(q))

    # Smoothing: if top-2 Qs are close, avoid flip-flopping
    if _LAST_ACTION is not None:
        order = np.argsort(-q)
        best_q, second_q = float(q[order[0]]), float(q[order[1]])
        if (best_q - second_q) < _CLOSE_Q_DELTA:
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
