import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_FRAME_QUEUE = deque([], maxlen=4)


def _load_once():
    """Load the trained SAC LSTM model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    # Update this to whatever you name your output weights
    wpath = os.path.join(submission_dir, "weights_sac_phase1a.pth")

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
            self.logAlpha = nn.Parameter(torch.zeros(1))

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
            l = lstm_out[:, -1, :]  # Extract final time step

            for hLayer in self.hLayers:
                l = self.activation(hLayer(l))

            logits = self.out(l)
            probs = F.softmax(logits, dim=-1)
            return probs

    # stateDim is 18 because the input is a sequence of 4 frames, each with 18 features
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
    """Use the trained SAC model with an emergency wall-collision reflex."""
    global _FRAME_QUEUE
    _load_once()

    # MANAGE THE STACK (Outputs a 2D matrix: 4 x 18)
    if len(_FRAME_QUEUE) == 0:
        for _ in range(4):
            _FRAME_QUEUE.append(obs)
    else:
        _FRAME_QUEUE.append(obs)

    stacked_obs = np.stack(list(_FRAME_QUEUE), axis=0)
    x = torch.from_numpy(stacked_obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        # Get raw probabilities directly from the forward pass
        probs = _MODEL(x)
        probs = probs[0].numpy()  # Convert to 1D numpy array

        # --- THE REFLEXIVE BUMPER (ACTION MASK) ---
        # obs[17] is the stuck_flag for the current frame.
        if obs[17] == 1:
            probs[2] = 0.0  # Force 'FW' probability to 0.0 to break argmax loops
        # ------------------------------------------

        action_idx = int(np.argmax(probs))

    return ACTIONS[action_idx]
