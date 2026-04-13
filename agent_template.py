import os
from typing import Sequence
import numpy as np

# All possible actions
ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_LAST_ACTION = None
_REPEAT_COUNT = 0
Q_TABLE = None

_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05


def _load_once():
    global Q_TABLE
    if Q_TABLE is not None:
        return
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "q_table1.npy")

    Q_TABLE = np.load(file_path)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _LAST_ACTION, _REPEAT_COUNT

    _load_once()

    base3_val = 0
    for i in range(8):
        far_bit = obs[2 * i]
        near_bit = obs[2 * i + 1]
        val = 2 if near_bit else (1 if far_bit else 0)
        base3_val += val * (3**i)

    ir_bit = obs[16]
    stuck_bit = obs[17]
    s = int(base3_val + (ir_bit * 6561) + (stuck_bit * 13122))

    q = Q_TABLE[s]

    best = int(np.argmax(q))

    # Smoothing: if top-2 Qs are close, avoid flip-flopping
    if _LAST_ACTION is not None:
        order = np.argsort(-q)
        best_q, second_q = float(q[order[0]]), float(q[order[1]])
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
