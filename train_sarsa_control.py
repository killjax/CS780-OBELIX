# all imports go in here
from __future__ import annotations
import numpy as np
import argparse, random
import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


def decay_step_size(
    initial_value, final_value, episode, max_episode, decay_type, decay_stop=None
):
    """
    Computes the step size (learning rate) for a given episode based on
    a specified decay strategy.

    Args:
        initial_value (float):
            Initial step size at the beginning of training.

        final_value (float):
            Final step size to which the learning rate decays.

        episode (int):
            Current episode index for which the step size is computed.

        max_episode (int):
            Total number of episodes over which decay is applied.

        decay_type (str):
            Type of decay strategy used for step size scheduling.
            Supported values include 'exponential' and 'linear'.

    Returns:
        float:
            Step size corresponding to the given episode based on the
            chosen decay strategy.
    """
    if max_episode == 1:
        return final_value
    if decay_type == "exponential":
        decay_rate = (final_value / initial_value) ** (1 / (max_episode - 1))
        step_size = initial_value * (decay_rate**episode)
    else:
        step_size = (
            initial_value
            + ((final_value - initial_value) / (max_episode - 1)) * episode
        )

    return max(step_size, final_value)


def decay_epsilon(
    initial_value, final_value, episode, max_episode, decay_type, decay_stop=None
):
    """
    Computes the exploration rate (epsilon) for a given episode based on
    a specified decay strategy.

    Args:
        initial_value (float):
            Initial value of epsilon at the beginning of training.

        final_value (float):
            Minimum value of epsilon after decay.

        episode (int):
            Current episode index for which epsilon is computed.

        max_episode (int):
            Total number of episodes over which epsilon decay is applied.

        decay_type (str):
            Type of decay strategy used for epsilon scheduling.
            Supported values include 'exponential' and 'linear'.

    Returns:
        float:
            Epsilon value corresponding to the given episode based on
            the chosen decay strategy.
    """
    if max_episode == 1:
        return final_value
    if decay_type == "exponential":
        decay_rate = (final_value / initial_value) ** (1 / (max_episode - 1))
        epsilon = initial_value * (decay_rate**episode)
    else:
        epsilon = (
            initial_value
            + ((final_value - initial_value) / (max_episode - 1)) * episode
        )

    return max(epsilon, final_value)


def encode_state(obs):
    base3_val = 0
    # Process the 8 near/far pairs (indices 0 to 15)
    for i in range(8):
        far_bit = obs[2 * i]
        near_bit = obs[2 * i + 1]
        # 0: clear, 1: far, 2: near
        val = 2 if near_bit else (1 if far_bit else 0)
        base3_val += val * (3**i)

    # Incorporate IR (index 16) and Stuck (index 17) as binary multipliers
    ir_bit = obs[16]
    stuck_bit = obs[17]

    # Shift the base3 value by the binary states
    # 3**8 = 6561. Multiply by 2 for IR, and 2 for Stuck.
    state_idx = base3_val + (ir_bit * 6561) + (stuck_bit * 13122)
    return int(state_idx)


def sarsa_control(environment, config=None):
    """
    Implements the SARSA (State–Action–Reward–State–Action) algorithm for a
    discrete Random Maze Environment (RME).

    This function applies the on-policy Temporal Difference (TD) control method
    SARSA to learn the optimal action-value function Q(s, a). An epsilon-greedy
    policy is used for both action selection and policy evaluation, with optional
    decay of the learning rate (alpha) and exploration rate (epsilon) across
    episodes.

    Args:
        environment (gym.Env): Environment with discrete observation and action spaces.
        config (dict): Configuration dictionary containing:
            - max_episodes (int): Number of training episodes.
            - discount_factor (float): Discount factor γ.
            - step_size (dict): Learning rate (alpha) decay parameters.
            - epsilon (dict): Epsilon-greedy exploration decay parameters.
            - seed (int): Random seed for environment reset.

    Returns:
        Q (np.ndarray): Final learned action-value function of shape (S, A)
    """

    if config is None:
        config = {}
    seed = config.get("seed", 20)
    max_episodes = config.get("max_episodes", 500)
    discount_factor = config.get("discount_factor", 0.99)

    step_size = config.get("step_size", {})
    initial_value_step_size = step_size.get("initial_value", 1.0)
    final_value_step_size = step_size.get("final_value", 0.01)
    decay_type_step_size = step_size.get("decay_type", "exponential")
    decay_stop_step_size = step_size.get("decay_stop", None)

    epsilon = config.get("epsilon", {})
    initial_value_epsilon = epsilon.get("initial_value", 1.0)
    final_value_epsilon = epsilon.get("final_value", 0.01)
    decay_type_epsilon = epsilon.get("decay_type", "exponential")
    decay_stop_epsilon = epsilon.get("decay_stop", None)

    total_states = config.get("environment", {}).get("total_states", 12)
    total_actions = config.get("environment", {}).get("total_actions", 4)

    Q = np.zeros((total_states, total_actions))
    td_target = 0.0
    observation = environment.reset(seed=seed)

    np.random.seed(seed)
    random.seed(seed)
    for e in tqdm.tqdm(range(max_episodes), desc="Training Episodes"):
        epl = decay_epsilon(
            initial_value_epsilon,
            final_value_epsilon,
            e,
            max_episodes,
            decay_type_epsilon,
            decay_stop=decay_stop_epsilon,
        )
        alpha = decay_step_size(
            initial_value_step_size,
            final_value_step_size,
            e,
            max_episodes,
            decay_type_step_size,
            decay_stop=decay_stop_step_size,
        )
        if e > 0:
            observation = environment.reset()
        s = encode_state(observation)
        done = False
        rand_val = np.random.random()
        if rand_val > epl:
            a = np.argmax(Q[s])
        else:
            a = np.random.randint(0, len(Q[s]))

        while not done:
            observation, r, done = environment.step(ACTIONS[a], render=False)
            n_s = encode_state(observation)
            rand_val = np.random.random()
            if rand_val > epl:
                n_a = np.argmax(Q[n_s])
            else:
                n_a = np.random.randint(0, len(Q[n_s]))
            td_target = r
            if not done:
                td_target += discount_factor * Q[n_s][n_a]
            td_error = td_target - Q[s][a]
            Q[s][a] += alpha * td_error
            s = n_s
            a = n_a

    return Q


def import_obelix(obelix_py: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="q_table.npy")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.01)
    ap.add_argument("--eps_decay_type", type=str, default="exponential")
    ap.add_argument("--steps_start", type=float, default=1.0)
    ap.add_argument("--steps_end", type=float, default=0.01)
    ap.add_argument("--steps_decay_type", type=str, default="exponential")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy_name", type=str, default="epsilon_greedy")
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    CONFIG = {
        "seed": args.seed,
        "max_episodes": args.episodes,
        "discount_factor": args.gamma,
        "policy_name": args.policy_name,
        "policy_action": None,
        "max_steps": args.max_steps,
        "environment": {"total_states": 3**8 * 4, "total_actions": 5},
        "step_size": {
            "initial_value": args.steps_start,
            "final_value": args.steps_end,
            "decay_type": args.steps_decay_type,
        },
        "epsilon": {
            "initial_value": args.eps_start,
            "final_value": args.eps_end,
            "decay_type": args.eps_decay_type,
        },
    }
    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )
    Q_TD = sarsa_control(env, config=CONFIG)

    with open(args.out, "wb") as f:
        np.save(f, Q_TD)
    print(f"Q-table saved to {args.out}")


if __name__ == "__main__":
    main()
