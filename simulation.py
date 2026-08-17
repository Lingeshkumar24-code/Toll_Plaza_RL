"""
simulation.py
-------------
Everything that "runs" the environment + agent:
  - train_agent()          -> trains Q-learning over many episodes
  - run_simulation()       -> runs one full episode with a trained agent
                               (step-by-step log for graphs / plaza view)
  - run_policy_simulation()-> runs one episode under a baseline policy
                               (random or fixed-rule), for comparison
  - compare_policies()     -> runs all three and returns a summary table

Nothing here is pre-computed. Every call actually steps the environment.
"""

import numpy as np
import pandas as pd

from environment import TollEnvironment
from q_learning import QLearningAgent


def build_env(cfg):
    return TollEnvironment(
        max_booths=cfg["max_booths"],
        capacity_per_booth=cfg["capacity_per_booth"],
        arrival_rate=cfg["arrival_rate"],
        initial_queue=cfg["initial_queue"],
        throughput_weight=cfg["throughput_weight"],
        waiting_weight=cfg["waiting_weight"],
        booth_cost_weight=cfg["booth_cost_weight"],
        congestion_penalty=cfg["congestion_penalty"],
        seed=cfg["random_seed"],
    )


def train_agent(cfg, progress_callback=None):
    """Train a fresh Q-learning agent from scratch using cfg. Returns
    (env, agent, training_dataframe)."""
    env = build_env(cfg)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_decay=cfg["epsilon_decay"],
        epsilon_min=cfg["epsilon_min"],
        seed=cfg["random_seed"],
    )

    episode_rewards = []
    epsilon_history = []

    for ep in range(cfg["episodes"]):
        state = env.reset()
        total_reward = 0.0
        for _ in range(cfg["sim_duration"]):
            action = agent.choose_action(state)
            next_state, reward, info = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        epsilon_history.append(agent.epsilon)

        if progress_callback is not None:
            progress_callback(ep + 1, cfg["episodes"], agent.epsilon, total_reward)

    training_df = pd.DataFrame({
        "episode": np.arange(1, cfg["episodes"] + 1),
        "reward": episode_rewards,
        "epsilon": epsilon_history,
    })
    window = max(1, cfg["episodes"] // 20)
    training_df["moving_avg_reward"] = (
        training_df["reward"].rolling(window=window, min_periods=1).mean()
    )

    return env, agent, training_df


def run_simulation(cfg, agent, greedy=True):
    """Run one full episode with the trained agent (fresh environment,
    same seed so results are reproducible for a given configuration)."""
    env = build_env(cfg)
    state = env.reset()
    rows = []
    for t in range(cfg["sim_duration"]):
        action = agent.choose_action(state, greedy=greedy)
        next_state, reward, info = env.step(action)
        rows.append({
            "step": t + 1,
            "state": state,
            "action": action,
            "booths_open": info["active_booths"],
            "arrivals": info["arrivals"],
            "processed": info["processed"],
            "queue": info["queue"],
            "reward": info["reward"],
            "queue_level": info["queue_level"],
        })
        state = next_state
    return env, pd.DataFrame(rows)


def random_policy_action(env, rng):
    return int(rng.integers(0, env.n_actions))


def fixed_policy_action(env):
    """A traditional, non-learning rule-based controller:
        IF queue is Low      -> open 1 booth
        IF queue is Medium   -> open 2 booths
        IF queue is High     -> open 3 booths
        IF queue is Critical -> open all booths
    Used only as a baseline to contrast against learned Q-learning
    behaviour - this policy never updates itself."""
    ql = env.queue_level()
    mapping = {
        0: 0,
        1: min(1, env.n_actions - 1),
        2: min(2, env.n_actions - 1),
        3: env.n_actions - 1,
    }
    return mapping[ql]


def run_policy_simulation(cfg, policy):
    env = build_env(cfg)
    rng = np.random.default_rng(cfg["random_seed"])
    state = env.reset()
    rows = []
    for t in range(cfg["sim_duration"]):
        if policy == "random":
            action = random_policy_action(env, rng)
        elif policy == "fixed":
            action = fixed_policy_action(env)
        else:
            raise ValueError("unknown policy: " + str(policy))
        next_state, reward, info = env.step(action)
        row = dict(info)
        row["step"] = t + 1
        row["action"] = action
        rows.append(row)
        state = next_state
    return pd.DataFrame(rows)


def compare_policies(cfg, agent):
    """Runs Q-Learning (greedy, trained agent), Random policy, and the
    Fixed rule-based policy, all on identical simulation settings, and
    returns a summary comparison table plus the raw per-step logs."""
    _, rl_df = run_simulation(cfg, agent, greedy=True)
    random_df = run_policy_simulation(cfg, "random")
    fixed_df = run_policy_simulation(cfg, "fixed")

    def summarize(df, booth_col):
        return {
            "Average Queue": df["queue"].mean(),
            "Average Waiting (veh)": df["queue"].mean(),
            "Vehicles Processed": df["processed"].sum(),
            "Total Reward": df["reward"].sum(),
            "Avg Booths Used": df[booth_col].mean(),
        }

    summary = pd.DataFrame({
        "Q-Learning": summarize(rl_df, "booths_open"),
        "Random Policy": summarize(random_df, "active_booths"),
        "Fixed Policy": summarize(fixed_df, "active_booths"),
    }).T

    return summary, rl_df, random_df, fixed_df
