"""
config.py
---------
Default configuration values for TollMind-RL.

Every value here is only a STARTING point for the Streamlit sidebar widgets.
Nothing in this project is hardcoded into the simulation results - changing
any of these numbers in the UI changes the environment, the Q-learning
agent, and every metric / graph that is derived from them.
"""

DEFAULT_CONFIG = {
    # --- Simulation settings ---
    "sim_duration": 120,          # minutes (time steps per episode)
    "max_booths": 4,
    "capacity_per_booth": 4,      # vehicles/minute a single open booth can clear
    "arrival_rate": 8,            # average vehicles/minute (Poisson lambda)
    "initial_queue": 5,
    "random_seed": 42,

    # --- RL settings ---
    "episodes": 300,
    "alpha": 0.10,                # learning rate (alpha)
    "gamma": 0.90,                # discount factor (gamma)
    "epsilon_start": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,

    # --- Reward weights ---
    "throughput_weight": 1.0,
    "waiting_weight": 0.5,
    "booth_cost_weight": 0.3,
    "congestion_penalty": 10.0,
}

QUEUE_LEVELS = ["Low", "Medium", "High", "Critical"]
ARRIVAL_LEVELS = ["Low", "Medium", "High"]

# Keys that, if changed, invalidate a previously trained agent
TRAINING_SENSITIVE_KEYS = [
    "sim_duration", "max_booths", "capacity_per_booth", "arrival_rate",
    "initial_queue", "random_seed", "episodes", "alpha", "gamma",
    "epsilon_start", "epsilon_decay", "epsilon_min",
    "throughput_weight", "waiting_weight", "booth_cost_weight",
    "congestion_penalty",
]
