"""
environment.py
---------------
The Toll Plaza environment (the "world" the RL agent interacts with).

This is a genuine Markov Decision Process:
    STATE  = (queue level, recent arrival level, previously active booths)
    ACTION = number of booths to open this minute (1 .. max_booths)
    REWARD = throughput reward - waiting penalty - operating cost
             (- congestion penalty if the queue is Critical)

Every number produced by this class comes from an actual calculation using
the current configuration. Nothing is pre-written or faked.
"""

import numpy as np


class TollEnvironment:
    def __init__(self, max_booths, capacity_per_booth, arrival_rate,
                 initial_queue, throughput_weight, waiting_weight,
                 booth_cost_weight, congestion_penalty, seed=None):

        if max_booths < 1:
            raise ValueError("max_booths must be at least 1")
        if capacity_per_booth <= 0:
            raise ValueError("capacity_per_booth must be positive")
        if arrival_rate < 0:
            raise ValueError("arrival_rate cannot be negative")
        if initial_queue < 0:
            raise ValueError("initial_queue cannot be negative")

        self.max_booths = int(max_booths)
        self.capacity_per_booth = float(capacity_per_booth)
        self.arrival_rate = float(arrival_rate)
        self.initial_queue = int(initial_queue)

        self.throughput_weight = float(throughput_weight)
        self.waiting_weight = float(waiting_weight)
        self.booth_cost_weight = float(booth_cost_weight)
        self.congestion_penalty = float(congestion_penalty)

        self.rng = np.random.default_rng(seed)

        # ---- Queue-level thresholds are DERIVED from the configured
        # capacity, not hardcoded. Total capacity = max_booths x per-booth
        # capacity. A queue is compared against multiples of this total
        # capacity to decide whether it is Low / Medium / High / Critical.
        self.total_capacity = self.max_booths * self.capacity_per_booth
        self.q_low_max = self.total_capacity          # <= 1x total capacity
        self.q_medium_max = self.total_capacity * 2    # <= 2x total capacity
        self.q_high_max = self.total_capacity * 4      # <= 4x total capacity
        # anything above q_high_max is Critical

        self.n_queue_levels = 4
        self.n_arrival_levels = 3
        self.n_booth_states = self.max_booths          # previous active booths: 1..max_booths
        self.n_actions = self.max_booths                # open 1..max_booths booths
        self.n_states = self.n_queue_levels * self.n_arrival_levels * self.n_booth_states

        self.queue = None
        self.active_booths = None
        self.last_arrivals = None
        self.t = None
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.queue = self.initial_queue
        self.active_booths = 1
        self.last_arrivals = 0
        self.t = 0
        return self.get_state()

    # ------------------------------------------------------------------
    def queue_level(self, queue=None):
        q = self.queue if queue is None else queue
        if q <= self.q_low_max:
            return 0
        if q <= self.q_medium_max:
            return 1
        if q <= self.q_high_max:
            return 2
        return 3

    def arrival_level(self, arrivals=None):
        a = self.last_arrivals if arrivals is None else arrivals
        if a <= self.total_capacity * 0.5:
            return 0
        if a <= self.total_capacity:
            return 1
        return 2

    # ------------------------------------------------------------------
    def get_state(self):
        ql = self.queue_level()
        al = self.arrival_level()
        b = self.active_booths - 1
        return (ql * self.n_arrival_levels + al) * self.n_booth_states + b

    def state_components(self, state_idx):
        """Inverse of get_state: returns (queue_level, arrival_level, active_booths)."""
        b = state_idx % self.n_booth_states
        rest = state_idx // self.n_booth_states
        al = rest % self.n_arrival_levels
        ql = rest // self.n_arrival_levels
        return ql, al, b + 1

    # ------------------------------------------------------------------
    def step(self, action):
        """
        action: integer in [0, n_actions - 1]
                action + 1 = number of booths opened this step
        """
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action must be in [0, {self.n_actions - 1}]")

        booths_open = action + 1

        arrivals = int(self.rng.poisson(self.arrival_rate)) if self.arrival_rate > 0 else 0
        self.queue += arrivals
        self.last_arrivals = arrivals

        capacity = booths_open * self.capacity_per_booth
        processed = min(self.queue, capacity)
        self.queue = max(0, self.queue - processed)
        self.active_booths = booths_open

        reward = (
            self.throughput_weight * processed
            - self.waiting_weight * self.queue
            - self.booth_cost_weight * booths_open
        )

        is_critical = self.queue_level() == 3
        if is_critical:
            reward -= self.congestion_penalty

        self.t += 1
        next_state = self.get_state()

        info = {
            "arrivals": arrivals,
            "processed": processed,
            "queue": self.queue,
            "active_booths": booths_open,
            "reward": reward,
            "queue_level": self.queue_level(),
            "is_critical": is_critical,
        }
        return next_state, reward, info
