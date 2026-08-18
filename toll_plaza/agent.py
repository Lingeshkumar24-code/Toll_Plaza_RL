"""Tabular reinforcement learning agents for the toll plaza environment."""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_states, n_actions))

    def act(self, state_idx: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[state_idx]))

    def update(self, s: int, a: int, r: float, ns: int, done: bool) -> None:
        target = r if done else r + self.gamma * float(np.max(self.q[ns]))
        self.q[s, a] += self.alpha * (target - self.q[s, a])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        np.save(path, self.q)

    def load(self, path: str) -> None:
        self.q = np.load(path)


class SarsaAgent(QLearningAgent):
    def update(self, s: int, a: int, r: float, ns: int, na: int, done: bool) -> None:
        target = r if done else r + self.gamma * float(self.q[ns, na])
        self.q[s, a] += self.alpha * (target - self.q[s, a])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)