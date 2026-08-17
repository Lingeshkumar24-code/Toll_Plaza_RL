"""
q_learning.py
-------------
A plain, from-scratch tabular Q-learning agent.

Deliberately NOT hidden behind an RL library (stable-baselines, gym
wrappers, etc.) so that the update rule is visible and explainable in a
viva:

    Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
"""

import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha, gamma,
                 epsilon_start, epsilon_decay, epsilon_min, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

        # The Q-table: one row per state, one column per action.
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state, greedy=False):
        """Epsilon-greedy action selection.
        greedy=True forces pure exploitation (used once training is done)."""
        if (not greedy) and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, self.n_actions))     # EXPLORE
        return int(np.argmax(self.Q[state]))                      # EXPLOIT

    def update(self, state, action, reward, next_state):
        best_next_value = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
