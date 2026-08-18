# Toll Plaza Reinforcement Learning

A reinforcement learning project where an agent learns to dispatch vehicles to
lanes at a toll plaza to minimize waiting time. Includes a discrete-time
queueing simulation, tabular Q-learning and SARSA agents, training with
learning-curve plots, and a live animated simulation.

## Features

- **Simulation** (`toll_plaza/toll_env.py`): tick-based queueing environment
  - 4 vehicle types (car, motorcycle, truck, bus) with different service times and toll fees
  - Fast **ETC lanes** that serve vehicles ~35% quicker than manual toll booths
  - Poisson arrivals; each arriving vehicle is dispatched to one lane by the agent
  - State: capped queue length per lane; Action: lane choice; Reward: `-total queue length`
- **Agents** (`toll_plaza/agent.py`): tabular Q-learning and SARSA with
  epsilon-greedy exploration and decaying epsilon
- **Training** (`toll_plaza/train.py`): episode loop, CSV log, Q-table export,
  learning-curve plots (reward + average wait)
- **Visualization** (`toll_plaza/visualize.py`): live matplotlib animation of
  vehicles queueing, being served at booths, and exiting

## Install

```bash
pip install -r requirements.txt
```

## Usage

Train a Q-learning agent for 300 episodes:

```bash
python -m toll_plaza.train --episodes 300 --lanes 4 --etc 1 --arrival-rate 0.3
```

Train SARSA instead:

```bash
python -m toll_plaza.train --agent sarsa --episodes 300
```

Outputs land in `out/`:

- `out/models/q_table_<agent>_<lanes>lanes.npy` – trained policy
- `out/training_<agent>.csv` – per-episode metrics
- `out/plots/learning_curve.png` – reward and wait-time learning curves

Watch the trained policy live (Q-learning usually outperforms a random policy
after ~100 episodes):

```bash
python -m toll_plaza.visualize --q-table out/models/q_table_qlearning_4lanes.npy --arrival-rate 0.3
```

To save a GIF animation instead of opening a window (needs Pillow):

```bash
python -m toll_plaza.visualize --q-table out/models/q_table_qlearning_4lanes.npy --frames 200 --save out/simulation.gif
```

## How it works

**MDP formulation**

- **State**: quantized queue length per lane, capped at 7 (one-hot mixed-radix index)
- **Action**: which lane the next arriving vehicle is assigned to (0..N-1)
- **Reward**: `-Σ queue lengths` each tick, so the agent minimizes total waiting
- **Transition**: booth service times follow per-type uniform distributions,
  scaled down in ETC lanes

**Learning**: classic tabular Q-learning update
`Q(s,a) ← Q(s,a) + α(r + γ·max Q(s',a') − Q(s,a))` with ε-greedy exploration
and per-step ε decay. With 4 lanes the table has 8⁴·4 = 16,384 entries — fully
trainable in seconds on a laptop.

**What the agent learns**: avoid piling every vehicle into one lane; exploit
ETC lanes for trucks/buses only when queues there are short; keep lane queues
balanced, mirroring real "dynamic lane assignment" systems.

## Browser version

Open the simulation in your web browser (Q-learning runs live in JavaScript, with the Python-trained policy embedded):

```bash
python -m toll_plaza.make_web       # builds toll_plaza_sim.html and opens your browser
```

Controls on the page:

- **Start / Stop** — pause and resume the live simulation
- **Slow-Mo** — 5x slower simulation so you can watch vehicles decelerate into the booths
- **Live Q-learning** — keeps training in the browser (Q-values carry over from the embedded Python policy)
- **Trained policy** — greedy execution of the embedded 300-episode policy
- **Results** — show the last episode's results panel
- **Reset** — restart episode and learning curve
- **Arrival rate / Speed** sliders — demand and simulation speed

Vehicles glide into queues, decelerate as they approach, pass through toll booths
(animated red/amber gates raise, signal lights turn green while serving) and drive
off the exit. At the end of every episode a **results panel** pops up with served /
avg wait / max wait / revenue, a verdict, and a comparison against a random-dispatch
baseline (computed at load) — e.g. "25s &rarr; 12s, 52% less waiting".

## Key parameters

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--arrival-rate` | 0.3 | vehicles per second (Poisson); keep below plaza capacity (~0.4/s for 4 lanes, ~0.36/s all-manual) or queues grow forever |
| `--lanes` / `--etc` | 4 / 1 | total lanes, how many are ETC |
| `--episodes` | 300 | training episodes |
| `--episode-len` | 1500 | simulation ticks per episode |
| `--alpha` / `--gamma` / `--decay` | 0.1 / 0.95 / 0.999 | learning rate, discount, ε decay |
| `--agent` | qlearning | `qlearning` or `sarsa` |

## Ideas to extend

- Deep Q-Network with state features (mean queue, vehicle type, arrival rate)
- Multi-agent: one controller per booth, or pricing lane decisions
- Include toll revenue in the reward to study revenue-vs-delay trade-offs
- Dynamic booth staffing: agent also opens/closes booths
- Continuous arrival rates and non-stationary demand (rush hour)