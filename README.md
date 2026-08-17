# 🚧 TollMind-RL

### Intelligent Toll Booth Management using Reinforcement Learning
*MCA Mini Project*

TollMind-RL is a working, interactive simulation of an RL agent that
decides how many toll booths to keep open at a toll plaza, based on
real-time traffic conditions. It is built with tabular **Q-learning**
implemented from scratch (no external RL library), a Streamlit dashboard,
and a fully parameterized traffic simulation — every result on screen is
computed live from the current settings.

---

## 1. Problem Statement

Toll plazas experience fluctuating traffic. Opening too few booths causes
long queues and congestion; opening too many wastes operating cost when
traffic is light. This project trains an RL agent to dynamically decide
how many booths to open, minute by minute, to balance throughput, waiting
time, and operating cost.

## 2. Objectives

- Simulate a toll plaza with stochastic vehicle arrivals.
- Model the problem as a Markov Decision Process (MDP).
- Implement tabular Q-learning from first principles.
- Provide an interactive dashboard to train, run, and inspect the agent.
- Compare the learned policy against simple baselines (random, fixed-rule).

## 3. MDP Formulation

### State
`(queue level, recent arrival level, previously active booths)`

- **Queue level** — Low / Medium / High / Critical, derived from the
  queue length relative to total plaza capacity (`max_booths ×
  capacity_per_booth`):
  - Low: queue ≤ 1× total capacity
  - Medium: queue ≤ 2× total capacity
  - High: queue ≤ 4× total capacity
  - Critical: above that
- **Arrival level** — Low / Medium / High, derived from last minute's
  arrivals relative to total capacity.
- **Previously active booths** — 1 .. max_booths.

### Action
Open `1, 2, 3, ... max_booths` booths this minute. The number of actions
automatically adapts if the user changes "Maximum Booths" in the sidebar.

### Reward
```
reward = throughput_weight * processed
       - waiting_weight    * queue
       - booth_cost_weight * active_booths
       - congestion_penalty   (only if queue level is Critical)
```
All four weights are configurable from the UI, and the current formula is
shown live in the sidebar.

## 4. Q-Learning

Update rule (implemented explicitly in `q_learning.py`, not hidden inside
a library):

```
Q(s,a) <- Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

- **α (alpha)** — learning rate
- **γ (gamma)** — discount factor
- **ε (epsilon)** — exploration rate, decayed every episode toward a
  minimum value

### Exploration vs Exploitation
Epsilon-greedy action selection: with probability ε the agent picks a
random action (explore); otherwise it picks `argmax Q(s, ·)` (exploit).

## 5. Architecture

```
TollMind-RL/
│
├── app.py            # Streamlit dashboard (UI + orchestration)
├── environment.py     # TollEnvironment: state/action/reward logic
├── q_learning.py       # QLearningAgent: epsilon-greedy + Bellman update
├── simulation.py        # Training loop, simulation runs, baselines, comparison
├── config.py             # Default configuration values
├── requirements.txt
├── README.md
└── results/               # Exported CSVs land here
```

Data flow:
```
Vehicles arrive (Poisson) -> Environment -> State -> Agent -> Action
-> Booths opened -> Vehicles processed -> Reward -> Q-table update
-> Next state -> ... (repeat)
```

## 6. Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 7. Running

```bash
streamlit run app.py
```

Then in the browser:
1. Adjust settings in the sidebar (simulation, RL, reward weights).
2. Click **🚀 Train** to train the Q-learning agent from scratch.
3. Click **▶ Run Sim** to run one full greedy (exploitation-only) episode.
4. Explore the tabs: Dashboard, Step-by-Step, Q-Table & Policy, Graphs,
   Comparison, Real World, Viva Mode, Export.

If any sidebar value is changed after training, a warning banner appears
prompting you to retrain — old results are never silently reused.

## 8. Experiments

The **Comparison** tab runs three policies under identical simulation
settings and reports average queue, waiting, throughput, total reward,
and booth utilization:

- **Random Policy** — opens a random number of booths each minute.
- **Fixed Policy** — a traditional rule-based controller
  (`if queue > threshold: open N booths`), the classic non-learning
  baseline.
- **Q-Learning** — the trained agent, acting greedily.

All values are computed live; no numbers are pre-written.

## 9. Real-World Application

| Real World | This Project |
|---|---|
| Vehicle sensors | Simulated Poisson arrivals |
| FASTag reader | Payment / service time |
| Camera / loop detector | Queue measurement |
| Toll booth status | Environment state |
| RL controller | Decision-making agent |
| Open/close booth | RL action |
| Waiting time | Reward penalty |
| Vehicles processed | Positive reward |
| Traffic congestion | Environment feedback |

A production system could ingest data from FASTag systems, RFID readers,
CCTV/computer vision, inductive loop sensors, toll transaction systems,
and IoT traffic sensors, and use the RL controller to recommend (or, with
safety constraints, automatically control) lane/booth allocation.

**This project is a simulation only. It is not connected to real toll
infrastructure.**

## 10. Advantages

- Adapts to changing traffic instead of relying on fixed thresholds.
- Learns a policy purely from experience (no need to manually tune rules).
- Every parameter is transparent and explainable in a viva.

## 11. Limitations

- Arrival data is simulated, not from real sensors.
- State space is discretized/simplified for tabular Q-learning.
- No safety layer for real-world automatic control.
- Assumes vehicles are processed independently per booth with no lane
  -changing or routing behaviour.

## 12. Future Scope

DQN, SARSA, Monte Carlo methods, REINFORCE (policy gradient), multi-agent
RL, real FASTag data, computer vision for queue detection, IoT sensors,
real-time traffic prediction, cloud deployment, adaptive lane routing.
