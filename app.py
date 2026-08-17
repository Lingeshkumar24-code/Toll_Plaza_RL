"""
app.py
------
TollMind-RL Streamlit dashboard.

Every number shown on this page is computed live from the environment /
agent objects held in st.session_state. Changing a sidebar control and
re-running the simulation (or retraining) will change every metric, graph,
Q-table value and policy shown here.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import DEFAULT_CONFIG, QUEUE_LEVELS, ARRIVAL_LEVELS, TRAINING_SENSITIVE_KEYS
from environment import TollEnvironment
from q_learning import QLearningAgent
import simulation as sim

st.set_page_config(page_title="TollMind-RL", page_icon="🚧", layout="wide")

# ----------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 0%, #101826 0%, #0b0f17 55%, #05070b 100%);
    color: #e6edf3;
}
.glass {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(6px);
}
.metric-card {
    background: linear-gradient(145deg, rgba(56,189,248,0.10), rgba(255,255,255,0.02));
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: #7dd3fc; }
.metric-label { font-size: 0.8rem; color: #94a3b8; letter-spacing: .04em; text-transform: uppercase;}
.booth-open {
    background: linear-gradient(145deg, rgba(34,197,94,0.18), rgba(255,255,255,0.02));
    border: 1px solid rgba(34,197,94,0.45);
    border-radius: 12px; padding: 0.7rem; text-align:center; margin-bottom: 0.4rem;
}
.booth-closed {
    background: linear-gradient(145deg, rgba(248,113,113,0.10), rgba(255,255,255,0.02));
    border: 1px solid rgba(248,113,113,0.35);
    border-radius: 12px; padding: 0.7rem; text-align:center; margin-bottom: 0.4rem; opacity: 0.7;
}
h1, h2, h3 { color: #e6edf3; }
.badge-warn {
    background: rgba(250,204,21,0.12); border: 1px solid rgba(250,204,21,0.45);
    color: #fde68a; padding: 0.5rem 0.8rem; border-radius: 10px; font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------------------------
def init_state():
    ss = st.session_state
    if "cfg" not in ss:
        ss.cfg = dict(DEFAULT_CONFIG)
    if "trained" not in ss:
        ss.trained = False
    if "env" not in ss:
        ss.env = None
    if "agent" not in ss:
        ss.agent = None
    if "training_df" not in ss:
        ss.training_df = None
    if "trained_cfg" not in ss:
        ss.trained_cfg = None
    if "sim_df" not in ss:
        ss.sim_df = None
    if "step_env" not in ss:
        ss.step_env = None
    if "step_log" not in ss:
        ss.step_log = []
    if "comparison" not in ss:
        ss.comparison = None

init_state()
ss = st.session_state

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Controls")

with st.sidebar.expander("🛣️ Simulation Settings", expanded=True):
    sim_duration = st.slider("Simulation Duration (minutes)", 30, 300, ss.cfg["sim_duration"], 10)
    max_booths = st.slider("Maximum Booths", 2, 8, ss.cfg["max_booths"], 1)
    capacity_per_booth = st.slider("Booth Capacity (vehicles/min/booth)", 1, 20, int(ss.cfg["capacity_per_booth"]), 1)
    arrival_rate = st.slider("Average Arrival Rate (vehicles/min)", 0, 60, int(ss.cfg["arrival_rate"]), 1)
    initial_queue = st.slider("Initial Queue Length", 0, 100, ss.cfg["initial_queue"], 1)
    random_seed = st.number_input("Random Seed", 0, 10_000, ss.cfg["random_seed"], 1)

with st.sidebar.expander("🧠 RL Settings", expanded=True):
    episodes = st.slider("Training Episodes", 50, 2000, ss.cfg["episodes"], 50)
    alpha = st.slider("Learning Rate (α)", 0.01, 1.0, float(ss.cfg["alpha"]), 0.01)
    gamma = st.slider("Discount Factor (γ)", 0.0, 0.999, float(ss.cfg["gamma"]), 0.01)
    epsilon_start = st.slider("Initial Epsilon", 0.0, 1.0, float(ss.cfg["epsilon_start"]), 0.05)
    epsilon_decay = st.slider("Epsilon Decay", 0.900, 0.9999, float(ss.cfg["epsilon_decay"]), 0.0005, format="%.4f")
    epsilon_min = st.slider("Minimum Epsilon", 0.0, 0.5, float(ss.cfg["epsilon_min"]), 0.01)

with st.sidebar.expander("🎯 Reward Settings", expanded=True):
    throughput_weight = st.slider("Throughput Weight", 0.0, 5.0, float(ss.cfg["throughput_weight"]), 0.1)
    waiting_weight = st.slider("Waiting Penalty Weight", 0.0, 5.0, float(ss.cfg["waiting_weight"]), 0.1)
    booth_cost_weight = st.slider("Booth Operating Cost Weight", 0.0, 5.0, float(ss.cfg["booth_cost_weight"]), 0.1)
    congestion_penalty = st.slider("Congestion Penalty (Critical queue)", 0.0, 50.0, float(ss.cfg["congestion_penalty"]), 1.0)

st.sidebar.markdown(
    f"""<div class="glass" style="font-size:0.82rem;">
    <b>Current reward formula</b><br>
    reward = {throughput_weight:g}·processed
    − {waiting_weight:g}·queue
    − {booth_cost_weight:g}·active_booths
    {"− " + str(congestion_penalty) + " (if queue is Critical)" if congestion_penalty else ""}
    </div>""", unsafe_allow_html=True)

new_cfg = {
    "sim_duration": sim_duration, "max_booths": max_booths,
    "capacity_per_booth": capacity_per_booth, "arrival_rate": arrival_rate,
    "initial_queue": initial_queue, "random_seed": int(random_seed),
    "episodes": episodes, "alpha": alpha, "gamma": gamma,
    "epsilon_start": epsilon_start, "epsilon_decay": epsilon_decay,
    "epsilon_min": epsilon_min, "throughput_weight": throughput_weight,
    "waiting_weight": waiting_weight, "booth_cost_weight": booth_cost_weight,
    "congestion_penalty": congestion_penalty,
}
ss.cfg = new_cfg

params_changed = (
    ss.trained_cfg is not None and
    any(ss.trained_cfg.get(k) != new_cfg.get(k) for k in TRAINING_SENSITIVE_KEYS)
)

col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
train_clicked = col_btn1.button("🚀 Train")
run_clicked = col_btn2.button("▶ Run Sim")
reset_clicked = col_btn3.button("🔄 Reset")

if reset_clicked:
    ss.trained = False
    ss.env = None
    ss.agent = None
    ss.training_df = None
    ss.trained_cfg = None
    ss.sim_df = None
    ss.step_env = None
    ss.step_log = []
    ss.comparison = None
    st.rerun()

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown("# 🚧 TOLLMIND-RL")
st.markdown("##### Adaptive Toll Plaza Optimization using Reinforcement Learning")

if params_changed:
    st.markdown('<div class="badge-warn">⚠ Parameters changed — retraining recommended. '
                'Old results below may not reflect the current configuration.</div>',
                unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TRAIN
# ----------------------------------------------------------------------
if train_clicked:
    progress_bar = st.progress(0, text="Starting training...")
    status_box = st.empty()

    def progress_cb(ep, total, eps, reward):
        progress_bar.progress(ep / total, text=f"Episode {ep}/{total}")
        if ep % max(1, total // 20) == 0 or ep == total:
            status_box.markdown(
                f"**Episode:** {ep}/{total} &nbsp; | &nbsp; "
                f"**Epsilon:** {eps:.3f} &nbsp; | &nbsp; "
                f"**Episode Reward:** {reward:.1f}"
            )

    env, agent, training_df = sim.train_agent(ss.cfg, progress_callback=progress_cb)
    ss.env = env
    ss.agent = agent
    ss.training_df = training_df
    ss.trained = True
    ss.trained_cfg = dict(ss.cfg)
    ss.sim_df = None
    ss.comparison = None
    progress_bar.progress(1.0, text="Training complete ✅")

if run_clicked:
    if not ss.trained:
        st.error("Train the agent first (click 🚀 Train), then run the simulation.")
    else:
        _, sim_df = sim.run_simulation(ss.cfg, ss.agent, greedy=True)
        ss.sim_df = sim_df

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_dash, tab_step, tab_q, tab_graphs, tab_compare, tab_real, tab_viva, tab_export = st.tabs([
    "📊 Dashboard", "⏭ Step-by-Step", "🧮 Q-Table & Policy", "📈 Graphs",
    "⚖️ Comparison", "🌍 Real World", "🎓 Viva Mode", "⬇️ Export"
])

# ================= DASHBOARD =================
with tab_dash:
    if ss.sim_df is None:
        st.info("Click **🚀 Train** in the sidebar, then **▶ Run Sim** to see live results.")
    else:
        df = ss.sim_df
        last = df.iloc[-1]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        metrics = [
            ("🚗 Current Vehicles", f"{int(last['queue'])}"),
            ("⏱ Avg Waiting (queue)", f"{df['queue'].mean():.1f}"),
            ("🚧 Active Booths", f"{int(last['booths_open'])}"),
            ("✅ Vehicles Processed", f"{int(df['processed'].sum())}"),
            ("📊 Average Queue", f"{df['queue'].mean():.1f}"),
            ("⭐ Total Reward", f"{df['reward'].sum():.1f}"),
        ]
        for col, (label, val) in zip([c1, c2, c3, c4, c5, c6], metrics):
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                         f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("### 🛣️ Live Toll Plaza (final state of last run)")
        n_booths = ss.cfg["max_booths"]
        active = int(last["booths_open"])
        booth_cols = st.columns(n_booths)
        per_booth_queue = int(last["queue"]) // max(1, active) if active else 0
        for i, col in enumerate(booth_cols):
            is_open = (i < active)
            css = "booth-open" if is_open else "booth-closed"
            status = "🟢 OPEN" if is_open else "🔴 CLOSED"
            q_show = per_booth_queue if is_open else 0
            col.markdown(f'<div class="{css}"><b>BOOTH {i+1}</b><br>{status}<br>Queue ≈ {q_show}</div>',
                         unsafe_allow_html=True)

        st.caption(f"Step {int(last['step'])} of {ss.cfg['sim_duration']} · "
                   f"Arrivals last minute: {int(last['arrivals'])} · "
                   f"Queue level: {QUEUE_LEVELS[int(last['queue_level'])]}")

        st.markdown("### Simulation Log (last 15 steps)")
        st.dataframe(df.tail(15), use_container_width=True)

# ================= STEP BY STEP =================
with tab_step:
    st.markdown("### ⏭ Step-by-Step RL Demonstration")
    st.caption("Use this to walk through exactly what the agent observes, decides, and learns — ideal for a viva.")

    if not ss.trained:
        st.info("Train the agent first in the sidebar.")
    else:
        colA, colB = st.columns(2)
        if colA.button("🔁 Reset Step Demo"):
            ss.step_env = sim.build_env(ss.cfg)
            ss.step_env.reset()
            ss.step_log = []

        if ss.step_env is None:
            ss.step_env = sim.build_env(ss.cfg)
            ss.step_env.reset()

        if colB.button("⏭ Next Step"):
            env = ss.step_env
            state_before = env.get_state()
            ql, al, b = env.state_components(state_before)
            action = ss.agent.choose_action(state_before, greedy=True)
            next_state, reward, info = env.step(action)
            td_error = ss.agent.update(state_before, action, reward, next_state)

            ss.step_log.append({
                "Step": env.t,
                "Prev State (Queue/Arrival/Booths)": f"{QUEUE_LEVELS[ql]}/{ARRIVAL_LEVELS[al]}/{b}",
                "Action (Booths Opened)": info["active_booths"],
                "Vehicles Arrived": info["arrivals"],
                "Vehicles Processed": int(info["processed"]),
                "Queue After": info["queue"],
                "Reward": round(info["reward"], 2),
                "Next Queue Level": QUEUE_LEVELS[info["queue_level"]],
                "Q-value Updated By": round(td_error, 3),
            })

        if ss.step_log:
            st.dataframe(pd.DataFrame(ss.step_log), use_container_width=True)
            last = ss.step_log[-1]
            st.markdown(f"""
            <div class="glass">
            <b>Current State:</b> {last['Prev State (Queue/Arrival/Booths)']} &nbsp;→&nbsp;
            <b>Action:</b> Open {last['Action (Booths Opened)']} booth(s) &nbsp;→&nbsp;
            <b>Reward:</b> {last['Reward']} &nbsp;→&nbsp;
            <b>Next Queue Level:</b> {last['Next Queue Level']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("Click **Next Step** to begin.")

# ================= Q-TABLE & POLICY =================
with tab_q:
    if not ss.trained:
        st.info("Train the agent first in the sidebar.")
    else:
        st.markdown("### 🧮 Trained Q-Table")
        st.caption("Rows = state (queue level / arrival level / previously active booths). "
                   "Columns = action (booths to open). Highest value per row is the learned best action.")

        env, agent = ss.env, ss.agent
        n_actions = env.n_actions
        action_cols = [f"Open {i+1}" for i in range(n_actions)]

        rows_idx = []
        rows_data = []
        for s in range(env.n_states):
            ql, al, b = env.state_components(s)
            rows_idx.append(f"{QUEUE_LEVELS[ql]} | Arr:{ARRIVAL_LEVELS[al]} | PrevBooths:{b}")
            rows_data.append(agent.Q[s])
        q_df = pd.DataFrame(rows_data, index=rows_idx, columns=action_cols)

        st.dataframe(
            q_df.style.highlight_max(axis=1, color="rgba(34,197,94,0.35)").format("{:.2f}"),
            use_container_width=True, height=420
        )

        st.markdown("### 🧭 Learned Policy (by Queue Level)")
        st.caption("For each queue level, this shows the action the trained agent prefers on average "
                   "across arrival levels and previous-booth states (argmax of mean Q-values).")

        policy_cols = st.columns(4)
        for ql in range(4):
            mask = [env.state_components(s)[0] == ql for s in range(env.n_states)]
            avg_q = agent.Q[mask].mean(axis=0)
            best_action = int(np.argmax(avg_q))
            policy_cols[ql].markdown(
                f'<div class="metric-card"><div class="metric-label">{QUEUE_LEVELS[ql]} QUEUE</div>'
                f'<div class="metric-value">Open {best_action + 1}</div></div>',
                unsafe_allow_html=True
            )

# ================= GRAPHS =================
with tab_graphs:
    if not ss.trained:
        st.info("Train the agent first in the sidebar.")
    else:
        tdf = ss.training_df

        st.markdown("### 📈 Learning Curve — Episode vs Reward")
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(tdf["episode"], tdf["reward"], color="#38bdf8", alpha=0.35, label="Episode Reward")
        ax1.plot(tdf["episode"], tdf["moving_avg_reward"], color="#7dd3fc", linewidth=2, label="Moving Avg")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward")
        ax1.legend(); ax1.grid(alpha=0.15)
        fig1.patch.set_alpha(0); ax1.set_facecolor("none")
        st.pyplot(fig1, use_container_width=True)

        st.markdown("### 🎲 Epsilon Decay — Episode vs Epsilon")
        st.caption("High epsilon → more exploration. Low epsilon → more exploitation of the learned policy.")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(tdf["episode"], tdf["epsilon"], color="#fbbf24")
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Epsilon")
        ax2.grid(alpha=0.15)
        fig2.patch.set_alpha(0); ax2.set_facecolor("none")
        st.pyplot(fig2, use_container_width=True)

        if ss.sim_df is not None:
            sdf = ss.sim_df
            st.markdown("### 🚦 Time vs Queue Length (last simulation run)")
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.plot(sdf["step"], sdf["queue"], color="#f87171")
            ax3.set_xlabel("Time step (minute)"); ax3.set_ylabel("Queue Length")
            ax3.grid(alpha=0.15)
            fig3.patch.set_alpha(0); ax3.set_facecolor("none")
            st.pyplot(fig3, use_container_width=True)

            st.markdown("### 🚗 Time vs Vehicles Processed (last simulation run)")
            fig4, ax4 = plt.subplots(figsize=(8, 3))
            ax4.plot(sdf["step"], sdf["processed"].cumsum(), color="#34d399")
            ax4.set_xlabel("Time step (minute)"); ax4.set_ylabel("Cumulative Vehicles Processed")
            ax4.grid(alpha=0.15)
            fig4.patch.set_alpha(0); ax4.set_facecolor("none")
            st.pyplot(fig4, use_container_width=True)
        else:
            st.info("Click **▶ Run Sim** in the sidebar to also see Queue and Throughput graphs.")

# ================= COMPARISON =================
with tab_compare:
    st.markdown("### ⚖️ Algorithm Comparison: Random vs Fixed-Rule vs Q-Learning")
    if not ss.trained:
        st.info("Train the agent first in the sidebar.")
    else:
        if st.button("Run Comparison Experiment"):
            summary, rl_df, random_df, fixed_df = sim.compare_policies(ss.cfg, ss.agent)
            ss.comparison = (summary, rl_df, random_df, fixed_df)

        if ss.comparison is not None:
            summary, rl_df, random_df, fixed_df = ss.comparison
            st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            summary["Average Queue"].plot(kind="bar", ax=ax, color=["#7dd3fc", "#f87171", "#fbbf24"])
            ax.set_ylabel("Average Queue")
            ax.set_title("Average Queue by Policy (lower is better)")
            fig.patch.set_alpha(0); ax.set_facecolor("none")
            st.pyplot(fig, use_container_width=True)

            best_policy = summary["Average Queue"].astype(float).idxmin()
            st.success(
                f"Under the current simulation configuration, **{best_policy}** achieved the "
                f"lowest average queue ({summary.loc[best_policy, 'Average Queue']:.2f}) "
                f"among the three policies tested."
            )

# ================= REAL WORLD =================
with tab_real:
    st.markdown("## 🌍 How This Works in the Real World")
    st.markdown("""
    | Real World | This Project |
    |---|---|
    | Vehicle sensors | Simulated Poisson arrivals |
    | FASTag reader | Payment / service time (booth capacity) |
    | Camera / loop detector | Queue length measurement |
    | Toll booth status | Environment state |
    | RL controller | Q-learning agent |
    | Open/close booth | RL action |
    | Waiting time | Reward penalty |
    | Vehicles processed | Positive reward |
    | Traffic congestion | Environment feedback / congestion penalty |
    """)
    st.markdown("""
    A real deployment could receive live data from **FASTag systems, RFID readers,
    CCTV / computer vision, inductive loop sensors, toll transaction systems, and IoT
    traffic sensors**. The RL controller could then recommend — or, with appropriate
    safety checks, automatically control — how many lanes/booths should be open.
    """)
    st.warning("⚠ This is a college simulation. It does **not** connect to real toll infrastructure.")

    st.markdown("### Example: Adaptive Decision-Making Over a Day")
    st.markdown("""
    ```
    Normal traffic        → 2 booths open
    Traffic increases     → queue increases → RL detects High Queue → 3 booths open
    Peak traffic          → Critical Queue → 4 booths open
    Traffic decreases     → RL learns keeping all booths open is unnecessary → 3 → 2 booths
    ```
    """)

    st.markdown("### 🤔 What Makes This RL (and not just an if/else system)?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Traditional rule-based system**")
        st.code("if queue > 10:\n    open 3 booths", language="python")
    with col2:
        st.markdown("**Reinforcement Learning system**")
        st.code("observe state\n-> try actions\n-> receive rewards\n"
                "-> update Q-values\n-> improve policy\n-> learn better decisions",
                language="text")

# ================= VIVA MODE =================
with tab_viva:
    st.markdown("## 🎓 Viva Mode — Quick Explanations")
    qa = [
        ("What is the Agent?", "The toll management controller that decides how many booths to open."),
        ("What is the Environment?", "The simulated toll plaza: vehicle arrivals, queue, and booth processing."),
        ("What is the State?", "The current queue level, recent arrival level, and previously active booths."),
        ("What is the Action?", "The number of toll booths to activate this minute (1 to max booths)."),
        ("What is the Reward?", "Feedback combining throughput gained, waiting penalty, operating cost, "
                                  "and a congestion penalty when the queue is Critical."),
        ("What is Q-learning?", "A model-free RL algorithm that learns the value (Q-value) of taking an "
                                  "action in a state, purely from experience — no model of the environment "
                                  "is needed in advance."),
        ("What is exploration?", "Trying different booth decisions (sometimes at random) to discover their effects."),
        ("What is exploitation?", "Choosing the action currently believed to be best, based on learned Q-values."),
    ]
    for q, a in qa:
        with st.expander(q):
            st.write(a)

    st.markdown("### ➗ The Core Update Rule")
    st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]")
    st.markdown("""
    - **Q(s,a)** — current estimated value of taking action *a* in state *s*
    - **α (alpha)** — learning rate: how much new information overrides old estimates
    - **r** — reward received after taking the action
    - **γ (gamma)** — discount factor: how much future rewards matter vs. immediate reward
    - **max Q(s',a')** — best possible value achievable from the next state
    """)

    st.markdown("### 📌 Project Limitations")
    st.markdown("""
    - Simulation is not connected to real toll infrastructure.
    - Arrival data is simulated using a Poisson process, not real sensor feeds.
    - Q-learning uses a simplified, discretized state representation.
    - A real deployment would require live sensor integration.
    - Safety constraints would be required before any automatic control of physical booths.
    """)

    st.markdown("### 🚀 Future Scope")
    st.markdown("""
    DQN · SARSA · Monte Carlo methods · REINFORCE (policy gradient) · Multi-agent RL ·
    real FASTag data integration · computer-vision-based queue detection · IoT sensors ·
    real-time traffic prediction · cloud deployment · adaptive lane routing.
    """)

# ================= EXPORT =================
with tab_export:
    st.markdown("### ⬇️ Export Results")
    if not ss.trained:
        st.info("Train the agent (and optionally run a simulation) to enable exports.")
    else:
        c1, c2, c3 = st.columns(3)

        if ss.sim_df is not None:
            csv_sim = ss.sim_df.to_csv(index=False).encode("utf-8")
            c1.download_button("Download Simulation CSV", csv_sim, "simulation_log.csv", "text/csv")
        else:
            c1.button("Download Simulation CSV", disabled=True)
            c1.caption("Run a simulation first.")

        csv_train = ss.training_df.to_csv(index=False).encode("utf-8")
        c2.download_button("Download Training Results CSV", csv_train, "training_results.csv", "text/csv")

        env, agent = ss.env, ss.agent
        action_cols = [f"Open {i+1}" for i in range(env.n_actions)]
        rows_idx = [f"{QUEUE_LEVELS[q]}|{ARRIVAL_LEVELS[a]}|Prev{b}"
                    for q, a, b in (env.state_components(s) for s in range(env.n_states))]
        q_export_df = pd.DataFrame(agent.Q, index=rows_idx, columns=action_cols)
        csv_q = q_export_df.to_csv().encode("utf-8")
        c3.download_button("Download Q-Table CSV", csv_q, "q_table.csv", "text/csv")

st.markdown("---")
st.caption("TollMind-RL · MCA Mini Project · Simulation only, not connected to real toll infrastructure.")
