"""Train a Q-learning or SARSA agent on the toll plaza environment."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toll_plaza.toll_env import TollPlazaEnv
from toll_plaza.agent import QLearningAgent, SarsaAgent

AGENTS = {"qlearning": QLearningAgent, "sarsa": SarsaAgent}


def moving_average(x: np.ndarray, w: int = 25) -> np.ndarray:
    if len(x) < w:
        return x.astype(float)
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    models_dir = os.path.join(args.outdir, "models")
    os.makedirs(models_dir, exist_ok=True)

    env = TollPlazaEnv(
        n_lanes=args.lanes,
        n_etc=args.etc,
        arrival_rate=args.arrival_rate,
        episode_len=args.episode_len,
        seed=args.seed,
    )
    agent_cls = AGENTS[args.agent]
    agent = agent_cls(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_decay=args.decay,
        seed=args.seed,
    )

    log_path = os.path.join(args.outdir, f"training_{args.agent}.csv")
    history = []
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "total_reward", "avg_wait", "throughput", "revenue", "epsilon"]
        )
        for ep in range(1, args.episodes + 1):
            metrics, final_eps = run_episode(env, agent)
            history.append(metrics)
            writer.writerow(
                [ep, metrics["reward"], metrics["avg_wait"], metrics["throughput"],
                 metrics["revenue"], round(final_eps, 4)]
            )
            if ep % 10 == 0 or ep == 1:
                print(
                    f"ep {ep:4d} | reward {metrics['reward']:8.1f} | "
                    f"avg_wait {metrics['avg_wait']:6.2f}s | served {metrics['throughput']:4d} | "
                    f"revenue {metrics['revenue']:7.1f} | eps {final_eps:.3f}"
                )

    agent.save(os.path.join(models_dir, f"q_table_{args.agent}_{args.lanes}lanes.npy"))
    plot_learning_curve(history, args, plots_dir)
    print(f"\nSaved Q-table -> {models_dir}")
    print(f"Saved CSV log -> {log_path}")
    print(f"Saved plot    -> {os.path.join(plots_dir, 'learning_curve.png')}")


def run_episode(env: TollPlazaEnv, agent: QLearningAgent) -> tuple[dict, float]:
    state, _ = env.reset()
    s = env.encode(state)
    done = False
    total_reward = 0.0
    while not done:
        a = agent.act(s)
        next_state, r, done, _ = env.step(a)
        ns = env.encode(next_state)
        if isinstance(agent, SarsaAgent):
            na = agent.act(ns)
            agent.update(s, a, r, ns, na, done)
            a = na
        else:
            agent.update(s, a, r, ns, done)
        s = ns
        total_reward += r
    return {
        "reward": total_reward,
        "avg_wait": env.avg_wait,
        "throughput": env.served_count,
        "revenue": env.revenue,
    }, agent.epsilon


def plot_learning_curve(
    history: list[dict], args: argparse.Namespace, plots_dir: str
) -> None:
    rewards = np.array([h["reward"] for h in history])
    waits = np.array([h["avg_wait"] for h in history])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(rewards, color="#94a3b8", alpha=0.5, lw=0.8, label="per episode")
    ax1.plot(moving_average(rewards), color="#3b82f6", lw=2, label="moving avg")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("total reward")
    ax1.set_title("Episode reward")
    ax1.legend()

    ax2.plot(waits, color="#94a3b8", alpha=0.5, lw=0.8, label="per episode")
    ax2.plot(moving_average(waits), color="#ef4444", lw=2, label="moving avg")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("avg wait time (s)")
    ax2.set_title("Average wait time")
    ax2.legend()

    fig.suptitle(
        f"{args.agent} | {args.lanes} lanes ({args.etc} ETC) | arrival rate {args.arrival_rate}"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "learning_curve.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toll plaza RL agent")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--etc", type=int, default=1, help="number of fast ETC lanes")
    parser.add_argument("--arrival-rate", type=float, default=0.3, help="vehicles per second (keep below plaza capacity ~0.4/s for 4 lanes)")
    parser.add_argument("--episode-len", type=int, default=1500, help="simulation ticks per episode")
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--decay", type=float, default=0.999, help="epsilon decay per step")
    parser.add_argument("--agent", choices=AGENTS, default="qlearning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="out")
    train(parser.parse_args())


if __name__ == "__main__":
    main()