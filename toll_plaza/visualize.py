"""Live visualization of the toll plaza with an RL agent dispatching vehicles."""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

from toll_plaza.toll_env import TollPlazaEnv
from toll_plaza.agent import QLearningAgent

BOOTH_X = 44.0
BOOTH_W = 5.0
EXIT_X = 80.0
VEHICLE_W = 3.0
VEHICLE_H = 0.55


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the toll plaza simulation")
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--etc", type=int, default=1)
    parser.add_argument("--arrival-rate", type=float, default=0.35, help="vehicles per second (keep below plaza capacity ~0.4/s for 4 lanes)")
    parser.add_argument("--episode-len", type=int, default=2000)
    parser.add_argument("--speed", type=int, default=2, help="simulation ticks per frame")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--q-table", default=None, help="path to trained Q-table .npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="max frames, then save/exit")
    parser.add_argument("--save", default=None, help="save animation to this file")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    env = TollPlazaEnv(
        n_lanes=args.lanes,
        n_etc=args.etc,
        arrival_rate=args.arrival_rate,
        episode_len=args.episode_len,
        seed=args.seed,
    )
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, seed=args.seed)
    if args.q_table:
        if not os.path.exists(args.q_table):
            raise FileNotFoundError(f"Q-table not found: {args.q_table}")
        agent.load(args.q_table)
        agent.epsilon = 0.0
        print(f"Loaded policy from {args.q_table}")
    else:
        print("No --q-table given; using a random (untrained) policy.")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title("Toll Plaza RL Simulation")

    def draw_static() -> None:
        ax.set_xlim(-2, 105)
        ax.set_ylim(-0.4, env.n_lanes + 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        for i, lane in enumerate(env.lanes):
            y = lane.index + 0.5
            ax.plot([-2, 105], [y, y], color="#334155", lw=2, zorder=0)
            booth_color = "#16a34a" if lane.is_etc else "#64748b"
            booth = Rectangle(
                (BOOTH_X, y - 0.35), BOOTH_W, 0.7,
                facecolor=booth_color, edgecolor="#1e293b", zorder=2,
            )
            ax.add_patch(booth)
            label = "ETC" if lane.is_etc else "TOLL"
            ax.text(
                BOOTH_X + BOOTH_W / 2, y + 0.65, label,
                ha="center", fontsize=8, fontweight="bold",
                color="#0f172a",
            )
            ax.text(
                -1.0, y, f"L{lane.index + 1}",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="#0f172a",
            )
            if lane.current is not None:
                _draw_vehicle(ax, lane.current, lane)
            for k, v in enumerate(lane.queue):
                _draw_vehicle(ax, v, lane, queue_pos=k)

    def _draw_vehicle(ax_, v, lane, queue_pos: int | None = None) -> None:
        y = lane.index + 0.5
        if queue_pos is not None:
            x = BOOTH_X - VEHICLE_W - 0.8 - (queue_pos + 1) * 3.6
        else:
            progress = 1.0 - v.service_remaining / max(v.service_time, 1e-9)
            x = BOOTH_X + BOOTH_W - 1.0 + progress * (EXIT_X - BOOTH_X - BOOTH_W)
        ax_.add_patch(
            Rectangle(
                (x, y - VEHICLE_H / 2), VEHICLE_W, VEHICLE_H,
                facecolor=v.vtype.color, edgecolor="#0f172a", zorder=3,
            )
        )

    def update(frame: int) -> list:
        for _ in range(args.speed):
            if env.done:
                env.reset()
            s = env.encode(env.state_tuple())
            a = agent.act(s, greedy=True)
            env.step(a)

        ax.clear()
        draw_static()
        ax.set_title(
            f"Toll Plaza RL Simulation  |  t={env.tick:5d}  arrivals={len(env.vehicles):5d}  "
            f"served={env.served_count:5d}  avg wait={env.avg_wait:5.1f}s  "
            f"revenue={env.revenue:7.1f}  queues={[len(l.queue) for l in env.lanes]}",
            fontsize=10,
        )
        return []

    anim = FuncAnimation(
        fig, update, frames=args.frames, interval=1000 // args.fps, blit=False, cache_frame_data=False
    )

    if args.save:
        anim.save(args.save, writer="pillow", fps=args.fps, dpi=100)
        print(f"Saved animation -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()