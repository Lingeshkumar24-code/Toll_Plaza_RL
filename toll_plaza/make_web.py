"""Generate a self-contained browser simulation with the trained Q-table embedded."""

from __future__ import annotations

import argparse
import json
import os
import webbrowser

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Build toll_plaza_sim.html for the browser")
    parser.add_argument(
        "--q-table",
        default=os.path.join("out", "models", "q_table_qlearning_4lanes.npy"),
        help="path to a trained Q-table .npy (skip to embed no policy)",
    )
    parser.add_argument("--out", default="toll_plaza_sim.html")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    template_path = os.path.join("toll_plaza", "web_template.html")
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    if os.path.exists(args.q_table):
        q = np.load(args.q_table)
        payload = json.dumps(q.astype(float).ravel().tolist(), separators=(",", ":"))
        print(f"Embedded Q-table {q.shape} from {args.q_table}")
    else:
        payload = "null"
        print("Q-table not found; embedding no policy (starts untrained).")

    html = html.replace("__QTABLE_JSON__", payload)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(args.out) // 1024
    print(f"Wrote {args.out} ({size_kb} KB)")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(args.out).replace("\\", "/"))


if __name__ == "__main__":
    main()