import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SCENE = "FloorPlan321"
TARGET = "LightSwitch"
IDX = 244

GRID = Path(
    "datasets/Scene_Data/FloorPlan321/grid.json"
)

FROZEN = Path(
    "results/runtime_frozen_zsval/"
    "episodes_bedroom.jsonl"
)

FULL = Path(
    "results/fullbr_case_floorplan321/"
    "episodes_bedroom.jsonl"
)

OUT = Path(
    "results/qualitative_floorplan321"
)
OUT.mkdir(parents=True, exist_ok=True)


def load_case(path):
    with path.open() as f:
        for line in f:
            x = json.loads(line)
            if (
                int(x["eval_index"]) == IDX
                and x["scene"] == SCENE
                and x["target"] == TARGET
            ):
                return x
    raise RuntimeError(path)


def xy(state):
    p = state.split("|")
    return float(p[0]), float(p[1])


F = load_case(FROZEN)
O = load_case(FULL)

with GRID.open() as f:
    grid = json.load(f)

gx = np.array([float(p["x"]) for p in grid])
gz = np.array([float(p["z"]) for p in grid])

fx, fz = zip(*[xy(s) for s in F["full_states"]])
ox, oz = zip(*[xy(s) for s in O["full_states"]])

fx = np.asarray(fx)
fz = np.asarray(fz)
ox = np.asarray(ox)
oz = np.asarray(oz)

# Important event positions
sem = {
    int(x["step"]): x
    for x in O["case_semantic_trace"]
}

exe = {
    int(x["step"]): x
    for x in O["case_execution_trace"]
}

p21 = xy(sem[21]["state"])
p45 = xy(sem[45]["state"])
p46_after = xy(exe[46]["state_after"])

start = xy(O["full_states"][0])
final_full = xy(O["full_states"][-1])
final_frozen = xy(F["full_states"][-1])


# ============================================================
# Panel 1: Frozen
# ============================================================

fig, ax = plt.subplots(figsize=(4.6, 4.2))

ax.scatter(
    gx, gz,
    s=80,
    marker="s",
    alpha=0.22,
    edgecolors="none",
)

ax.plot(
    fx, fz,
    linewidth=1.8,
    label="Frozen AKGVP",
)

ax.scatter(
    [start[0]], [start[1]],
    s=90,
    marker="o",
    zorder=5,
)

ax.scatter(
    [final_frozen[0]],
    [final_frozen[1]],
    s=110,
    marker="x",
    zorder=6,
)

ax.annotate(
    "Start",
    start,
    xytext=(6, 6),
    textcoords="offset points",
)

ax.annotate(
    "Fail / 100",
    final_frozen,
    xytext=(6, -16),
    textcoords="offset points",
)

ax.set_title("Frozen AKGVP")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_aspect("equal")
ax.grid(False)

fig.tight_layout()

fig.savefig(
    OUT / "floorplan321_frozen.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "floorplan321_frozen.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Panel 2: Full method
# ============================================================

fig, ax = plt.subplots(figsize=(4.6, 4.2))

ax.scatter(
    gx, gz,
    s=80,
    marker="s",
    alpha=0.22,
    edgecolors="none",
)

ax.plot(
    ox, oz,
    linewidth=1.8,
    label="Full Method",
)

ax.scatter(
    [start[0]], [start[1]],
    s=90,
    marker="o",
    zorder=5,
)

ax.scatter(
    [p21[0]], [p21[1]],
    s=95,
    marker="D",
    zorder=6,
)

ax.scatter(
    [p45[0]], [p45[1]],
    s=105,
    marker="s",
    zorder=6,
)

ax.scatter(
    [p46_after[0]],
    [p46_after[1]],
    s=105,
    marker="^",
    zorder=6,
)

ax.scatter(
    [final_full[0]],
    [final_full[1]],
    s=110,
    marker="*",
    zorder=7,
)

ax.annotate(
    "Start",
    start,
    xytext=(6, 6),
    textcoords="offset points",
)

ax.annotate(
    "A  Semantic pathway\ntrajectory divergence (t=21)",
    p21,
    xytext=(8, 10),
    textcoords="offset points",
)

ax.annotate(
    "B  5 ineffective\nMoveAhead (t=41–45)",
    p45,
    xytext=(-110, -38),
    textcoords="offset points",
)

ax.annotate(
    "C  Execution Feedback\nRotateRight (t=46)",
    p46_after,
    xytext=(10, 8),
    textcoords="offset points",
)

ax.annotate(
    "Success / 51",
    final_full,
    xytext=(7, -18),
    textcoords="offset points",
)

ax.set_title("Full Method")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_aspect("equal")
ax.grid(False)

fig.tight_layout()

fig.savefig(
    OUT / "floorplan321_full.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "floorplan321_full.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Useful textual diagnostics
# ============================================================

from collections import Counter

fc = Counter(F["full_states"])

print("=" * 90)
print("CASE")
print("=" * 90)

print(
    "Frozen:",
    F["success"],
    F["ep_length"],
    F["spl"],
)

print(
    "Full:",
    O["success"],
    O["ep_length"],
    O["spl"],
)

print("\nStart:", start)
print("t21:", p21)
print("blocked:", p45)
print("after recovery:", p46_after)
print("final full:", final_full)
print("final frozen:", final_frozen)

print("\nMost repeated Frozen states:")
for state, n in fc.most_common(10):
    print(n, state)

print("\nSaved:")
print(OUT / "floorplan321_frozen.pdf")
print(OUT / "floorplan321_full.pdf")
