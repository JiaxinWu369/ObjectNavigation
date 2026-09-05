import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


SCENE = "FloorPlan321"
TARGET = "LightSwitch"
IDX = 244

GRID = Path(
    "datasets/Scene_Data/FloorPlan321/grid.json"
)

FILES = {
    "Frozen":
        Path(
            "results/runtime_frozen_zsval/"
            "episodes_bedroom.jsonl"
        ),
    "Semantic":
        Path(
            "results/guard_tau02_zsval/"
            "episodes_bedroom.jsonl"
        ),
    "ExecutionOnly":
        Path(
            "results/base500_br5_moveonly_zsval/"
            "episodes_bedroom.jsonl"
        ),
    "Full":
        Path(
            "results/fullbr_case_floorplan321/"
            "episodes_bedroom.jsonl"
        ),
}

OUT = Path(
    "results/qualitative_floorplan321"
)
OUT.mkdir(parents=True, exist_ok=True)

ACTIONS = [
    "Ahead",
    "Left",
    "Right",
    "Down",
    "Up",
    "Done",
]


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


def state_xz(s):
    p = s.split("|")
    return float(p[0]), float(p[1])


def parse_state(s):
    p = s.split("|")
    return {
        "x": float(p[0]),
        "z": float(p[1]),
        "rot": float(p[2]),
        "hor": float(p[3]),
    }


def heading_vec(rot, length=0.22):
    # AI2-THOR:
    # 0 deg   -> +z
    # 90 deg  -> +x
    r = np.deg2rad(rot)
    return (
        length * np.sin(r),
        length * np.cos(r),
    )


D = {
    name: load_case(path)
    for name, path in FILES.items()
}

F = D["Frozen"]
O = D["Full"]

with GRID.open() as f:
    grid = json.load(f)

gx = np.asarray(
    [float(p["x"]) for p in grid]
)

gz = np.asarray(
    [float(p["z"]) for p in grid]
)

fx, fz = zip(
    *[
        state_xz(s)
        for s in F["full_states"]
    ]
)

ox, oz = zip(
    *[
        state_xz(s)
        for s in O["full_states"]
    ]
)

fx = np.asarray(fx)
fz = np.asarray(fz)
ox = np.asarray(ox)
oz = np.asarray(oz)

sem = {
    int(x["step"]): x
    for x in O["case_semantic_trace"]
}

exe = {
    int(x["step"]): x
    for x in O["case_execution_trace"]
}

start = state_xz(
    O["full_states"][0]
)

p21 = state_xz(
    sem[21]["state"]
)

blocked_state = parse_state(
    sem[45]["state"]
)

final_full = state_xz(
    O["full_states"][-1]
)

# Frozen most repeated state
frozen_counts = Counter(
    F["full_states"]
)

frozen_stuck_state, frozen_stuck_n = (
    frozen_counts.most_common(1)[0]
)

frozen_stuck = state_xz(
    frozen_stuck_state
)

# Semantic t45
s45 = sem[45]

p_base = np.asarray(
    s45["prob_base"],
    dtype=float,
)

p_cal = np.asarray(
    s45["prob_calibrated"],
    dtype=float,
)

# Execution t46
e46 = exe[46]

p_exec = np.asarray(
    e46["policy_probs"],
    dtype=float,
)


# ============================================================
# Figure
# ============================================================

fig = plt.figure(
    figsize=(11.2, 6.4)
)

gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.08, 1.0],
    height_ratios=[1.0, 1.0],
    wspace=0.28,
    hspace=0.43,
)

ax_frozen = fig.add_subplot(
    gs[:, 0]
)

ax_sem = fig.add_subplot(
    gs[0, 1]
)

ax_exec = fig.add_subplot(
    gs[1, 1]
)


# ============================================================
# Left: combined Frozen vs Full trajectory
# ============================================================

ax = ax_frozen

ax.scatter(
    gx,
    gz,
    s=95,
    marker="s",
    alpha=0.18,
    edgecolors="none",
)

ax.plot(
    fx,
    fz,
    linewidth=1.8,
    linestyle="--",
    label="Frozen AKGVP",
)

ax.plot(
    ox,
    oz,
    linewidth=2.0,
    linestyle="-",
    label="Full Method",
)

# start
ax.scatter(
    [start[0]],
    [start[1]],
    marker="o",
    s=90,
    zorder=8,
)

ax.annotate(
    "Start",
    start,
    xytext=(5, 7),
    textcoords="offset points",
    fontsize=9,
)

# semantic-pathway divergence
ax.scatter(
    [p21[0]],
    [p21[1]],
    marker="D",
    s=75,
    zorder=8,
)

ax.annotate(
    "Semantic-pathway\n"
    "divergence (t=21)",
    p21,
    xytext=(8, -32),
    textcoords="offset points",
    fontsize=8.5,
)

# Frozen stuck location
ax.scatter(
    [frozen_stuck[0]],
    [frozen_stuck[1]],
    marker="X",
    s=120,
    zorder=9,
)

ax.annotate(
    "Frozen stuck\n"
    f"same state ×{frozen_stuck_n}",
    frozen_stuck,
    xytext=(-92, 12),
    textcoords="offset points",
    fontsize=8.5,
)

# Blocked point in Full
bx = blocked_state["x"]
bz = blocked_state["z"]

ax.scatter(
    [bx],
    [bz],
    marker="s",
    s=95,
    zorder=9,
)

ax.annotate(
    "MoveAhead ×5\n"
    "state unchanged",
    (bx, bz),
    xytext=(-104, -34),
    textcoords="offset points",
    fontsize=8.5,
)

# Heading before recovery: 315
dx1, dz1 = heading_vec(
    315,
    length=0.24,
)

ax.quiver(
    bx,
    bz,
    dx1,
    dz1,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.012,
    zorder=10,
)

# Heading after recovery: 0
dx2, dz2 = heading_vec(
    0,
    length=0.29,
)

ax.quiver(
    bx,
    bz,
    dx2,
    dz2,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.012,
    zorder=10,
)

ax.annotate(
    r"$315^\circ \rightarrow 0^\circ$"
    "\nExecution Feedback (t=46)",
    (bx, bz),
    xytext=(10, 18),
    textcoords="offset points",
    fontsize=8.5,
)

# Full success
ax.scatter(
    [final_full[0]],
    [final_full[1]],
    marker="*",
    s=150,
    zorder=10,
)

ax.annotate(
    "Success / 51",
    final_full,
    xytext=(7, 6),
    textcoords="offset points",
    fontsize=9,
)

ax.set_title(
    "(a) End-to-End Navigation",
    fontsize=11,
)

ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_aspect("equal")
ax.grid(False)

ax.legend(
    frameon=False,
    fontsize=9,
    loc="best",
)


# ============================================================
# Top-right: semantic calibration
# ============================================================

ax = ax_sem

x = np.arange(
    len(ACTIONS)
)

w = 0.34

ax.bar(
    x - w / 2,
    p_base,
    width=w,
    label="Base",
)

ax.bar(
    x + w / 2,
    p_cal,
    width=w,
    label="Calibrated",
)

ax.set_xticks(x)

ax.set_xticklabels(
    ACTIONS,
    rotation=25,
    fontsize=8,
)

ax.set_ylabel(
    "Probability",
    fontsize=9,
)

ax.set_ylim(
    0,
    max(
        0.58,
        float(
            max(
                p_base.max(),
                p_cal.max(),
            )
        )
        + 0.05,
    ),
)

ax.set_title(
    "(b) Semantic Calibration at t=45",
    fontsize=11,
)

ax.legend(
    frameon=False,
    fontsize=8,
)

ax.text(
    0.02,
    0.95,
    "4 local views\n"
    "Base: Right\n"
    "Calibrated: Ahead",
    transform=ax.transAxes,
    va="top",
    fontsize=8.5,
)


# ============================================================
# Bottom-right: execution feedback
# ============================================================

ax = ax_exec

ax.bar(
    x,
    p_exec,
    width=0.58,
)

ax.set_xticks(x)

ax.set_xticklabels(
    ACTIONS,
    rotation=25,
    fontsize=8,
)

ax.set_ylabel(
    "Probability",
    fontsize=9,
)

ax.set_ylim(
    0,
    max(
        0.58,
        float(p_exec.max()) + 0.05,
    ),
)

ax.set_title(
    "(c) Execution Feedback at t=46",
    fontsize=11,
)

ax.annotate(
    "Nominal\n0.524",
    xy=(0, p_exec[0]),
    xytext=(0, p_exec[0] + 0.045),
    ha="center",
    fontsize=8,
)

ax.annotate(
    "Executed\n0.444",
    xy=(2, p_exec[2]),
    xytext=(2, p_exec[2] + 0.045),
    ha="center",
    fontsize=8,
)

ax.text(
    0.02,
    0.95,
    "5 failed MoveAhead actions\n"
    "→ MoveAhead constrained\n"
    "→ highest-ranked admissible action",
    transform=ax.transAxes,
    va="top",
    fontsize=8.5,
)


# ============================================================
# Outcome strip
# ============================================================

fig.text(
    0.50,
    0.012,
    "Frozen: Fail / 100     |     "
    "Semantic only: Fail / 100     |     "
    "Execution only: Fail / 100     |     "
    "Full: Success / 51",
    ha="center",
    fontsize=9.5,
)

fig.savefig(
    OUT / "floorplan321_case_final.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "floorplan321_case_final.png",
    dpi=300,
    bbox_inches="tight",
)

print(
    "Saved:",
    OUT / "floorplan321_case_final.pdf"
)

print(
    "Saved:",
    OUT / "floorplan321_case_final.png"
)

print()
print(
    "Frozen most repeated state:",
    frozen_stuck_n,
    frozen_stuck_state,
)

print(
    "t45:",
    s45["action_base"],
    s45["action_calibrated"],
)

print(
    "t46 nominal/executed:",
    e46["nominal_action"],
    e46["executed_action"],
)
