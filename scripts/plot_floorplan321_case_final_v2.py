import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ============================================================
# Configuration
# ============================================================

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

PDF = OUT / "floorplan321_case_final_v2.pdf"
PNG = OUT / "floorplan321_case_final_v2.png"


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# restrained publication colors
C_GRID = "#D9E5EC"
C_FROZEN = "#5B6573"
C_FULL = "#D97706"
C_SEM = "#2E7D32"
C_BLOCK = "#7E57C2"
C_EXEC = "#C62828"
C_TEXT = "#202124"
C_MUTED = "#666666"
C_PANEL = "#F7F7F7"


# ============================================================
# Helpers
# ============================================================

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

    raise RuntimeError(
        f"Case not found in {path}"
    )


def state_xz(state):
    p = state.split("|")

    return (
        float(p[0]),
        float(p[1]),
    )


def parse_state(state):
    p = state.split("|")

    return {
        "x": float(p[0]),
        "z": float(p[1]),
        "rotation": float(p[2]),
        "horizon": float(p[3]),
    }


def heading_vector(rotation, length=0.18):
    """
    AI2-THOR convention:
      rotation 0   -> +z
      rotation 90  -> +x
    """
    r = np.deg2rad(rotation)

    return (
        length * np.sin(r),
        length * np.cos(r),
    )


def get_any(d, *keys):
    for k in keys:
        if k in d:
            return d[k]

    raise KeyError(
        f"None of these keys exists: {keys}\n"
        f"available keys = {sorted(d.keys())}"
    )


def result_text(rec):
    success = bool(rec["success"])

    length = int(
        rec.get(
            "ep_length",
            len(rec.get("full_states", []))
        )
    )

    return (
        ("Success" if success else "Fail")
        + f" / {length}"
    )


def add_text_box(
    ax,
    x,
    y,
    text,
    width,
    height,
    fontsize=8.5,
    facecolor="#F7F7F7",
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#BBBBBB",
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )

    ax.add_patch(box)

    ax.text(
        x + 0.02,
        y + height / 2,
        text,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=fontsize,
        color=C_TEXT,
    )


# ============================================================
# Load episodes
# ============================================================

D = {
    name: load_case(path)
    for name, path in FILES.items()
}

F = D["Frozen"]
S = D["Semantic"]
E = D["ExecutionOnly"]
O = D["Full"]


# ============================================================
# Load reachable grid
# ============================================================

with GRID.open() as f:
    grid = json.load(f)

gx = np.asarray([
    float(p["x"])
    for p in grid
])

gz = np.asarray([
    float(p["z"])
    for p in grid
])


# ============================================================
# Trajectories
# ============================================================

f_xy = np.asarray([
    state_xz(s)
    for s in F["full_states"]
])

o_xy = np.asarray([
    state_xz(s)
    for s in O["full_states"]
])

fx = f_xy[:, 0]
fz = f_xy[:, 1]

ox = o_xy[:, 0]
oz = o_xy[:, 1]


# ============================================================
# Full trace
# ============================================================

sem = {
    int(x["step"]): x
    for x in O["case_semantic_trace"]
}

exe = {
    int(x["step"]): x
    for x in O["case_execution_trace"]
}


# ============================================================
# Important states
# ============================================================

start = state_xz(
    O["full_states"][0]
)

# First Frozen-vs-Semantic action divergence occurs at t=21.
# State trajectories diverge one step later at t=22.
p21 = state_xz(
    sem[21]["state"]
)

blocked = parse_state(
    sem[45]["state"]
)

final_full = state_xz(
    O["full_states"][-1]
)

final_frozen = state_xz(
    F["full_states"][-1]
)


# ============================================================
# Frozen repeated state
# ============================================================

frozen_counts = Counter(
    F["full_states"]
)

frozen_stuck_state, frozen_stuck_n = (
    frozen_counts.most_common(1)[0]
)

frozen_stuck = state_xz(
    frozen_stuck_state
)


# ============================================================
# Semantic probabilities at t=45
# ============================================================

s45 = sem[45]

p_base = np.asarray(
    get_any(
        s45,
        "prob_base",
        "base_probs",
    ),
    dtype=float,
)

p_cal = np.asarray(
    get_any(
        s45,
        "prob_calibrated",
        "calibrated_probs",
        "lcr_probs",
    ),
    dtype=float,
)

# action indices:
# 0 MoveAhead
# 1 RotateLeft
# 2 RotateRight
# 3 LookDown
# 4 LookUp
# 5 Done

SEM_ACTIONS = [
    "Ahead",
    "Right",
]

sem_base_vals = [
    p_base[0],
    p_base[2],
]

sem_cal_vals = [
    p_cal[0],
    p_cal[2],
]


# ============================================================
# Execution probabilities at t=46
# ============================================================

e46 = exe[46]

p_exec = np.asarray(
    get_any(
        e46,
        "policy_probs",
        "probs",
        "prob",
    ),
    dtype=float,
)

exec_ahead = float(
    p_exec[0]
)

exec_right = float(
    p_exec[2]
)


# ============================================================
# Crop navigable layout around trajectories
# ============================================================

all_x = np.concatenate([
    fx,
    ox,
])

all_z = np.concatenate([
    fz,
    oz,
])

PAD_X = 0.45
PAD_Z = 0.45

xmin = all_x.min() - PAD_X
xmax = all_x.max() + PAD_X

zmin = all_z.min() - PAD_Z
zmax = all_z.max() + PAD_Z

grid_mask = (
    (gx >= xmin)
    & (gx <= xmax)
    & (gz >= zmin)
    & (gz <= zmax)
)


# ============================================================
# Figure structure
#
# left:
#   episode trajectory
#
# right-top:
#   three-module timeline
#
# right-bottom-left:
#   semantic calibration
#
# right-bottom-right:
#   execution feedback
# ============================================================

fig = plt.figure(
    figsize=(12.2, 6.35)
)

outer = fig.add_gridspec(
    nrows=2,
    ncols=2,
    width_ratios=[1.18, 1.0],
    height_ratios=[0.78, 1.0],
    wspace=0.24,
    hspace=0.32,
)

ax_traj = fig.add_subplot(
    outer[:, 0]
)

ax_time = fig.add_subplot(
    outer[0, 1]
)

bottom = outer[1, 1].subgridspec(
    1,
    2,
    wspace=0.32,
)

ax_sem = fig.add_subplot(
    bottom[0, 0]
)

ax_exec = fig.add_subplot(
    bottom[0, 1]
)


# ============================================================
# (a) Episode-level trajectory
# ============================================================

ax = ax_traj

# reachable layout
ax.scatter(
    gx[grid_mask],
    gz[grid_mask],
    s=95,
    marker="s",
    color=C_GRID,
    edgecolors="none",
    zorder=1,
)

# Frozen
ax.plot(
    fx,
    fz,
    color=C_FROZEN,
    linestyle="--",
    linewidth=2.2,
    alpha=0.95,
    label="Frozen AKGVP",
    zorder=3,
)

# Full
ax.plot(
    ox,
    oz,
    color=C_FULL,
    linestyle="-",
    linewidth=2.5,
    label="Full Method",
    zorder=4,
)


# ------------------------------------------------------------
# Start
# ------------------------------------------------------------

ax.scatter(
    start[0],
    start[1],
    s=85,
    marker="o",
    facecolor="white",
    edgecolor=C_TEXT,
    linewidth=1.4,
    zorder=10,
)

ax.annotate(
    "Start",
    xy=start,
    xytext=(7, 10),
    textcoords="offset points",
    fontsize=8.5,
    color=C_TEXT,
    bbox=dict(
        boxstyle="round,pad=0.18",
        facecolor="white",
        edgecolor="none",
        alpha=0.9,
    ),
)


# ------------------------------------------------------------
# Semantic-pathway decision point
# ------------------------------------------------------------

ax.scatter(
    p21[0],
    p21[1],
    s=82,
    marker="D",
    color=C_SEM,
    zorder=10,
)

ax.annotate(
    "First Frozen–Semantic\n"
    "action divergence\n"
    "(t=21)",
    xy=p21,
    xytext=(14, -43),
    textcoords="offset points",
    fontsize=8.2,
    color=C_TEXT,
    arrowprops=dict(
        arrowstyle="-",
        linewidth=0.8,
        color=C_SEM,
    ),
    bbox=dict(
        boxstyle="round,pad=0.22",
        facecolor="white",
        edgecolor=C_SEM,
        linewidth=0.7,
        alpha=0.95,
    ),
)


# ------------------------------------------------------------
# Frozen stuck point
# ------------------------------------------------------------

ax.scatter(
    frozen_stuck[0],
    frozen_stuck[1],
    s=120,
    marker="X",
    color=C_EXEC,
    zorder=11,
)

ax.annotate(
    f"Frozen stuck\n"
    f"same state ×{frozen_stuck_n}\n"
    f"Fail / 100",
    xy=frozen_stuck,
    xytext=(-102, 24),
    textcoords="offset points",
    fontsize=8.2,
    color=C_TEXT,
    arrowprops=dict(
        arrowstyle="-",
        linewidth=0.8,
        color=C_EXEC,
    ),
    bbox=dict(
        boxstyle="round,pad=0.22",
        facecolor="white",
        edgecolor=C_EXEC,
        linewidth=0.7,
        alpha=0.95,
    ),
)


# ------------------------------------------------------------
# Blocked / Execution Feedback point
# ------------------------------------------------------------

bx = blocked["x"]
bz = blocked["z"]

ax.scatter(
    bx,
    bz,
    s=90,
    marker="s",
    color=C_BLOCK,
    zorder=11,
)

ax.annotate(
    "5 ineffective\n"
    "MoveAhead\n"
    "(t=41–45)",
    xy=(bx, bz),
    xytext=(-92, -47),
    textcoords="offset points",
    fontsize=8.2,
    color=C_TEXT,
    arrowprops=dict(
        arrowstyle="-",
        linewidth=0.8,
        color=C_BLOCK,
    ),
    bbox=dict(
        boxstyle="round,pad=0.22",
        facecolor="white",
        edgecolor=C_BLOCK,
        linewidth=0.7,
        alpha=0.95,
    ),
)


# headings 315 -> 0
dx1, dz1 = heading_vector(
    315,
    length=0.17,
)

dx2, dz2 = heading_vector(
    0,
    length=0.21,
)

ax.arrow(
    bx,
    bz,
    dx1,
    dz1,
    width=0.008,
    head_width=0.055,
    head_length=0.06,
    length_includes_head=True,
    color=C_FROZEN,
    zorder=12,
)

ax.arrow(
    bx,
    bz,
    dx2,
    dz2,
    width=0.008,
    head_width=0.055,
    head_length=0.06,
    length_includes_head=True,
    color=C_EXEC,
    zorder=12,
)

ax.annotate(
    "Execution Feedback\n"
    r"$315^\circ \rightarrow 0^\circ$"
    "\n(t=46)",
    xy=(bx, bz),
    xytext=(18, 24),
    textcoords="offset points",
    fontsize=8.2,
    color=C_TEXT,
    arrowprops=dict(
        arrowstyle="-",
        linewidth=0.8,
        color=C_EXEC,
    ),
    bbox=dict(
        boxstyle="round,pad=0.22",
        facecolor="white",
        edgecolor=C_EXEC,
        linewidth=0.7,
        alpha=0.95,
    ),
)


# ------------------------------------------------------------
# Final success
# ------------------------------------------------------------

ax.scatter(
    final_full[0],
    final_full[1],
    s=145,
    marker="*",
    color=C_FULL,
    edgecolor=C_TEXT,
    linewidth=0.5,
    zorder=12,
)

ax.annotate(
    "Success / 51",
    xy=final_full,
    xytext=(8, 10),
    textcoords="offset points",
    fontsize=8.5,
    color=C_TEXT,
    bbox=dict(
        boxstyle="round,pad=0.18",
        facecolor="white",
        edgecolor="none",
        alpha=0.9,
    ),
)


# ------------------------------------------------------------
# Trajectory axis
# ------------------------------------------------------------

ax.set_xlim(
    xmin,
    xmax,
)

ax.set_ylim(
    zmin,
    zmax,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.set_xlabel(
    "x (m)"
)

ax.set_ylabel(
    "z (m)"
)

ax.set_title(
    "(a) End-to-End Navigation",
    pad=8,
)

ax.legend(
    frameon=False,
    loc="lower left",
)

ax.text(
    0.02,
    0.02,
    "Cropped top-down projection of reachable locations",
    transform=ax.transAxes,
    fontsize=7.5,
    color=C_MUTED,
)


# ============================================================
# (b) Three-module timeline
# ============================================================

ax = ax_time

ax.axis("off")

ax.set_title(
    "(b) Integrated Method Timeline",
    pad=4,
)


# Base line
y = 0.50

x_pts = [
    0.10,
    0.39,
    0.70,
    0.92,
]

for i in range(
    len(x_pts) - 1
):
    ax.annotate(
        "",
        xy=(
            x_pts[i + 1] - 0.04,
            y,
        ),
        xytext=(
            x_pts[i] + 0.04,
            y,
        ),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.4,
            color="#777777",
        ),
    )


# Local Evidence
add_text_box(
    ax,
    0.02,
    0.28,
    "Local Evidence Modeling\n"
    "t=9\n"
    "3 distinct local views",
    width=0.22,
    height=0.44,
    fontsize=8.0,
    facecolor="#F2F6F3",
)

# Semantic
add_text_box(
    ax,
    0.29,
    0.28,
    "Semantic Calibration\n"
    "t=21\n"
    "first Frozen–Semantic\n"
    "action divergence",
    width=0.25,
    height=0.44,
    fontsize=8.0,
    facecolor="#F2F6F3",
)

# Execution detection
add_text_box(
    ax,
    0.59,
    0.28,
    "Execution Feedback\n"
    "t=41–45\n"
    "5 ineffective\n"
    "MoveAhead actions",
    width=0.22,
    height=0.44,
    fontsize=8.0,
    facecolor="#FAF3F2",
)

# Recovery
add_text_box(
    ax,
    0.84,
    0.28,
    "t=46–50\n"
    "constrain Ahead\n"
    "→ RotateRight\n"
    "→ Success",
    width=0.15,
    height=0.44,
    fontsize=7.8,
    facecolor="#FAF3F2",
)


ax.text(
    0.39,
    0.10,
    "The calibrated recurrent state is retained when Base and "
    "Calibrated actions agree.",
    transform=ax.transAxes,
    ha="center",
    fontsize=7.5,
    color=C_MUTED,
)


# ============================================================
# (c) Semantic Calibration snapshot
# ============================================================

ax = ax_sem

ypos = np.arange(
    len(SEM_ACTIONS)
)

h = 0.34

ax.barh(
    ypos + h / 2,
    sem_base_vals,
    height=h,
    color=C_FROZEN,
    label="Base",
)

ax.barh(
    ypos - h / 2,
    sem_cal_vals,
    height=h,
    color=C_FULL,
    label="Calibrated",
)

ax.set_yticks(
    ypos
)

ax.set_yticklabels(
    SEM_ACTIONS
)

ax.set_xlim(
    0,
    0.58,
)

ax.set_xlabel(
    "Probability"
)

ax.set_title(
    "(c) Semantic Calibration\nat t=45",
    pad=5,
)

ax.legend(
    frameon=False,
    loc="lower right",
)


for i, v in enumerate(
    sem_base_vals
):
    ax.text(
        v + 0.008,
        i + h / 2,
        f"{v:.3f}",
        va="center",
        fontsize=7.8,
    )

for i, v in enumerate(
    sem_cal_vals
):
    ax.text(
        v + 0.008,
        i - h / 2,
        f"{v:.3f}",
        va="center",
        fontsize=7.8,
    )


ax.text(
    0.02,
    0.96,
    "4 local views\n"
    "target evidence = 0.999\n"
    "context support = 0.324",
    transform=ax.transAxes,
    va="top",
    fontsize=7.5,
    color=C_MUTED,
)


# ============================================================
# (d) Execution Feedback
# ============================================================

ax = ax_exec

labels = [
    "Nominal\nAhead",
    "Executed\nRight",
]

vals = [
    exec_ahead,
    exec_right,
]

bars = ax.bar(
    [0, 1],
    vals,
    width=0.52,
    color=[
        C_FROZEN,
        C_EXEC,
    ],
)

ax.set_xticks(
    [0, 1]
)

ax.set_xticklabels(
    labels
)

ax.set_ylim(
    0,
    0.60,
)

ax.set_ylabel(
    "Probability"
)

ax.set_title(
    "(d) Execution Feedback\nat t=46",
    pad=5,
)


for bar, v in zip(
    bars,
    vals,
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        v + 0.018,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )


# constraint arrow
ax.annotate(
    "5 stalls\n"
    "→ Ahead constrained",
    xy=(
        0,
        exec_ahead * 0.55,
    ),
    xytext=(
        0.52,
        0.30,
    ),
    textcoords="data",
    fontsize=7.8,
    ha="center",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.9,
        color=C_EXEC,
    ),
)


ax.text(
    0.50,
    0.06,
    "highest-ranked\nadmissible action",
    transform=ax.transAxes,
    ha="center",
    fontsize=7.5,
    color=C_MUTED,
)


# ============================================================
# Outcome strip
# ============================================================

outcome = (
    f"Frozen: {result_text(F)}"
    "     |     "
    f"Semantic only: {result_text(S)}"
    "     |     "
    f"Execution only: {result_text(E)}"
    "     |     "
    f"Full: {result_text(O)}"
)

fig.text(
    0.5,
    0.018,
    outcome,
    ha="center",
    va="bottom",
    fontsize=9.2,
    color=C_TEXT,
)


# ============================================================
# Save
# ============================================================

fig.subplots_adjust(
    bottom=0.095,
    top=0.94,
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Diagnostics
# ============================================================

print("=" * 90)
print("FINAL CASE FIGURE V2")
print("=" * 90)

print(
    "Frozen:",
    result_text(F)
)

print(
    "Semantic:",
    result_text(S)
)

print(
    "ExecutionOnly:",
    result_text(E)
)

print(
    "Full:",
    result_text(O)
)

print()

print(
    "Frozen most repeated state:",
    frozen_stuck_n,
    frozen_stuck_state,
)

print(
    "First Frozen-Semantic action divergence:",
    "t=21",
    p21,
)

print(
    "Blocked state:",
    sem[45]["state"],
)

print(
    "Execution state after RotateRight:",
    exe[46]["state_after"],
)

print()

print(
    "t45 Base:",
    "Ahead={:.6f}".format(p_base[0]),
    "Right={:.6f}".format(p_base[2]),
)

print(
    "t45 Calibrated:",
    "Ahead={:.6f}".format(p_cal[0]),
    "Right={:.6f}".format(p_cal[2]),
)

print(
    "t46 policy:",
    "Ahead={:.6f}".format(exec_ahead),
    "Right={:.6f}".format(exec_right),
)

print()

print("Saved:")
print(PDF)
print(PNG)
