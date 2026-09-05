#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export real RGB frames + paired K10/K20 trajectories for Figure 1.

Default paired policies:
    K10: semantic context restored at t=10
    K20: semantic context restored at t=20

Default Figure-1 cases:
    (a) Restore Now Succeeds
        living_room / FloorPlan223 / LightSwitch / eval_index=0
        K10 success, K20 failure

    (b) Delay Succeeds
        bedroom / FloorPlan325 / Laptop / eval_index=39
        K10 failure, K20 success

Outputs:
    figures/fig1_final_cases/
        restore_help_FloorPlan223_LightSwitch_idx0/
            early10_episode.jsonl
            early20_episode.jsonl
            paired_meta.json
            shared_t00.png
            shared_t05.png
            shared_t10.png
            k10_t20.png
            k10_final.png
            k20_t20.png
            k20_final.png
            trajectory.png
            trajectory.pdf
            trajectory.svg

        delay_help_FloorPlan325_Laptop_idx39/
            ...

        selected_cases.csv
        fig1_draft.png
        fig1_draft.pdf
"""

import os
import sys
import json
import csv
import glob
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from ai2thor.controller import Controller


# ============================================================
# Figure style
# ============================================================

COLOR_SHARED = "#66788A"
COLOR_K10 = "#E69F00"       # Restore @ 10
COLOR_K20 = "#4C78A8"       # Delay to 20
COLOR_SUCCESS = "#2E8B57"
COLOR_FAILURE = "#C95A5A"
COLOR_FLOOR = "#E8EAED"
COLOR_TEXT = "#222222"

DECISION_STEP = 10
DELAY_STEP = 20


# ============================================================
# Cases fixed for Figure 1
# ============================================================

CASES = [
    {
        "case_id": "restore_help_FloorPlan223_LightSwitch_idx0",
        "panel_title": "Restore Now Succeeds",
        "room": "living_room",
        "scene": "FloorPlan223",
        "target": "LightSwitch",
        "eval_index": 0,
        "expect_k10_success": True,
        "expect_k20_success": False,
    },
    {
        "case_id": "delay_help_FloorPlan325_Laptop_idx39",
        "panel_title": "Delay Succeeds",
        "room": "bedroom",
        "scene": "FloorPlan325",
        "target": "Laptop",
        "eval_index": 39,
        "expect_k10_success": False,
        "expect_k20_success": True,
    },
]


# ============================================================
# Basic helpers
# ============================================================

def log(msg=""):
    print(msg, flush=True)


def parse_state(state):
    """
    AKGVP state format:
        x|z|rotation|horizon

    Example:
        0.00|1.50|225|30
    """
    if not isinstance(state, str):
        raise TypeError(f"State must be str, got {type(state)}")

    p = state.split("|")
    if len(p) < 4:
        raise ValueError(f"Invalid state: {state}")

    return {
        "x": float(p[0]),
        "z": float(p[1]),
        "rotation": float(p[2]),
        "horizon": float(p[3]),
    }


def state_xy(states):
    pts = [parse_state(s) for s in states]
    x = np.asarray([p["x"] for p in pts], dtype=np.float32)
    z = np.asarray([p["z"] for p in pts], dtype=np.float32)
    return x, z


def bool_value(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "success"}
    return bool(v)


def episode_length(rec):
    if "ep_length" in rec:
        try:
            return int(rec["ep_length"])
        except Exception:
            pass

    if "full_actions" in rec:
        return len(rec["full_actions"])

    if "actions" in rec:
        return len(rec["actions"])

    return -1


# ============================================================
# JSONL discovery
# ============================================================

def list_episode_files(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Result directory does not exist: {root}")

    files = sorted(root.rglob("episodes_*.jsonl"))

    if not files:
        # fallback in case naming differs slightly
        files = sorted(root.rglob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No JSONL files found under: {root}")

    return files


def find_episode(root, case):
    """
    Search recursively so we do not depend on exact room JSONL filename.
    Match by:
        scene
        target
        eval_index
    scene_type is used as an additional check when present.
    """
    matches = []

    for fp in list_episode_files(root):
        with open(fp, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                if str(rec.get("scene")) != str(case["scene"]):
                    continue

                if str(rec.get("target")) != str(case["target"]):
                    continue

                try:
                    idx = int(rec.get("eval_index"))
                except Exception:
                    continue

                if idx != int(case["eval_index"]):
                    continue

                scene_type = rec.get("scene_type")
                if scene_type is not None:
                    if str(scene_type) != str(case["room"]):
                        continue

                matches.append((fp, line_no, rec))

    if len(matches) != 1:
        msg = [
            f"Expected exactly one episode but found {len(matches)}",
            f"root={root}",
            f"scene={case['scene']}",
            f"target={case['target']}",
            f"eval_index={case['eval_index']}",
        ]

        for x in matches[:20]:
            msg.append(f"  {x[0]}:{x[1]}")

        raise RuntimeError("\n".join(msg))

    return matches[0]


# ============================================================
# Pair validation
# ============================================================

def first_difference(a, b, start=0):
    n = min(len(a), len(b))
    for i in range(start, n):
        if a[i] != b[i]:
            return i

    if len(a) != len(b):
        return n

    return None


def validate_pair(rec10, rec20, case):
    required = [
        "scene",
        "target",
        "eval_index",
        "success",
        "full_actions",
        "full_states",
    ]

    for key in required:
        if key not in rec10:
            raise KeyError(f"K10 record missing {key}")
        if key not in rec20:
            raise KeyError(f"K20 record missing {key}")

    assert rec10["scene"] == rec20["scene"], (
        rec10["scene"], rec20["scene"]
    )
    assert rec10["target"] == rec20["target"], (
        rec10["target"], rec20["target"]
    )
    assert int(rec10["eval_index"]) == int(rec20["eval_index"])

    a10 = rec10["full_actions"]
    a20 = rec20["full_actions"]
    s10 = rec10["full_states"]
    s20 = rec20["full_states"]

    if len(a10) < DECISION_STEP or len(a20) < DECISION_STEP:
        raise RuntimeError("Episode does not reach t=10")

    if len(s10) <= DECISION_STEP or len(s20) <= DECISION_STEP:
        raise RuntimeError("full_states does not contain state s10")

    # shared actions are actions 0,...,9
    if a10[:DECISION_STEP] != a20[:DECISION_STEP]:
        raise AssertionError(
            "K10/K20 first 10 actions are NOT identical."
        )

    # state after 10 actions
    if s10[DECISION_STEP] != s20[DECISION_STEP]:
        raise AssertionError(
            "K10/K20 state at t=10 is NOT identical:\n"
            f"K10: {s10[DECISION_STEP]}\n"
            f"K20: {s20[DECISION_STEP]}"
        )

    actual10 = bool_value(rec10["success"])
    actual20 = bool_value(rec20["success"])

    if actual10 != case["expect_k10_success"]:
        raise AssertionError(
            f"Unexpected K10 outcome for {case['case_id']}: "
            f"{actual10}"
        )

    if actual20 != case["expect_k20_success"]:
        raise AssertionError(
            f"Unexpected K20 outcome for {case['case_id']}: "
            f"{actual20}"
        )

    first_action_div = first_difference(
        a10, a20, start=DECISION_STEP
    )
    first_state_div = first_difference(
        s10, s20, start=DECISION_STEP
    )

    return {
        "prefix_actions_match_0_9": True,
        "state_s10_match": True,
        "decision_state": s10[DECISION_STEP],
        "first_action_difference": first_action_div,
        "first_state_difference": first_state_div,
        "k10_success": actual10,
        "k20_success": actual20,
        "k10_ep_length": episode_length(rec10),
        "k20_ep_length": episode_length(rec20),
    }


# ============================================================
# Grid / floor footprint
# ============================================================

def recursively_extract_xz(obj, out):
    if isinstance(obj, dict):
        if "x" in obj and "z" in obj:
            try:
                out.append((float(obj["x"]), float(obj["z"])))
            except Exception:
                pass

        for v in obj.values():
            recursively_extract_xz(v, out)

    elif isinstance(obj, list):
        for v in obj:
            recursively_extract_xz(v, out)


def load_grid_points(repo_root, scene):
    """
    Best-effort loading of offline AI2-THOR grid.json.
    If not found or format differs, simply returns None.
    """
    base = Path(repo_root) / "datasets" / "Scene_Data"

    candidates = [
        base / scene / "grid.json",
        base / f"{scene}_physics" / "grid.json",
    ]

    # fallback recursive lookup
    if base.exists():
        for p in base.rglob("grid.json"):
            parent = p.parent.name
            if parent in {scene, f"{scene}_physics"}:
                candidates.append(p)

    seen = set()

    for fp in candidates:
        fp = Path(fp)

        if fp in seen:
            continue
        seen.add(fp)

        if not fp.exists():
            continue

        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)

            pts = []
            recursively_extract_xz(obj, pts)

            if pts:
                # unique
                pts = sorted(set(
                    (round(x, 5), round(z, 5))
                    for x, z in pts
                ))
                return np.asarray(pts, dtype=np.float32)

        except Exception:
            continue

    return None


# ============================================================
# AI2-THOR live renderer
# ============================================================

class LiveRGBRenderer:
    def __init__(self, quality="Low", width=300, height=300):
        self.quality = quality
        self.width = int(width)
        self.height = int(height)

        self.controller = Controller(quality=quality)
        self.current_scene = None
        self.y = None

        log(
            f"[AI2-THOR] starting controller "
            f"quality={quality}, size={width}x{height}"
        )

        self.controller.start(
            player_screen_width=self.width,
            player_screen_height=self.height,
        )

        log("[AI2-THOR] controller started")

    def set_scene(self, scene):
        if scene == self.current_scene:
            return

        log(f"[AI2-THOR] reset -> {scene}")

        ev = self.controller.reset(scene)

        ev = self.controller.step(dict(
            action="Initialize",
            gridSize=0.25,
            fieldOfView=100,
            rotateStepDegrees=45,
        ))

        if not ev.metadata.get("lastActionSuccess", False):
            # Old THOR occasionally reports non-fatal environment messages.
            log(
                "[WARN] Initialize lastActionSuccess=False: "
                + str(ev.metadata.get("errorMessage", ""))
            )

        self.y = float(
            ev.metadata["agent"]["position"]["y"]
        )

        self.current_scene = scene

        log(
            f"[AI2-THOR] scene ready: {scene}, y={self.y:.6f}"
        )

    def render_state(self, scene, state, save_path):
        self.set_scene(scene)

        p = parse_state(state)

        ev = self.controller.step(dict(
            action="TeleportFull",
            x=p["x"],
            y=self.y,
            z=p["z"],
            rotation=dict(
                x=0.0,
                y=p["rotation"],
                z=0.0,
            ),
            horizon=p["horizon"],
        ))

        if not ev.metadata.get("lastActionSuccess", False):
            raise RuntimeError(
                "TeleportFull failed\n"
                f"scene={scene}\n"
                f"state={state}\n"
                f"error={ev.metadata.get('errorMessage', '')}"
            )

        frame = np.asarray(ev.frame)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(
                f"Expected RGB image, got shape={frame.shape}"
            )

        frame = frame.astype(np.uint8)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(frame).save(save_path)

        return {
            "requested_state": state,
            "actual_position": ev.metadata["agent"]["position"],
            "actual_rotation": ev.metadata["agent"]["rotation"],
            "actual_horizon": float(
                ev.metadata["agent"]["cameraHorizon"]
            ),
            "frame_shape": list(frame.shape),
        }

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass


# ============================================================
# RGB frame export
# ============================================================

def safe_state(states, t):
    if len(states) == 0:
        raise RuntimeError("Empty full_states")

    t = max(0, min(int(t), len(states) - 1))
    return t, states[t]


def export_case_rgb(renderer, case_dir, case, rec10, rec20):
    """
    Minimal set for quickly completing Figure 1:

    Shared:
        t0
        t5
        t10

    K10 branch:
        t20
        final

    K20 branch:
        t20
        final
    """
    scene = case["scene"]

    states10 = rec10["full_states"]
    states20 = rec20["full_states"]

    frame_specs = []

    # shared prefix: use K10 copy, already validated equal through s10
    for t in [0, 5, 10]:
        tt, state = safe_state(states10, t)
        frame_specs.append(
            (f"shared_t{tt:02d}.png", "shared", tt, state)
        )

    # K10 branch
    tt, state = safe_state(states10, DELAY_STEP)
    frame_specs.append(
        (f"k10_t{tt:02d}.png", "k10", tt, state)
    )

    tt10_final = len(states10) - 1
    frame_specs.append(
        (
            "k10_final.png",
            "k10",
            tt10_final,
            states10[-1],
        )
    )

    # K20 branch
    tt, state = safe_state(states20, DELAY_STEP)
    frame_specs.append(
        (f"k20_t{tt:02d}.png", "k20", tt, state)
    )

    tt20_final = len(states20) - 1
    frame_specs.append(
        (
            "k20_final.png",
            "k20",
            tt20_final,
            states20[-1],
        )
    )

    render_meta = []

    # Avoid duplicate rendering if two specs happen to be same file/state
    for filename, branch, t, state in frame_specs:
        out = case_dir / filename

        log(
            f"    RGB {case['case_id']} "
            f"{branch} t={t}: {state}"
        )

        info = renderer.render_state(
            scene=scene,
            state=state,
            save_path=out,
        )

        info.update({
            "file": filename,
            "branch": branch,
            "t": int(t),
        })

        render_meta.append(info)

    return render_meta


# ============================================================
# Top-down trajectory
# ============================================================

def plot_trajectory_axis(
    ax,
    rec10,
    rec20,
    grid_points=None,
    show_legend=True,
):
    s10 = rec10["full_states"]
    s20 = rec20["full_states"]

    x10, z10 = state_xy(s10)
    x20, z20 = state_xy(s20)

    # Optional navigable floor grid
    if grid_points is not None and len(grid_points):
        ax.scatter(
            grid_points[:, 0],
            grid_points[:, 1],
            s=7,
            c=COLOR_FLOOR,
            marker="s",
            linewidths=0,
            zorder=0,
        )

    shared_end = min(
        DECISION_STEP,
        len(x10) - 1,
        len(x20) - 1,
    )

    # Shared prefix
    ax.plot(
        x10[:shared_end + 1],
        z10[:shared_end + 1],
        color=COLOR_SHARED,
        linewidth=2.3,
        solid_capstyle="round",
        label="Shared prefix",
        zorder=3,
    )

    # K10: Restore at t=10
    ax.plot(
        x10[shared_end:],
        z10[shared_end:],
        color=COLOR_K10,
        linewidth=2.0,
        solid_capstyle="round",
        label="Restore @ t=10",
        zorder=2,
    )

    # K20: delay until t=20
    ax.plot(
        x20[shared_end:],
        z20[shared_end:],
        color=COLOR_K20,
        linewidth=2.0,
        solid_capstyle="round",
        label="Delay to t=20",
        zorder=2,
    )

    # Start
    ax.scatter(
        [x10[0]],
        [z10[0]],
        s=34,
        c=COLOR_TEXT,
        marker="o",
        zorder=5,
    )

    ax.annotate(
        "Start",
        (x10[0], z10[0]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
        color=COLOR_TEXT,
    )

    # Decision state
    ax.scatter(
        [x10[shared_end]],
        [z10[shared_end]],
        s=48,
        c=COLOR_SHARED,
        marker="D",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
    )

    ax.annotate(
        r"$s_{10}$",
        (x10[shared_end], z10[shared_end]),
        xytext=(5, -10),
        textcoords="offset points",
        fontsize=7,
        color=COLOR_TEXT,
    )

    # Outcomes
    success10 = bool_value(rec10["success"])
    success20 = bool_value(rec20["success"])

    marker10 = "o" if success10 else "x"
    color10 = COLOR_SUCCESS if success10 else COLOR_FAILURE

    marker20 = "o" if success20 else "x"
    color20 = COLOR_SUCCESS if success20 else COLOR_FAILURE

    ax.scatter(
        [x10[-1]],
        [z10[-1]],
        s=55,
        c=color10,
        marker=marker10,
        linewidths=1.5,
        zorder=7,
    )

    ax.scatter(
        [x20[-1]],
        [z20[-1]],
        s=55,
        c=color20,
        marker=marker20,
        linewidths=1.5,
        zorder=7,
    )

    ax.set_aspect("equal", adjustable="datalim")

    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("z", fontsize=7)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=6,
        length=2,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    if show_legend:
        ax.legend(
            loc="best",
            fontsize=6.5,
            frameon=False,
            handlelength=2.0,
        )


def save_trajectory(case_dir, rec10, rec20, grid_points):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))

    plot_trajectory_axis(
        ax,
        rec10,
        rec20,
        grid_points=grid_points,
        show_legend=True,
    )

    success10 = bool_value(rec10["success"])
    success20 = bool_value(rec20["success"])

    text = (
        f"Restore@10: "
        f"{'Success' if success10 else 'Failure'} "
        f"({episode_length(rec10)} steps)\n"
        f"Delay@20: "
        f"{'Success' if success20 else 'Failure'} "
        f"({episode_length(rec20)} steps)"
    )

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color=COLOR_TEXT,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.84,
            pad=2.5,
        ),
    )

    fig.tight_layout()

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            case_dir / f"trajectory.{ext}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# Figure 1 draft
# ============================================================

def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))



def add_case_panel(
    fig,
    sub_spec,
    case,
    case_dir,
    rec10,
    rec20,
    grid_points,
    panel_letter,
):
    """
    Figure-1 case panel.

    Layout
    ------
    Row 1:
        large RGB t=0  ->  large RGB t=10

    Row 2:
        matched-prefix / decision-state diagram

    Row 3:
        compact paired top-down trajectory + outcomes
    """

    gs = GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=sub_spec,
        height_ratios=[0.86, 0.47, 1.20],
        hspace=0.10,
        wspace=0.055,
    )

    # --------------------------------------------------------
    # Panel title: title left, target right.
    # No scene ID in the final figure.
    # --------------------------------------------------------

    bbox = sub_spec.get_position(fig)

    fig.text(
        bbox.x0,
        bbox.y1 + 0.013,
        f"({panel_letter}) {case['panel_title']}",
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLOR_TEXT,
    )

    fig.text(
        bbox.x1,
        bbox.y1 + 0.013,
        f"Target: {case['target']}",
        fontsize=7.4,
        ha="right",
        va="bottom",
        color=COLOR_TEXT,
    )

    # --------------------------------------------------------
    # Row 1: only t=0 and t=10.
    # Larger RGB views are easier to understand than
    # three narrow t0/t5/t10 strips.
    # --------------------------------------------------------

    rgb_files = [
        ("shared_t00.png", r"$t=0$"),
        ("shared_t10.png", r"$t=10$"),
    ]

    for i, (filename, label) in enumerate(rgb_files):
        ax = fig.add_subplot(gs[0, i])

        img = read_rgb(case_dir / filename)

        ax.imshow(img)
        ax.set_axis_off()

        ax.text(
            0.5,
            1.025,
            label,
            transform=ax.transAxes,
            fontsize=7.7,
            ha="center",
            va="bottom",
            color=COLOR_TEXT,
        )

    # --------------------------------------------------------
    # Row 2: explicit matched-prefix -> s10 -> branch logic.
    # This is the conceptual core of Figure 1.
    # --------------------------------------------------------

    ax_dec = fig.add_subplot(gs[1, :])
    ax_dec.set_xlim(0, 1)
    ax_dec.set_ylim(0, 1)
    ax_dec.axis("off")

    # Shared prefix arrow
    ax_dec.annotate(
        "",
        xy=(0.49, 0.70),
        xytext=(0.13, 0.70),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.45,
            color=COLOR_SHARED,
        ),
    )

    ax_dec.text(
        0.30,
        0.91,
        "Shared prefix  ·  context suppressed",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLOR_SHARED,
    )

    # Decision state
    ax_dec.scatter(
        [0.50],
        [0.70],
        s=62,
        marker="D",
        c=COLOR_SHARED,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )

    ax_dec.text(
        0.50,
        0.47,
        r"$s_{10}$",
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLOR_TEXT,
    )

    # Restore branch
    ax_dec.annotate(
        "",
        xy=(0.27, 0.10),
        xytext=(0.47, 0.52),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.55,
            color=COLOR_K10,
        ),
    )

    # Delay branch
    ax_dec.annotate(
        "",
        xy=(0.73, 0.10),
        xytext=(0.53, 0.52),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.55,
            color=COLOR_K20,
        ),
    )

    ax_dec.text(
        0.22,
        0.02,
        "Restore @ t=10\ncontext ON",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=COLOR_K10,
        fontweight="bold",
    )

    ax_dec.text(
        0.78,
        0.02,
        "Delay to t=20\ncontext OFF until t=20",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=COLOR_K20,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Row 3: compact trajectory.
    # No axes / ticks / duplicated legend.
    # --------------------------------------------------------

    ax_traj = fig.add_subplot(gs[2, :])

    plot_trajectory_axis(
        ax_traj,
        rec10,
        rec20,
        grid_points=grid_points,
        show_legend=False,
    )

    # Hide academic plotting furniture; keep actual trajectory,
    # Start/s10 markers, footprint and endpoints.
    ax_traj.set_xlabel("")
    ax_traj.set_ylabel("")
    ax_traj.set_xticks([])
    ax_traj.set_yticks([])

    for spine in ax_traj.spines.values():
        spine.set_visible(False)

    success10 = bool_value(rec10["success"])
    success20 = bool_value(rec20["success"])

    status10 = "SUCCESS" if success10 else "FAILURE"
    status20 = "SUCCESS" if success20 else "FAILURE"

    status_color10 = (
        COLOR_SUCCESS if success10 else COLOR_FAILURE
    )
    status_color20 = (
        COLOR_SUCCESS if success20 else COLOR_FAILURE
    )

    # Outcome cards. Branch identity is the border color;
    # outcome identity is the text color.
    ax_traj.text(
        0.025,
        0.025,
        f"Restore @10   {status10}\n"
        f"{episode_length(rec10)} steps",
        transform=ax_traj.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color=status_color10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor=COLOR_K10,
            linewidth=1.0,
            alpha=0.94,
        ),
        zorder=20,
    )

    ax_traj.text(
        0.975,
        0.025,
        f"Delay @20   {status20}\n"
        f"{episode_length(rec20)} steps",
        transform=ax_traj.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.7,
        color=status_color20,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor=COLOR_K20,
            linewidth=1.0,
            alpha=0.94,
        ),
        zorder=20,
    )


def add_motivation_panel(fig, sub_spec):
    """
    Compact motivation panel.

    This panel does NOT claim that the final ACA selector has already
    predicted these two examples correctly. It only expresses the
    method motivation implied by the paired interventions.
    """

    ax = fig.add_subplot(sub_spec)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.0,
        1.013,
        "(c) Adaptive Context Activation",
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLOR_TEXT,
    )

    # Pre-decision evidence
    ax.text(
        0.5,
        0.78,
        "Pre-decision\nnavigation evidence",
        ha="center",
        va="center",
        fontsize=8.6,
        color=COLOR_TEXT,
        bbox=dict(
            boxstyle="round,pad=0.42",
            facecolor="white",
            edgecolor=COLOR_SHARED,
            linewidth=1.1,
        ),
    )

    ax.annotate(
        "",
        xy=(0.5, 0.605),
        xytext=(0.5, 0.695),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.25,
            color=COLOR_SHARED,
        ),
    )

    # ACA selector
    ax.text(
        0.5,
        0.535,
        "ACA",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLOR_TEXT,
        bbox=dict(
            boxstyle="round,pad=0.48",
            facecolor="white",
            edgecolor=COLOR_TEXT,
            linewidth=1.15,
        ),
    )

    # Two activation schedules
    ax.annotate(
        "",
        xy=(0.24, 0.315),
        xytext=(0.46, 0.465),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.55,
            color=COLOR_K10,
        ),
    )

    ax.annotate(
        "",
        xy=(0.76, 0.315),
        xytext=(0.54, 0.465),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.55,
            color=COLOR_K20,
        ),
    )

    ax.text(
        0.22,
        0.245,
        "Restore @ t=10\nContext ON",
        ha="center",
        va="center",
        fontsize=7.7,
        color=COLOR_K10,
        fontweight="bold",
    )

    ax.text(
        0.78,
        0.245,
        "Delay to t=20\nOFF until t=20",
        ha="center",
        va="center",
        fontsize=7.7,
        color=COLOR_K20,
        fontweight="bold",
    )

    # Keep the conclusion short and unambiguous.
    ax.text(
        0.5,
        0.075,
        "Select context timing from\n"
        "pre-decision evidence.",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLOR_TEXT,
    )


def make_fig1_draft(out_root, processed_cases):
    """
    Final compact Figure-1 layout.

    Approximate width:
        panel (a): 39%
        panel (b): 39%
        panel (c): 22%
    """

    fig = plt.figure(
        figsize=(14.2, 5.15),
        constrained_layout=False,
        facecolor="white",
    )

    outer = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.56],
        left=0.030,
        right=0.985,
        bottom=0.055,
        top=0.895,
        wspace=0.105,
    )

    add_case_panel(
        fig,
        outer[0, 0],
        processed_cases[0]["case"],
        processed_cases[0]["case_dir"],
        processed_cases[0]["rec10"],
        processed_cases[0]["rec20"],
        processed_cases[0]["grid_points"],
        "a",
    )

    add_case_panel(
        fig,
        outer[0, 1],
        processed_cases[1]["case"],
        processed_cases[1]["case_dir"],
        processed_cases[1]["rec10"],
        processed_cases[1]["rec20"],
        processed_cases[1]["grid_points"],
        "b",
    )

    add_motivation_panel(
        fig,
        outer[0, 2],
    )

    fig.savefig(
        out_root / "fig1_draft.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        out_root / "fig1_draft.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k10-root",
        default="results/early10_500_zsval",
    )

    parser.add_argument(
        "--k20-root",
        default="results/early20_500_zsval",
    )

    parser.add_argument(
        "--output-root",
        default="figures/fig1_final_cases",
    )

    parser.add_argument(
        "--quality",
        default="Low",
        choices=["Low", "Medium", "High", "Ultra"],
    )

    parser.add_argument(
        "--width",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    repo_root = Path.cwd()
    k10_root = Path(args.k10_root)
    k20_root = Path(args.k20_root)
    out_root = Path(args.output_root)

    out_root.mkdir(parents=True, exist_ok=True)

    log("=" * 78)
    log("FIGURE 1 FINAL CASE EXPORT")
    log("=" * 78)

    log(f"repo root : {repo_root}")
    log(f"K10 root  : {k10_root}")
    log(f"K20 root  : {k20_root}")
    log(f"output    : {out_root}")
    log(f"quality   : {args.quality}")
    log("")

    # --------------------------------------------------------
    # Locate + validate pair records BEFORE starting Unity
    # --------------------------------------------------------

    pairs = []

    for case in CASES:
        log("-" * 78)
        log(f"Locate: {case['case_id']}")

        fp10, line10, rec10 = find_episode(
            k10_root,
            case,
        )

        fp20, line20, rec20 = find_episode(
            k20_root,
            case,
        )

        log(f"  K10: {fp10}:{line10}")
        log(f"  K20: {fp20}:{line20}")

        pair_meta = validate_pair(
            rec10,
            rec20,
            case,
        )

        log("  Prefix check : PASS")
        log(
            f"  s10          : "
            f"{pair_meta['decision_state']}"
        )
        log(
            f"  K10 outcome  : "
            f"{pair_meta['k10_success']} "
            f"len={pair_meta['k10_ep_length']}"
        )
        log(
            f"  K20 outcome  : "
            f"{pair_meta['k20_success']} "
            f"len={pair_meta['k20_ep_length']}"
        )
        log(
            f"  first action difference: "
            f"{pair_meta['first_action_difference']}"
        )
        log(
            f"  first state difference : "
            f"{pair_meta['first_state_difference']}"
        )

        pairs.append({
            "case": case,
            "rec10": rec10,
            "rec20": rec20,
            "fp10": str(fp10),
            "fp20": str(fp20),
            "line10": line10,
            "line20": line20,
            "pair_meta": pair_meta,
        })

    log("")
    log("ALL PAIR VALIDATION: PASS")
    log("")

    # --------------------------------------------------------
    # Start live renderer exactly once
    # --------------------------------------------------------

    renderer = LiveRGBRenderer(
        quality=args.quality,
        width=args.width,
        height=args.height,
    )

    processed = []

    try:
        for item in pairs:
            case = item["case"]
            rec10 = item["rec10"]
            rec20 = item["rec20"]

            case_dir = out_root / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)

            log("")
            log("=" * 78)
            log(f"PROCESS {case['case_id']}")
            log("=" * 78)

            # Save exact source episode records
            with open(
                case_dir / "early10_episode.jsonl",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        rec10,
                        ensure_ascii=False,
                    ) + "\n"
                )

            with open(
                case_dir / "early20_episode.jsonl",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        rec20,
                        ensure_ascii=False,
                    ) + "\n"
                )

            # RGB
            render_meta = export_case_rgb(
                renderer,
                case_dir,
                case,
                rec10,
                rec20,
            )

            # Floor grid
            grid_points = load_grid_points(
                repo_root,
                case["scene"],
            )

            if grid_points is None:
                log("    floor grid: unavailable, trajectory only")
            else:
                log(
                    f"    floor grid: {len(grid_points)} points"
                )

            # Trajectory
            save_trajectory(
                case_dir,
                rec10,
                rec20,
                grid_points,
            )

            # Metadata
            meta = {
                "case": case,
                "source": {
                    "k10_file": item["fp10"],
                    "k10_line": item["line10"],
                    "k20_file": item["fp20"],
                    "k20_line": item["line20"],
                },
                "pair_validation": item["pair_meta"],
                "render": {
                    "quality": args.quality,
                    "width": args.width,
                    "height": args.height,
                    "frames": render_meta,
                },
            }

            with open(
                case_dir / "paired_meta.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    meta,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            processed.append({
                "case": case,
                "case_dir": case_dir,
                "rec10": rec10,
                "rec20": rec20,
                "grid_points": grid_points,
            })

            log(f"CASE EXPORT PASS: {case_dir}")

    finally:
        renderer.close()

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    csv_path = out_root / "selected_cases.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "panel_title",
                "room",
                "scene",
                "target",
                "eval_index",
                "k10_success",
                "k10_ep_length",
                "k20_success",
                "k20_ep_length",
            ],
        )

        writer.writeheader()

        for item in pairs:
            c = item["case"]
            m = item["pair_meta"]

            writer.writerow({
                "case_id": c["case_id"],
                "panel_title": c["panel_title"],
                "room": c["room"],
                "scene": c["scene"],
                "target": c["target"],
                "eval_index": c["eval_index"],
                "k10_success": m["k10_success"],
                "k10_ep_length": m["k10_ep_length"],
                "k20_success": m["k20_success"],
                "k20_ep_length": m["k20_ep_length"],
            })

    # --------------------------------------------------------
    # Automatic Figure-1 draft
    # --------------------------------------------------------

    if len(processed) == 2:
        log("")
        log("Creating combined Figure-1 draft...")

        make_fig1_draft(
            out_root,
            processed,
        )

        log(
            f"Saved: {out_root / 'fig1_draft.png'}"
        )
        log(
            f"Saved: {out_root / 'fig1_draft.pdf'}"
        )

    log("")
    log("=" * 78)
    log("FIG1 FINAL CASE EXPORT: PASS")
    log("=" * 78)

    log("")
    log("Main outputs:")
    log(f"  {out_root / 'fig1_draft.png'}")
    log(f"  {out_root / 'fig1_draft.pdf'}")
    log(f"  {out_root / 'selected_cases.csv'}")


if __name__ == "__main__":
    main()
