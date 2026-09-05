import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageDraw

from datasets.environment import Environment
from datasets.offline_controller_with_small_rotation import ThorAgentState


CSV_FILE = Path(
    "figures/fig1_candidates/"
    "fig1_candidate_cases.csv"
)

OUT_ROOT = Path(
    "figures/fig1_candidates/exported"
)

TOP_N = 6


# ---------------------------------------------------------
# Load episode results
# ---------------------------------------------------------

ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

ROOT10 = Path(
    "results/early10_500_zsval"
)

ROOT20 = Path(
    "results/early20_500_zsval"
)


def load_results(root):

    out = {}

    for room in ROOMS:

        p = root / f"episodes_{room}.jsonl"

        with open(
            p,
            encoding="utf-8",
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                r = json.loads(line)

                key = (
                    room,
                    int(r["eval_index"]),
                )

                out[key] = r

    return out


R10 = load_results(ROOT10)
R20 = load_results(ROOT20)


# ---------------------------------------------------------
# Load ranked candidates
# ---------------------------------------------------------

with open(
    CSV_FILE,
    encoding="utf-8",
) as f:

    candidates = list(
        csv.DictReader(f)
    )


selected = []

for kind in [
    "restore_help",
    "delay_help",
]:

    group = [
        x for x in candidates
        if x["case_type"] == kind
    ]

    group = sorted(
        group,
        key=lambda x:
            -float(x["visual_score"])
    )

    selected.extend(
        group[:TOP_N]
    )


# ---------------------------------------------------------
# AI2-THOR offline environment
# ---------------------------------------------------------

env = Environment(
    use_offline_controller=True,
    grid_size=0.25,
    fov=100.0,
    offline_data_dir=
        "datasets/Scene_Data",

    detection_feature_file_name=
        "det_feature_22_cates.hdf5",

    images_file_name=
        "clip_featuremap.hdf5",

    visible_object_map_file_name=
        "visible_object_map_1.5.json",

    optimal_action_file_name=
        "optimal_action.json",
)


current_scene = None


def set_scene(scene):
    global current_scene

    if current_scene is None:
        env.start(scene)
        current_scene = scene

    elif current_scene != scene:
        env.reset(scene)
        current_scene = scene


def parse_state(s):

    p = str(s).split("|")

    if len(p) != 4:
        raise ValueError(
            f"Bad state: {s}"
        )

    x = float(p[0])
    z = float(p[1])
    rotation = float(p[2])
    horizon = float(p[3])

    return (
        x,
        z,
        rotation,
        horizon,
    )


def frame_from_state(scene, state):

    set_scene(scene)

    x, z, rotation, horizon = (
        parse_state(state)
    )

    # Preserve the scene's current y.
    y = env.controller.state.y

    env.controller.state = (
        ThorAgentState(
            x,
            y,
            z,
            rotation,
            horizon,
        )
    )

    # Refresh offline cached event.
    env.controller.last_action_success = True

    env.controller.last_event = (
        env.controller._successful_event()
    )

    frame = np.asarray(
        env.current_frame
    )

    if frame is None:
        raise RuntimeError(
            f"No frame: {scene} {state}"
        )

    return frame.astype(
        np.uint8
    )


def save_rgb(scene, state, path):

    frame = frame_from_state(
        scene,
        state,
    )

    Image.fromarray(
        frame
    ).save(path)


def xy(states):

    out = []

    for s in states:

        x, z, _, _ = (
            parse_state(s)
        )

        out.append(
            (x, z)
        )

    return np.asarray(
        out,
        dtype=float,
    )


def reachable_points(scene):

    set_scene(scene)

    pts = []

    # The offline image keys correspond to
    # valid agent states.
    for s in env.controller.images.keys():

        try:
            x, z, _, _ = (
                parse_state(s)
            )

            pts.append(
                (x, z)
            )

        except Exception:
            pass

    if not pts:
        return np.empty(
            (0, 2)
        )

    pts = np.asarray(
        pts,
        dtype=float,
    )

    # Multiple orientations map to the same x,z.
    pts = np.unique(
        pts,
        axis=0,
    )

    return pts


def save_trajectory(
    scene,
    states10,
    states20,
    out_png,
    out_svg,
):

    a = xy(states10)
    b = xy(states20)

    bg = reachable_points(scene)

    fig, ax = plt.subplots(
        figsize=(4.4, 4.0)
    )

    if len(bg):
        ax.scatter(
            bg[:, 0],
            bg[:, 1],
            s=7,
            c="#E8EAED",
            edgecolors="none",
            zorder=0,
        )

    # Shared prefix: state 0 ... state 10.
    shared = a[:11]

    ax.plot(
        shared[:, 0],
        shared[:, 1],
        linewidth=3.0,
        color="#66788A",
        zorder=3,
    )

    # Restore@10
    ax.plot(
        a[10:, 0],
        a[10:, 1],
        linewidth=2.6,
        color="#E69F00",
        zorder=2,
    )

    # Delay@20
    ax.plot(
        b[10:, 0],
        b[10:, 1],
        linewidth=2.6,
        color="#4C78A8",
        zorder=2,
    )

    # Start
    ax.scatter(
        [shared[0, 0]],
        [shared[0, 1]],
        s=70,
        c="#222222",
        marker="o",
        zorder=5,
    )

    # Decision state
    ax.scatter(
        [shared[-1, 0]],
        [shared[-1, 1]],
        s=95,
        c="#FFFFFF",
        edgecolors="#222222",
        linewidths=1.6,
        marker="o",
        zorder=6,
    )

    # Final states
    ax.scatter(
        [a[-1, 0]],
        [a[-1, 1]],
        s=70,
        c="#E69F00",
        marker="s",
        zorder=5,
    )

    ax.scatter(
        [b[-1, 0]],
        [b[-1, 1]],
        s=70,
        c="#4C78A8",
        marker="s",
        zorder=5,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(
        pad=0.05
    )

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        out_svg,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)


def panel(
    image,
    title,
    size=(360, 240),
):

    img = Image.open(image).convert(
        "RGB"
    )

    img = ImageOps.fit(
        img,
        size,
        method=Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (
            size[0],
            size[1] + 32,
        ),
        "white",
    )

    canvas.paste(
        img,
        (0, 32),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (8, 8),
        title,
        fill="black",
    )

    return canvas


def make_contact_sheet(folder):

    items = [
        (
            folder / "rgb_t00.png",
            "Shared t=0",
        ),
        (
            folder / "rgb_t05.png",
            "Shared t=5",
        ),
        (
            folder / "rgb_t10.png",
            "Decision t=10",
        ),
        (
            folder / "rgb_k10_final.png",
            "Restore@10 final",
        ),
        (
            folder / "rgb_k20_final.png",
            "Delay@20 final",
        ),
        (
            folder / "trajectory.png",
            "Trajectory",
        ),
    ]

    panels = [
        panel(p, title)
        for p, title in items
    ]

    w = 360
    h = 272

    canvas = Image.new(
        "RGB",
        (3 * w, 2 * h),
        "white",
    )

    for i, img in enumerate(
        panels
    ):
        x = (i % 3) * w
        y = (i // 3) * h

        canvas.paste(
            img,
            (x, y),
        )

    canvas.save(
        folder /
        "contact_sheet.jpg",
        quality=94,
    )


# ---------------------------------------------------------
# Export
# ---------------------------------------------------------

for rank, row in enumerate(
    selected,
    1,
):

    kind = row[
        "case_type"
    ]

    room = row["room"]
    idx = int(
        row["eval_index"]
    )

    key = (
        room,
        idx,
    )

    a = R10[key]
    b = R20[key]

    scene = a["scene"]
    target = a["target"]

    folder = (
        OUT_ROOT
        /
        kind
        /
        (
            f"{scene}_"
            f"{target}_"
            f"idx{idx:04d}"
        )
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    s10 = a["full_states"]
    s20 = b["full_states"]

    assert (
        a["full_actions"][:10]
        ==
        b["full_actions"][:10]
    )

    assert s10[10] == s20[10]

    # Shared RGB observations.
    save_rgb(
        scene,
        s10[0],
        folder / "rgb_t00.png",
    )

    save_rgb(
        scene,
        s10[5],
        folder / "rgb_t05.png",
    )

    save_rgb(
        scene,
        s10[10],
        folder / "rgb_t10.png",
    )

    # Branch outcomes.
    save_rgb(
        scene,
        s10[-1],
        folder /
        "rgb_k10_final.png",
    )

    save_rgb(
        scene,
        s20[-1],
        folder /
        "rgb_k20_final.png",
    )

    save_trajectory(
        scene,
        s10,
        s20,
        folder /
        "trajectory.png",
        folder /
        "trajectory.svg",
    )

    with open(
        folder / "meta.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                **row,
                "k10_final_state":
                    s10[-1],
                "k20_final_state":
                    s20[-1],
            },
            f,
            indent=2,
        )

    make_contact_sheet(
        folder
    )

    print(
        kind,
        scene,
        target,
        idx,
        "->",
        folder,
    )


print()
print(
    "Exported:",
    len(selected),
    "candidate cases"
)

print(
    "FIG1 CANDIDATE EXPORT: PASS"
)
