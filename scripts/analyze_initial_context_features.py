import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from datasets.constants import AI2THOR_TARGET_CLASSES


ROOT = Path("results/context_reversal_100k")
DATA_ROOT = Path("datasets/Scene_Data")

CLASSES = list(AI2THOR_TARGET_CLASSES[22])

GROUP_FILES = {
    "suppress_only": ROOT / "suppress_only.csv",
    "base_only": ROOT / "base_only.csv",
}


def state_key(row):
    x = float(row["start_x"])
    z = float(row["start_z"])

    if abs(x) < 0.005:
        x = 0.0
    if abs(z) < 0.005:
        z = 0.0

    rot = int(round(float(row["start_rotation"])))
    hor = int(round(float(row["start_horizon"])))

    return f"{x:.2f}|{z:.2f}|{rot}|{hor}"


def entropy_from_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[scores > 0]

    if len(scores) <= 1:
        return 0.0

    p = scores / scores.sum()

    return float(
        -(p * np.log(p + 1e-12)).sum()
    )


def extract_features(row):
    scene = row["scene"]
    target = row["target"]
    state = state_key(row)

    target_idx = CLASSES.index(target)

    p = (
        DATA_ROOT
        / scene
        / "det_feature_22_cates.hdf5"
    )

    with h5py.File(p, "r") as f:
        if state not in f:
            raise KeyError(
                f"Missing HDF5 state: "
                f"{scene} {state}"
            )

        det = f[state][()]

    if det.shape != (22, 517):
        raise RuntimeError(
            f"Unexpected shape "
            f"{det.shape}: {scene} {state}"
        )

    scores = np.asarray(
        det[:, -1],
        dtype=np.float64,
    )

    goal_score = float(scores[target_idx])

    mask = np.ones(
        len(CLASSES),
        dtype=bool,
    )
    mask[target_idx] = False

    ctx = scores[mask]
    positive = ctx[ctx > 0]

    if len(positive) > 0:
        ctx_max = float(positive.max())
        ctx_mean_detected = float(
            positive.mean()
        )
        ctx_sum = float(
            positive.sum()
        )
    else:
        ctx_max = 0.0
        ctx_mean_detected = 0.0
        ctx_sum = 0.0

    sorted_ctx = np.sort(positive)[::-1]

    if len(sorted_ctx) >= 2:
        ctx_top_gap = float(
            sorted_ctx[0] - sorted_ctx[1]
        )
    elif len(sorted_ctx) == 1:
        ctx_top_gap = float(sorted_ctx[0])
    else:
        ctx_top_gap = 0.0

    # Rank 1 = largest detector score.
    rank_order = (
        np.argsort(-scores).tolist()
    )

    goal_rank = (
        rank_order.index(target_idx) + 1
    )

    eps = 1e-6

    return {
        "scene": scene,
        "scene_type": row["scene_type"],
        "eval_index": int(row["eval_index"]),
        "target": target,
        "state": state,

        "goal_score": goal_score,
        "goal_detected":
            int(goal_score > 0),

        "goal_rank":
            goal_rank,

        "ctx_detect_count":
            int((ctx > 0).sum()),

        "ctx_score_max":
            ctx_max,

        "ctx_score_mean_detected":
            ctx_mean_detected,

        "ctx_score_mean_all":
            float(ctx.mean()),

        "ctx_score_sum":
            ctx_sum,

        "ctx_top1_top2_gap":
            ctx_top_gap,

        "ctx_above_05":
            int((ctx >= 0.5).sum()),

        "ctx_above_08":
            int((ctx >= 0.8).sum()),

        "ctx_minus_goal":
            ctx_max - goal_score,

        "log_ctx_goal_ratio":
            float(
                math.log(
                    (ctx_max + eps)
                    /
                    (goal_score + eps)
                )
            ),

        "ctx_score_entropy":
            entropy_from_scores(
                positive
            ),
    }


rows = []

for group, path in GROUP_FILES.items():
    df = pd.read_csv(path)

    print(
        f"loading {group}: {len(df)}"
    )

    for _, row in df.iterrows():
        feat = extract_features(row)
        feat["group"] = group

        rows.append(feat)


out = pd.DataFrame(rows)

out_path = (
    ROOT
    / "initial_context_features.csv"
)

out.to_csv(
    out_path,
    index=False,
)

print()
print(
    "saved:",
    out_path,
)
print("N =", len(out))


FEATURES = [
    "goal_score",
    "goal_detected",
    "goal_rank",
    "ctx_detect_count",
    "ctx_score_max",
    "ctx_score_mean_detected",
    "ctx_score_mean_all",
    "ctx_score_sum",
    "ctx_top1_top2_gap",
    "ctx_above_05",
    "ctx_above_08",
    "ctx_minus_goal",
    "log_ctx_goal_ratio",
    "ctx_score_entropy",
]


def compare_groups(
    df,
    title,
):
    sup = df[
        df["group"]
        == "suppress_only"
    ]

    base = df[
        df["group"]
        == "base_only"
    ]

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)

    print(
        f"N suppress-only={len(sup)}  "
        f"base-only={len(base)}"
    )

    print(
        f"{'Feature':<26}"
        f"{'Sup mean':>12}"
        f"{'Base mean':>12}"
        f"{'Sup med':>12}"
        f"{'Base med':>12}"
        f"{'AUC':>10}"
        f"{'RBC':>10}"
        f"{'p':>14}"
    )

    print("-" * 118)

    for feature in FEATURES:
        x = (
            sup[feature]
            .astype(float)
            .to_numpy()
        )

        y = (
            base[feature]
            .astype(float)
            .to_numpy()
        )

        if len(x) == 0 or len(y) == 0:
            continue

        u, p = mannwhitneyu(
            x,
            y,
            alternative="two-sided",
        )

        auc = (
            float(u)
            /
            (len(x) * len(y))
        )

        # Rank-biserial correlation.
        # Positive means feature tends
        # to be larger in suppress-only.
        rbc = 2.0 * auc - 1.0

        print(
            f"{feature:<26}"
            f"{x.mean():>12.4f}"
            f"{y.mean():>12.4f}"
            f"{np.median(x):>12.4f}"
            f"{np.median(y):>12.4f}"
            f"{auc:>10.3f}"
            f"{rbc:>10.3f}"
            f"{p:>14.6g}"
        )


compare_groups(
    out,
    "POOLED REVERSAL FEATURE COMPARISON",
)


# -------------------------------------------------
# Within-target analysis.
#
# Only print strata with enough examples from
# BOTH directions. This avoids tiny-group claims.
# -------------------------------------------------

for target in sorted(
    out["target"].unique()
):
    sub = out[
        out["target"] == target
    ]

    ns = int(
        (
            sub["group"]
            == "suppress_only"
        ).sum()
    )

    nb = int(
        (
            sub["group"]
            == "base_only"
        ).sum()
    )

    if ns >= 8 and nb >= 8:
        compare_groups(
            sub,
            f"WITHIN TARGET: {target}",
        )


# -------------------------------------------------
# Within-room analysis.
# Diagnostic only because target and room can
# be strongly correlated.
# -------------------------------------------------

for room in sorted(
    out["scene_type"].unique()
):
    sub = out[
        out["scene_type"] == room
    ]

    ns = int(
        (
            sub["group"]
            == "suppress_only"
        ).sum()
    )

    nb = int(
        (
            sub["group"]
            == "base_only"
        ).sum()
    )

    if ns >= 8 and nb >= 8:
        compare_groups(
            sub,
            f"WITHIN ROOM: {room}",
        )
