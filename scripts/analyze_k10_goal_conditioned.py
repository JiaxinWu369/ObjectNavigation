import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from models.object_distribution import Object_Distribution
from datasets.constants import AI2THOR_TARGET_CLASSES


DATA = Path(
    "results/k10_dynamic_dataset_100k/"
    "reversal_79.jsonl"
)

OUT = Path(
    "results/k10_goal_conditioned_100k"
)
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "Laptop",
    "LightSwitch",
]

FEATURES = [
    "prior_support_score_t10",
    "prior_support_mass_t10",
    "prior_support_score_0_10",
    "prior_support_mass_0_10",
]

classes = list(
    AI2THOR_TARGET_CLASSES[22]
)

obj_dist = Object_Distribution(
    dataset="AI2THOR",
    mode="test",
)

alpha_all = (
    obj_dist.init_prior_alpha
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)


def room_name(scene_id):
    return [
        "kitchen",
        "living_room",
        "bedroom",
        "bathroom",
    ][scene_id]


def prior_profile(scene, target):
    """
    Static room-conditioned AKGVP semantic
    relation profile P(:, target), before
    online observation updates.
    """

    sid = obj_dist.get_scene_id(scene)
    g = classes.index(target)

    alpha = alpha_all[sid].copy()

    denom = alpha.sum(
        axis=1,
        keepdims=True
    )

    denom[denom <= 0] = 1.0

    P = alpha / denom

    # Goal-conditioned column.
    p_raw = P[:, g].copy()

    # Context only.
    p_raw[g] = 0.0
    p_raw[~np.isfinite(p_raw)] = 0.0
    p_raw = np.maximum(p_raw, 0.0)

    total = float(p_raw.sum())

    if total > 0:
        p = p_raw / total
    else:
        p = np.zeros_like(p_raw)

    return sid, g, p_raw, p


def profile_stats(p):
    """
    Five predeclared semantic-profile measures.
    """

    values = np.asarray(
        p,
        dtype=np.float64,
    )

    positive = values[
        values > 0
    ]

    if len(positive) == 0:
        entropy = 0.0
    else:
        entropy = -float(
            np.sum(
                positive
                * np.log(
                    positive + 1e-12
                )
            )
        ) / math.log(21.0)

    sorted_p = np.sort(
        values
    )[::-1]

    top1 = float(sorted_p[0])
    top2 = float(sorted_p[1])
    top3 = float(
        sorted_p[:3].sum()
    )

    return {
        "prior_max":
            top1,

        "prior_entropy":
            entropy,

        "prior_top1_mass":
            top1,

        "prior_top3_mass":
            top3,

        "prior_top1_top2_gap":
            top1 - top2,
    }


def detector_vector(trace_row):
    x = np.asarray(
        trace_row["detector_scores"],
        dtype=np.float64,
    ).reshape(-1)

    if len(x) != 22:
        raise RuntimeError(
            "Expected 22 detector scores"
        )

    x[~np.isfinite(x)] = 0.0
    x = np.maximum(x, 0.0)

    return x


def extract(row):
    trace = row["trace"]

    assert len(trace) >= 11

    sid, g, p_raw, p = prior_profile(
        row["scene"],
        row["target"],
    )

    stats = profile_stats(p)

    # --------------------------------------------------
    # Current visual support at matched state t=10.
    # --------------------------------------------------

    d10 = detector_vector(
        trace[10]
    )
    d10[g] = 0.0

    current_binary = (
        d10 > 0
    ).astype(np.float64)

    # Weighted detector confidence support.
    support_score_t10 = float(
        np.dot(
            p,
            d10,
        )
    )

    # Fraction of prior semantic mass whose
    # categories are currently visually supported.
    support_mass_t10 = float(
        np.dot(
            p,
            current_binary,
        )
    )

    # --------------------------------------------------
    # Accumulated support over matched history t=0..10.
    # --------------------------------------------------

    hist = []

    for t in range(11):
        d = detector_vector(
            trace[t]
        )
        d[g] = 0.0
        hist.append(d)

    hist = np.stack(
        hist,
        axis=0,
    )

    max_score = hist.max(
        axis=0
    )

    ever_seen = (
        hist > 0
    ).any(
        axis=0
    ).astype(
        np.float64
    )

    support_score_hist = float(
        np.dot(
            p,
            max_score,
        )
    )

    support_mass_hist = float(
        np.dot(
            p,
            ever_seen,
        )
    )

    stats.update({
        "scene_prior_id":
            int(sid),

        "scene_prior_name":
            room_name(sid),

        "goal_index":
            int(g),

        "prior_context_raw_mass":
            float(p_raw.sum()),

        "prior_support_score_t10":
            support_score_t10,

        "prior_support_mass_t10":
            support_mass_t10,

        "prior_support_score_0_10":
            support_score_hist,

        "prior_support_mass_0_10":
            support_mass_hist,
    })

    return stats


def auc_score(y, x):
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)

    pos = y == 1

    n1 = int(pos.sum())
    n0 = int((~pos).sum())

    if n1 == 0 or n0 == 0:
        return float("nan")

    ranks = rankdata(
        x,
        method="average",
    )

    u = (
        ranks[pos].sum()
        -
        n1 * (n1 + 1) / 2
    )

    return float(
        u / (n1 * n0)
    )


rows = []

with open(
    DATA,
    encoding="utf-8",
) as f:

    for line in f:
        if not line.strip():
            continue

        r = json.loads(line)

        feat = extract(r)

        out = {
            "scene_type":
                r["scene_type"],

            "eval_index":
                int(r["eval_index"]),

            "scene":
                r["scene"],

            "target":
                r["target"],

            "restore_label":
                int(r["restore_label"]),

            "outcome":
                r["outcome"],
        }

        out.update(feat)

        rows.append(out)


assert len(rows) == 79


# ------------------------------------------------------
# Save episode-level features.
# ------------------------------------------------------

fields = list(
    rows[0].keys()
)

with open(
    OUT / "goal_conditioned_79.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )
    w.writeheader()
    w.writerows(rows)


# ------------------------------------------------------
# Static goal-profile table for all six unseen targets
# and four room types.
# ------------------------------------------------------

unseen = [
    "Bowl",
    "DeskLamp",
    "Laptop",
    "LightSwitch",
    "Plate",
    "StoveBurner",
]

representative_scene = {
    "kitchen": "FloorPlan21",
    "living_room": "FloorPlan221",
    "bedroom": "FloorPlan321",
    "bathroom": "FloorPlan421",
}

profile_rows = []

for room, scene in representative_scene.items():

    for target in unseen:

        sid, g, p_raw, p = (
            prior_profile(
                scene,
                target,
            )
        )

        s = profile_stats(p)

        profile_rows.append({
            "room": room,
            "target": target,
            "goal_index": g,
            "prior_context_raw_mass":
                float(p_raw.sum()),
            **s,
        })


with open(
    OUT / "goal_profiles_24.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            profile_rows[0].keys()
        ),
    )
    w.writeheader()
    w.writerows(profile_rows)


# ------------------------------------------------------
# Primary within-target check.
# ------------------------------------------------------

print("=" * 100)
print("GOAL-CONDITIONED CONTEXT AGREEMENT")
print("=" * 100)

stats_out = []

for target in TARGETS:

    group = [
        r for r in rows
        if r["target"] == target
    ]

    y = np.asarray([
        r["restore_label"]
        for r in group
    ])

    print()
    print("=" * 80)
    print(target)
    print("=" * 80)

    print(
        "N={} help={} harm={}".format(
            len(group),
            int(y.sum()),
            int((y == 0).sum()),
        )
    )

    print(
        f"{'Feature':<32}"
        f"{'HelpMean':>12}"
        f"{'HarmMean':>12}"
        f"{'AUC':>10}"
    )

    print("-" * 66)

    for feature in FEATURES:

        x = np.asarray([
            float(r[feature])
            for r in group
        ])

        auc = auc_score(
            y,
            x,
        )

        hm = float(
            x[y == 1].mean()
        )

        xm = float(
            x[y == 0].mean()
        )

        print(
            f"{feature:<32}"
            f"{hm:>12.4f}"
            f"{xm:>12.4f}"
            f"{auc:>10.3f}"
        )

        stats_out.append({
            "target": target,
            "feature": feature,
            "help_mean": hm,
            "harm_mean": xm,
            "auc": auc,
        })


print()
print("=" * 100)
print("CROSS-TARGET CHECK")
print("=" * 100)

for feature in FEATURES:

    a = next(
        x for x in stats_out
        if x["target"] == "Laptop"
        and x["feature"] == feature
    )

    b = next(
        x for x in stats_out
        if x["target"] == "LightSwitch"
        and x["feature"] == feature
    )

    aa = a["auc"]
    bb = b["auc"]

    same = (
        (aa - 0.5)
        * (bb - 0.5)
        > 0
    )

    weakest = min(
        abs(aa - 0.5),
        abs(bb - 0.5),
    )

    print(
        "{:<32} "
        "Laptop={:.3f} "
        "LightSwitch={:.3f} "
        "same={} "
        "min_effect={:.3f}".format(
            feature,
            aa,
            bb,
            int(same),
            weakest,
        )
    )


print()
print(
    "Saved:",
    OUT / "goal_conditioned_79.csv"
)

print(
    "Saved:",
    OUT / "goal_profiles_24.csv"
)

print()
print(
    "GOAL-CONDITIONED ANALYSIS: PASS"
)
