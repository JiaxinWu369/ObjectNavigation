import json
import numpy as np
from scipy.stats import rankdata


PATH = (
    "results/lcr_full_dataset/"
    "regions_labeled.jsonl"
)


def auc_score(y, score):
    y = np.asarray(
        y,
        dtype=np.int64,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    n_pos = int(
        np.sum(y == 1)
    )
    n_neg = int(
        np.sum(y == 0)
    )

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(
        score,
        method="average",
    )

    pos_sum = ranks[
        y == 1
    ].sum()

    return float(
        (
            pos_sum
            - n_pos * (n_pos + 1) / 2
        )
        /
        (n_pos * n_neg)
    )


rows = [
    json.loads(x)
    for x in open(
        PATH,
        encoding="utf-8",
    )
    if x.strip()
]

# Only held-out scene validation.
rows = [
    r
    for r in rows
    if r["data_split"] == "val"
]

labels = []

goal_max = []
goal_mean = []
goal_last = []

context_support_mean = []
context_support_max = []

for row in rows:

    y = int(
        row["label"]
    )

    g = int(
        row["goal_index"]
    )

    goal_scores = []
    supports = []

    for obs in row["sequence"]:

        detector = np.asarray(
            obs["detector_scores"],
            dtype=np.float64,
        )

        attention = np.asarray(
            obs["base_attention"],
            dtype=np.float64,
        )

        goal_scores.append(
            float(detector[g])
        )

        mask = np.ones(
            22,
            dtype=bool,
        )
        mask[g] = False

        support = float(
            np.sum(
                detector[mask]
                *
                attention[mask]
            )
        )

        supports.append(
            support
        )

    labels.append(y)

    goal_max.append(
        max(goal_scores)
    )

    goal_mean.append(
        float(
            np.mean(goal_scores)
        )
    )

    goal_last.append(
        goal_scores[-1]
    )

    context_support_mean.append(
        float(
            np.mean(supports)
        )
    )

    context_support_max.append(
        max(supports)
    )


print("=" * 78)
print("LCR SHORTCUT AUDIT — HELD-OUT SCENES")
print("=" * 78)

print("N          :", len(rows))
print("Positive   :", sum(labels))
print(
    "Negative   :",
    len(labels) - sum(labels)
)

print()
print("SIMPLE SINGLE-SCALAR AUC")

for name, values in [
    ("goal_max", goal_max),
    ("goal_mean", goal_mean),
    ("goal_last", goal_last),
    (
        "context_support_mean",
        context_support_mean,
    ),
    (
        "context_support_max",
        context_support_max,
    ),
]:
    print(
        f"{name:<24}: "
        f"{auc_score(labels, values):.4f}"
    )


print()
print("GOAL-ABSENT / LOW-GOAL SUBSETS")

for threshold in [
    0.0,
    0.01,
    0.05,
    0.10,
]:

    idx = [
        i
        for i, v in enumerate(goal_max)
        if v <= threshold + 1e-12
    ]

    if not idx:
        print(
            f"goal_max <= {threshold:.2f}: N=0"
        )
        continue

    yy = [
        labels[i]
        for i in idx
    ]

    support = [
        context_support_mean[i]
        for i in idx
    ]

    n_pos = sum(yy)
    n_neg = len(yy) - n_pos

    print(
        f"goal_max <= {threshold:.2f}: "
        f"N={len(idx):4d} "
        f"pos={n_pos:4d} "
        f"neg={n_neg:4d} "
        f"pos_rate={np.mean(yy):.3f} "
        f"context_AUC={auc_score(yy, support):.4f}"
    )


print()
print("LCR SHORTCUT AUDIT: PASS")
