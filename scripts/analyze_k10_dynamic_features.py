import csv
import json
import math
import zlib
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.stats import rankdata


DATA_FILE = Path(
    "results/k10_dynamic_dataset_100k/"
    "reversal_79.jsonl"
)

OUT_DIR = Path(
    "results/k10_dynamic_features_100k"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAIN_TARGETS = [
    "Laptop",
    "LightSwitch",
]

FEATURES = [
    "ctx_att_max_t10",
    "ctx_att_entropy_t10",
    "ctx_att_l1_recent5",
    "top_ctx_persistence_recent5",
    "det_jaccard_recent5",
    "unique_ctx_seen_0_10",
    "goal_score_max_0_10",
    "goal_seen_count_0_10",
]

N_BOOT = 10000
N_PERM = 10000
BASE_SEED = 20260819


# ================================================================
# Basic feature helpers
# ================================================================

def context_attention(
    trace_row
):
    """
    Return 21-dimensional-equivalent context vector
    represented in original 22-d indexing, with goal
    category explicitly zeroed.
    """

    a = np.asarray(
        trace_row["base_attention"],
        dtype=np.float64,
    ).reshape(-1)

    if len(a) != 22:
        raise RuntimeError(
            f"Expected 22 attention values, got {len(a)}"
        )

    goal = int(
        trace_row["goal_index"]
    )

    a = a.copy()

    # Numerical safety.
    a[~np.isfinite(a)] = 0.0
    a = np.maximum(a, 0.0)

    a[goal] = 0.0

    return a


def normalized_context_attention(
    trace_row
):
    a = context_attention(
        trace_row
    )

    s = float(a.sum())

    if s <= 0:
        return np.zeros_like(a)

    return a / s


def normalized_entropy(
    x
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
        & (x > 0)
    ]

    if len(x) == 0:
        return 0.0

    p = x / x.sum()

    h = -float(
        np.sum(
            p * np.log(p + 1e-12)
        )
    )

    # Normalize to [0,1] using the number
    # of possible non-target categories = 21.
    denom = math.log(21.0)

    if denom <= 0:
        return 0.0

    return h / denom


def detected_context_set(
    trace_row
):
    scores = np.asarray(
        trace_row["detector_scores"],
        dtype=np.float64,
    ).reshape(-1)

    if len(scores) != 22:
        raise RuntimeError(
            f"Expected 22 detector scores, got {len(scores)}"
        )

    goal = int(
        trace_row["goal_index"]
    )

    return {
        int(i)
        for i, score
        in enumerate(scores)
        if (
            i != goal
            and np.isfinite(score)
            and score > 0
        )
    }


def jaccard(
    a,
    b,
):
    a = set(a)
    b = set(b)

    if not a and not b:
        # Same observation in the sense that
        # neither contains detected context.
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return (
        len(a & b)
        /
        len(union)
    )


# ================================================================
# Extract the 8 predeclared features
# ================================================================

def extract_features(
    row
):
    trace = row["trace"]

    if len(trace) < 11:
        raise RuntimeError(
            "Reversal episode does not contain t=10."
        )

    # Ensure exact chronological trace.
    for t in range(11):
        if int(trace[t]["step"]) != t:
            raise RuntimeError(
                f"Invalid trace ordering: "
                f"expected {t}, got "
                f"{trace[t]['step']}"
            )

    t10 = trace[10]

    goal = int(
        t10["goal_index"]
    )

    # ------------------------------------------------------------
    # F1: current context semantic strength
    # ------------------------------------------------------------

    ctx10 = context_attention(
        t10
    )

    ctx_att_max_t10 = float(
        ctx10.max()
    )


    # ------------------------------------------------------------
    # F2: current context attention entropy
    # ------------------------------------------------------------

    ctx_att_entropy_t10 = (
        normalized_entropy(
            ctx10
        )
    )


    # ------------------------------------------------------------
    # F3:
    # mean L1 change of normalized context attention
    # across states t=6,7,8,9,10.
    # Four transitions.
    # ------------------------------------------------------------

    recent_attention = [
        normalized_context_attention(
            trace[t]
        )
        for t in range(6, 11)
    ]

    l1_changes = [
        float(
            np.abs(
                recent_attention[i]
                -
                recent_attention[i - 1]
            ).sum()
        )
        for i in range(
            1,
            len(recent_attention)
        )
    ]

    ctx_att_l1_recent5 = float(
        np.mean(l1_changes)
    )


    # ------------------------------------------------------------
    # F4:
    # persistence of the most common top context category
    # during t=6..10.
    #
    # Denominator stays 5. A state without semantic context
    # contributes no top-context vote.
    # ------------------------------------------------------------

    top_contexts = []

    for t in range(6, 11):

        a = context_attention(
            trace[t]
        )

        if float(a.sum()) <= 0:
            top_contexts.append(None)
        else:
            top_contexts.append(
                int(np.argmax(a))
            )

    valid_top = [
        x
        for x in top_contexts
        if x is not None
    ]

    if len(valid_top) == 0:
        top_ctx_persistence_recent5 = 0.0
    else:
        counts = Counter(
            valid_top
        )

        top_ctx_persistence_recent5 = (
            max(counts.values())
            / 5.0
        )


    # ------------------------------------------------------------
    # F5:
    # visual context-set stability t=6..10.
    # ------------------------------------------------------------

    recent_sets = [
        detected_context_set(
            trace[t]
        )
        for t in range(6, 11)
    ]

    det_jaccards = [
        jaccard(
            recent_sets[i - 1],
            recent_sets[i],
        )
        for i in range(
            1,
            len(recent_sets)
        )
    ]

    det_jaccard_recent5 = float(
        np.mean(det_jaccards)
    )


    # ------------------------------------------------------------
    # F6:
    # accumulated context categories seen in states 0..10.
    # ------------------------------------------------------------

    union_context = set()

    for t in range(11):
        union_context.update(
            detected_context_set(
                trace[t]
            )
        )

    unique_ctx_seen_0_10 = int(
        len(union_context)
    )


    # ------------------------------------------------------------
    # F7/F8:
    # accumulated direct target evidence.
    # ------------------------------------------------------------

    goal_scores = []

    for t in range(11):

        scores = np.asarray(
            trace[t]["detector_scores"],
            dtype=np.float64,
        ).reshape(-1)

        current_goal = int(
            trace[t]["goal_index"]
        )

        if current_goal != goal:
            raise RuntimeError(
                "Goal index changed inside one episode."
            )

        score = float(
            scores[goal]
        )

        if not np.isfinite(score):
            score = 0.0

        goal_scores.append(
            score
        )

    goal_score_max_0_10 = float(
        max(goal_scores)
    )

    goal_seen_count_0_10 = int(
        sum(
            score > 0
            for score in goal_scores
        )
    )


    return {
        "ctx_att_max_t10":
            ctx_att_max_t10,

        "ctx_att_entropy_t10":
            ctx_att_entropy_t10,

        "ctx_att_l1_recent5":
            ctx_att_l1_recent5,

        "top_ctx_persistence_recent5":
            top_ctx_persistence_recent5,

        "det_jaccard_recent5":
            det_jaccard_recent5,

        "unique_ctx_seen_0_10":
            unique_ctx_seen_0_10,

        "goal_score_max_0_10":
            goal_score_max_0_10,

        "goal_seen_count_0_10":
            goal_seen_count_0_10,
    }


# ================================================================
# Statistics
# ================================================================

def auc_score(
    y,
    x,
):
    """
    AUC:
      >0.5: higher feature -> restore-help
      <0.5: higher feature -> restore-harm
    """

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    pos = (
        y == 1
    )

    n_pos = int(
        pos.sum()
    )

    n_neg = int(
        (~pos).sum()
    )

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(
        x,
        method="average",
    )

    rank_sum_pos = float(
        ranks[pos].sum()
    )

    u = (
        rank_sum_pos
        -
        n_pos
        * (n_pos + 1)
        / 2.0
    )

    return float(
        u
        /
        (n_pos * n_neg)
    )


def bootstrap_auc_ci(
    y,
    x,
    seed,
):
    y = np.asarray(
        y,
        dtype=np.int64,
    )

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    xp = x[
        y == 1
    ]

    xn = x[
        y == 0
    ]

    rng = np.random.default_rng(
        seed
    )

    values = np.empty(
        N_BOOT,
        dtype=np.float64,
    )

    for i in range(
        N_BOOT
    ):
        bp = rng.choice(
            xp,
            size=len(xp),
            replace=True,
        )

        bn = rng.choice(
            xn,
            size=len(xn),
            replace=True,
        )

        by = np.concatenate(
            [
                np.ones(
                    len(bp),
                    dtype=np.int64,
                ),
                np.zeros(
                    len(bn),
                    dtype=np.int64,
                ),
            ]
        )

        bx = np.concatenate(
            [
                bp,
                bn,
            ]
        )

        values[i] = auc_score(
            by,
            bx,
        )

    lo, hi = np.quantile(
        values,
        [
            0.025,
            0.975,
        ]
    )

    return (
        float(lo),
        float(hi),
    )


def permutation_auc_p(
    y,
    x,
    seed,
):
    """
    Two-sided permutation test around AUC=0.5.
    """

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    observed = auc_score(
        y,
        x,
    )

    obs_effect = abs(
        observed - 0.5
    )

    rng = np.random.default_rng(
        seed
    )

    extreme = 0

    for _ in range(
        N_PERM
    ):
        yp = rng.permutation(
            y
        )

        ap = auc_score(
            yp,
            x,
        )

        if (
            abs(ap - 0.5)
            >=
            obs_effect - 1e-12
        ):
            extreme += 1

    return (
        extreme + 1
    ) / (
        N_PERM + 1
    )


def bh_fdr(
    p_values
):
    """
    Benjamini-Hochberg adjusted p-values.
    """

    p = np.asarray(
        p_values,
        dtype=np.float64,
    )

    m = len(p)

    order = np.argsort(
        p
    )

    adjusted = np.empty(
        m,
        dtype=np.float64,
    )

    prev = 1.0

    for i in range(
        m - 1,
        -1,
        -1,
    ):
        idx = order[i]

        rank = i + 1

        q = (
            p[idx]
            * m
            / rank
        )

        prev = min(
            prev,
            q,
        )

        adjusted[idx] = min(
            prev,
            1.0,
        )

    return adjusted


# ================================================================
# Load and extract
# ================================================================

rows = []

with open(
    DATA_FILE,
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        r = json.loads(line)

        label = int(
            r["restore_label"]
        )

        assert label in (
            0,
            1,
        )

        feat = extract_features(
            r
        )

        row = {
            "scene_type":
                r["scene_type"],

            "eval_index":
                int(r["eval_index"]),

            "scene":
                r["scene"],

            "target":
                r["target"],

            "restore_label":
                label,

            "outcome":
                r["outcome"],
        }

        row.update(
            feat
        )

        rows.append(
            row
        )


assert len(rows) == 79


# ================================================================
# Save extracted feature matrix
# ================================================================

feature_csv = (
    OUT_DIR
    / "k10_dynamic_features_79.csv"
)

fieldnames = [
    "scene_type",
    "eval_index",
    "scene",
    "target",
    "restore_label",
    "outcome",
] + FEATURES

with open(
    feature_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ================================================================
# Analyze Laptop, LightSwitch, and pooled exploratory set
# ================================================================

groups = {
    "Laptop": [
        r
        for r in rows
        if r["target"] == "Laptop"
    ],

    "LightSwitch": [
        r
        for r in rows
        if r["target"] == "LightSwitch"
    ],

    # Exploratory only.
    "Pooled79": rows,
}


stats_rows = []

for group_name, group_rows in groups.items():

    y = np.asarray(
        [
            int(r["restore_label"])
            for r in group_rows
        ],
        dtype=np.int64,
    )

    help_n = int(
        y.sum()
    )

    harm_n = int(
        len(y) - y.sum()
    )

    print()
    print("=" * 100)
    print(group_name)
    print("=" * 100)

    print(
        "N={} help={} harm={}".format(
            len(y),
            help_n,
            harm_n,
        )
    )

    print()

    group_stats = []

    for feature in FEATURES:

        x = np.asarray(
            [
                float(r[feature])
                for r in group_rows
            ],
            dtype=np.float64,
        )

        auc = auc_score(
            y,
            x,
        )

        seed = (
            BASE_SEED
            +
            zlib.crc32(
                (
                    group_name
                    + "|"
                    + feature
                ).encode(
                    "utf-8"
                )
            )
        ) % (2**32)

        ci_lo, ci_hi = (
            bootstrap_auc_ci(
                y,
                x,
                seed,
            )
        )

        p_perm = (
            permutation_auc_p(
                y,
                x,
                seed + 1,
            )
        )

        pos_values = x[
            y == 1
        ]

        neg_values = x[
            y == 0
        ]

        item = {
            "group":
                group_name,

            "feature":
                feature,

            "n":
                len(y),

            "help_n":
                help_n,

            "harm_n":
                harm_n,

            "help_mean":
                float(
                    pos_values.mean()
                ),

            "harm_mean":
                float(
                    neg_values.mean()
                ),

            "auc":
                float(auc),

            "auc_ci_lo":
                float(ci_lo),

            "auc_ci_hi":
                float(ci_hi),

            "p_perm":
                float(p_perm),
        }

        group_stats.append(
            item
        )

    # Primary multiplicity correction:
    # 8 features separately inside each target.
    #
    # Pooled is exploratory but we still report q.
    q_values = bh_fdr(
        [
            x["p_perm"]
            for x in group_stats
        ]
    )

    for item, q in zip(
        group_stats,
        q_values,
    ):
        item[
            "q_bh"
        ] = float(q)

        stats_rows.append(
            item
        )

    print(
        f"{'Feature':<34}"
        f"{'HelpMean':>10}"
        f"{'HarmMean':>10}"
        f"{'AUC':>8}"
        f"{'95% CI':>20}"
        f"{'p':>10}"
        f"{'qBH':>10}"
    )

    print("-" * 102)

    for item in group_stats:

        ci = (
            "[{:.3f},{:.3f}]".format(
                item["auc_ci_lo"],
                item["auc_ci_hi"],
            )
        )

        print(
            f"{item['feature']:<34}"
            f"{item['help_mean']:>10.4f}"
            f"{item['harm_mean']:>10.4f}"
            f"{item['auc']:>8.3f}"
            f"{ci:>20}"
            f"{item['p_perm']:>10.4f}"
            f"{item['q_bh']:>10.4f}"
        )


# ================================================================
# Same-direction cross-target summary
# ================================================================

lookup = {
    (
        r["group"],
        r["feature"],
    ): r
    for r in stats_rows
}

cross_rows = []

for feature in FEATURES:

    a = lookup[
        (
            "Laptop",
            feature,
        )
    ]

    b = lookup[
        (
            "LightSwitch",
            feature,
        )
    ]

    auc_laptop = float(
        a["auc"]
    )

    auc_switch = float(
        b["auc"]
    )

    same_direction = int(
        (
            auc_laptop - 0.5
        )
        *
        (
            auc_switch - 0.5
        )
        > 0
    )

    # Conservative weakest-target discrimination.
    min_abs_auc_effect = min(
        abs(
            auc_laptop - 0.5
        ),
        abs(
            auc_switch - 0.5
        ),
    )

    cross_rows.append(
        {
            "feature":
                feature,

            "laptop_auc":
                auc_laptop,

            "lightswitch_auc":
                auc_switch,

            "same_direction":
                same_direction,

            "min_abs_auc_effect":
                float(
                    min_abs_auc_effect
                ),

            "laptop_q_bh":
                float(
                    a["q_bh"]
                ),

            "lightswitch_q_bh":
                float(
                    b["q_bh"]
                ),
        }
    )


# Rank same-direction features by the weaker
# target's effect size.
cross_rows.sort(
    key=lambda r: (
        r["same_direction"],
        r["min_abs_auc_effect"],
    ),
    reverse=True,
)


print()
print("=" * 100)
print("CROSS-TARGET DIRECTION CHECK")
print("=" * 100)

print(
    f"{'Feature':<34}"
    f"{'Laptop':>10}"
    f"{'LightSw':>10}"
    f"{'SameDir':>10}"
    f"{'Min|AUC-.5|':>14}"
)

print("-" * 80)

for r in cross_rows:

    print(
        f"{r['feature']:<34}"
        f"{r['laptop_auc']:>10.3f}"
        f"{r['lightswitch_auc']:>10.3f}"
        f"{r['same_direction']:>10d}"
        f"{r['min_abs_auc_effect']:>14.3f}"
    )


# ================================================================
# Save statistics
# ================================================================

stats_file = (
    OUT_DIR
    / "within_target_stats.csv"
)

with open(
    stats_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "group",
            "feature",
            "n",
            "help_n",
            "harm_n",
            "help_mean",
            "harm_mean",
            "auc",
            "auc_ci_lo",
            "auc_ci_hi",
            "p_perm",
            "q_bh",
        ],
    )

    writer.writeheader()
    writer.writerows(
        stats_rows
    )


cross_file = (
    OUT_DIR
    / "cross_target_summary.csv"
)

with open(
    cross_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "feature",
            "laptop_auc",
            "lightswitch_auc",
            "same_direction",
            "min_abs_auc_effect",
            "laptop_q_bh",
            "lightswitch_q_bh",
        ],
    )

    writer.writeheader()
    writer.writerows(
        cross_rows
    )


print()
print(
    "Feature matrix :",
    feature_csv
)

print(
    "Target stats   :",
    stats_file
)

print(
    "Cross-target   :",
    cross_file
)

print()
print(
    "K10 DYNAMIC FEATURE ANALYSIS: PASS"
)
