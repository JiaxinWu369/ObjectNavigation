import csv
import json
import math
from pathlib import Path

import numpy as np

from datasets.constants import AI2THOR_TARGET_CLASSES
from models.object_distribution import Object_Distribution


INPUT = Path(
    "results/aca_train500/selector/"
    "eligible_all_6655.jsonl"
)

OUT_DIR = Path(
    "results/aca_train500/selector"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FEATURE_NAMES = [
    "prior_entropy",
    "prior_top3_mass",
    "prior_support_score_0_10",
    "prior_support_mass_0_10",
    "goal_score_max_0_10",
    "goal_seen_count_0_10",
    "kl_mean_recent5",
    "ctx_att_l1_recent5",
    "prior_entropy_x_support_score",
    "prior_top3_x_support_mass",
]


CLASSES = list(
    AI2THOR_TARGET_CLASSES[22]
)

assert len(CLASSES) == 22

CLASS_TO_IDX = {
    x: i
    for i, x in enumerate(CLASSES)
}


# --------------------------------------------------
# Use the repository's own prior loader so that
# feature construction has exactly the same class
# order and room prior as AKGVP.
# --------------------------------------------------

obj_dist = Object_Distribution(
    dataset="AI2THOR"
)

INIT_ALPHA = (
    obj_dist
    .init_prior_alpha
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)

assert INIT_ALPHA.shape == (
    4,
    22,
    22,
)


def static_goal_profile(scene, goal_idx):
    """
    Reproduce the pre-observation AKGVP probability
    matrix, then select P[:, goal].

    Goal self-entry is removed and the remaining
    21 context entries are normalized to sum to 1.
    """

    scene_id = obj_dist.get_scene_id(
        scene
    )

    alpha = INIT_ALPHA[
        scene_id
    ].copy()

    row_sum = alpha.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(row_sum <= 0):
        raise RuntimeError(
            f"Non-positive prior row sum "
            f"in {scene}"
        )

    prob = alpha / row_sum

    # AKGVP later adds I, but goal self-entry is
    # excluded below, so the non-goal profile is
    # unchanged by the identity term.
    profile = prob[
        :,
        goal_idx,
    ].copy()

    profile[
        goal_idx
    ] = 0.0

    z = float(
        profile.sum()
    )

    if z <= 0:
        raise RuntimeError(
            f"Empty context profile: "
            f"{scene}, goal={goal_idx}"
        )

    profile /= z

    return profile


def prior_descriptors(profile):

    positive = profile[
        profile > 0
    ]

    entropy = -float(
        np.sum(
            positive
            * np.log(positive)
        )
    )

    # Normalize to [0,1] relative to a uniform
    # distribution over the 21 non-goal classes.
    entropy /= math.log(21.0)

    top3 = float(
        np.sort(profile)[-3:].sum()
    )

    return entropy, top3


def extract_row(row):

    target = row["target"]

    if target not in CLASS_TO_IDX:
        raise RuntimeError(
            f"Unknown target: {target}"
        )

    goal_idx = CLASS_TO_IDX[
        target
    ]

    trace = sorted(
        row["context_trace"],
        key=lambda x: int(
            x["step"]
        ),
    )

    steps = [
        int(x["step"])
        for x in trace
    ]

    if steps != list(range(11)):
        raise RuntimeError(
            "Expected trace steps 0..10, "
            f"got {steps} for "
            f"{row['scene']} / {target}"
        )

    for x in trace:

        if int(
            x["goal_index"]
        ) != goal_idx:

            raise RuntimeError(
                "goal_index mismatch: "
                f"{row['scene']} / "
                f"{target} / "
                f"{x['goal_index']} / "
                f"{goal_idx}"
            )

    detector = np.asarray(
        [
            x["detector_scores"]
            for x in trace
        ],
        dtype=np.float64,
    )

    attention = np.asarray(
        [
            x["base_attention"]
            for x in trace
        ],
        dtype=np.float64,
    )

    kl = np.asarray(
        [
            x["semantic_kl"]
            for x in trace
        ],
        dtype=np.float64,
    )

    if detector.shape != (
        11,
        22,
    ):
        raise RuntimeError(
            f"Bad detector shape: "
            f"{detector.shape}"
        )

    if attention.shape != (
        11,
        22,
    ):
        raise RuntimeError(
            f"Bad attention shape: "
            f"{attention.shape}"
        )

    if kl.shape != (11,):
        raise RuntimeError(
            f"Bad KL shape: {kl.shape}"
        )

    if not (
        np.all(np.isfinite(detector))
        and np.all(np.isfinite(attention))
        and np.all(np.isfinite(kl))
    ):
        raise RuntimeError(
            "Non-finite trace value"
        )

    profile = static_goal_profile(
        row["scene"],
        goal_idx,
    )

    prior_entropy, prior_top3 = (
        prior_descriptors(
            profile
        )
    )

    # ------------------------------------------------
    # Accumulated evidence available by s_10.
    # ------------------------------------------------

    det_max = detector.max(
        axis=0
    )

    det_ever = (
        detector > 0
    ).any(
        axis=0
    ).astype(
        np.float64
    )

    prior_support_score = float(
        np.dot(
            profile,
            det_max,
        )
    )

    prior_support_mass = float(
        np.dot(
            profile,
            det_ever,
        )
    )

    goal_score_max = float(
        det_max[
            goal_idx
        ]
    )

    goal_seen_count = float(
        np.sum(
            detector[
                :,
                goal_idx
            ] > 0
        )
    )

    # ------------------------------------------------
    # recent5 = states t=6,7,8,9,10.
    #
    # base_attention is used, not action_probs.
    # Goal self-attention is excluded.
    # ------------------------------------------------

    recent = slice(
        6,
        11,
    )

    kl_mean_recent5 = float(
        np.mean(
            kl[recent]
        )
    )

    context_mask = np.ones(
        22,
        dtype=bool,
    )

    context_mask[
        goal_idx
    ] = False

    context_attention = (
        attention[
            recent
        ][
            :,
            context_mask
        ]
    )

    ctx_att_l1_recent5 = float(
        np.mean(
            np.sum(
                np.abs(
                    context_attention
                ),
                axis=1,
            )
        )
    )

    features = {
        "prior_entropy":
            prior_entropy,

        "prior_top3_mass":
            prior_top3,

        "prior_support_score_0_10":
            prior_support_score,

        "prior_support_mass_0_10":
            prior_support_mass,

        "goal_score_max_0_10":
            goal_score_max,

        "goal_seen_count_0_10":
            goal_seen_count,

        "kl_mean_recent5":
            kl_mean_recent5,

        "ctx_att_l1_recent5":
            ctx_att_l1_recent5,

        "prior_entropy_x_support_score":
            prior_entropy
            * prior_support_score,

        "prior_top3_x_support_mass":
            prior_top3
            * prior_support_mass,
    }

    values = np.asarray(
        [
            features[k]
            for k in FEATURE_NAMES
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError(
            "Non-finite feature vector"
        )

    return features


rows = []

with open(
    INPUT,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        if not line.strip():
            continue

        row = json.loads(
            line
        )

        feats = extract_row(
            row
        )

        out = {
            "batch":
                row["batch"],

            "room":
                row["room"],

            "scene":
                row["scene"],

            "target":
                row["target"],

            "start_state":
                row["start_state"],

            "pair_state":
                row["pair_state"],

            "early10_success":
                int(
                    row[
                        "early10_success"
                    ]
                ),

            "early20_success":
                int(
                    row[
                        "early20_success"
                    ]
                ),
        }

        if row[
            "pair_state"
        ] == "N10":

            out[
                "restore_now_label"
            ] = 1

        elif row[
            "pair_state"
        ] == "N01":

            out[
                "restore_now_label"
            ] = 0

        else:
            out[
                "restore_now_label"
            ] = None

        out.update(
            feats
        )

        rows.append(
            out
        )


assert len(rows) == 6655


discordant = [
    r
    for r in rows
    if r["restore_now_label"]
    is not None
]

assert len(discordant) == 240


all_jsonl = (
    OUT_DIR
    / "aca_v1_features_all_6655.jsonl"
)

disc_jsonl = (
    OUT_DIR
    / "aca_v1_features_discordant_240.jsonl"
)

csv_path = (
    OUT_DIR
    / "aca_v1_features_discordant_240.csv"
)


with open(
    all_jsonl,
    "w",
    encoding="utf-8",
) as f:

    for r in rows:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


with open(
    disc_jsonl,
    "w",
    encoding="utf-8",
) as f:

    for r in discordant:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


csv_fields = [
    "batch",
    "room",
    "scene",
    "target",
    "start_state",
    "pair_state",
    "restore_now_label",
] + FEATURE_NAMES


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields,
        extrasaction="ignore",
    )

    writer.writeheader()

    writer.writerows(
        discordant
    )


print("=" * 88)
print("ACA-v1 FEATURE EXTRACTION")
print("=" * 88)

print(
    "Classes:",
    CLASSES,
)

print()
print(
    "All decision states :",
    len(rows),
)

print(
    "Discordant labels   :",
    len(discordant),
)

print(
    "RestoreNow          :",
    sum(
        r["restore_now_label"] == 1
        for r in discordant
    ),
)

print(
    "Delay               :",
    sum(
        r["restore_now_label"] == 0
        for r in discordant
    ),
)

print()
print(
    f"{'Feature':<38}"
    f"{'Mean':>12}"
    f"{'Std':>12}"
    f"{'Min':>12}"
    f"{'Max':>12}"
)

print("-" * 86)

for name in FEATURE_NAMES:

    a = np.asarray(
        [
            r[name]
            for r in rows
        ],
        dtype=np.float64,
    )

    print(
        f"{name:<38}"
        f"{a.mean():>12.6f}"
        f"{a.std():>12.6f}"
        f"{a.min():>12.6f}"
        f"{a.max():>12.6f}"
    )


print()
print(
    "Ignored selector input: action_probs"
)

print(
    "Recent temporal window: steps 6..10"
)

print(
    "Static profile orientation: P[:, goal]"
)

print()
print(
    "All features :",
    all_jsonl,
)

print(
    "Preference   :",
    disc_jsonl,
)

print()
print(
    "ACA-v1 FEATURE EXTRACTION: PASS"
)
