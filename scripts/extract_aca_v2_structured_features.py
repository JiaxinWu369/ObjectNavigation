import json
from pathlib import Path

import numpy as np

from datasets.constants import AI2THOR_TARGET_CLASSES
from models.object_distribution import Object_Distribution


INPUT = Path(
    "results/aca_train500/selector/"
    "eligible_all_6655.jsonl"
)

OUT = Path(
    "results/aca_train500/selector/"
    "aca_v2_structured_all_6655.jsonl"
)


CLASSES = list(
    AI2THOR_TARGET_CLASSES[22]
)

CLASS_TO_IDX = {
    x: i
    for i, x in enumerate(CLASSES)
}


obj_dist = Object_Distribution(
    dataset="AI2THOR"
)

ALPHA = (
    obj_dist
    .init_prior_alpha
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)


def goal_profile(scene, goal_idx):

    sid = obj_dist.get_scene_id(
        scene
    )

    alpha = ALPHA[
        sid
    ].copy()

    prob = (
        alpha
        /
        alpha.sum(
            axis=1,
            keepdims=True,
        )
    )

    profile = prob[
        :,
        goal_idx
    ].copy()

    profile[
        goal_idx
    ] = 0.0

    z = profile.sum()

    if z <= 0:
        raise RuntimeError(
            f"empty profile: "
            f"{scene}/{goal_idx}"
        )

    profile /= z

    return profile


def extract(row):

    goal_idx = CLASS_TO_IDX[
        row["target"]
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

    if steps != list(
        range(11)
    ):
        raise RuntimeError(
            f"bad trace: {steps}"
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

    assert detector.shape == (
        11,
        22,
    )

    assert attention.shape == (
        11,
        22,
    )

    profile = goal_profile(
        row["scene"],
        goal_idx,
    )

    # ------------------------------------------
    # Context-level evidence available at s10.
    # ------------------------------------------

    det_max = detector.max(
        axis=0
    )

    det_seen_fraction = (
        detector > 0
    ).mean(
        axis=0
    )

    att_t10 = attention[
        10
    ]

    # ------------------------------------------
    # Remove goal and sort remaining objects
    # according to static P(context | goal).
    # Stable sort makes construction deterministic.
    # ------------------------------------------

    context_idx = [
        i
        for i in range(22)
        if i != goal_idx
    ]

    context_idx = sorted(
        context_idx,
        key=lambda i: (
            -profile[i],
            i,
        ),
    )

    assert len(
        context_idx
    ) == 21

    vec = []

    for idx in context_idx:

        vec.extend(
            [
                float(
                    profile[idx]
                ),

                float(
                    det_max[idx]
                ),

                float(
                    det_seen_fraction[
                        idx
                    ]
                ),

                float(
                    att_t10[idx]
                ),
            ]
        )

    # ------------------------------------------
    # Goal/global state.
    # ------------------------------------------

    vec.extend(
        [
            float(
                det_max[
                    goal_idx
                ]
            ),

            float(
                det_seen_fraction[
                    goal_idx
                ]
            ),

            float(
                np.mean(
                    kl[6:11]
                )
            ),
        ]
    )

    vec = np.asarray(
        vec,
        dtype=np.float64,
    )

    if vec.shape != (87,):
        raise RuntimeError(
            f"bad feature shape: "
            f"{vec.shape}"
        )

    if not np.all(
        np.isfinite(vec)
    ):
        raise RuntimeError(
            "non-finite feature"
        )

    return vec


rows = []

with open(
    INPUT,
    encoding="utf-8",
) as f:

    for line in f:

        if not line.strip():
            continue

        r = json.loads(
            line
        )

        vec = extract(r)

        rows.append(
            {
                "batch":
                    r["batch"],

                "room":
                    r["room"],

                "scene":
                    r["scene"],

                "target":
                    r["target"],

                "start_state":
                    r["start_state"],

                "pair_state":
                    r["pair_state"],

                "early10_success":
                    int(
                        r[
                            "early10_success"
                        ]
                    ),

                "early20_success":
                    int(
                        r[
                            "early20_success"
                        ]
                    ),

                "features":
                    vec.tolist(),
            }
        )


assert len(rows) == 6655

assert all(
    len(r["features"]) == 87
    for r in rows
)


with open(
    OUT,
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


disc = [
    r
    for r in rows
    if r["pair_state"]
    in {"N10", "N01"}
]

assert len(disc) == 240


X = np.asarray(
    [
        r["features"]
        for r in rows
    ],
    dtype=np.float64,
)


print("=" * 88)
print("ACA-v2 STRUCTURED FEATURE EXTRACTION")
print("=" * 88)

print(
    "Decision states   :",
    len(rows),
)

print(
    "Discordant        :",
    len(disc),
)

print(
    "Feature dimension :",
    X.shape[1],
)

print(
    "Finite            :",
    bool(
        np.all(
            np.isfinite(X)
        )
    ),
)

print()
print(
    "Representation:"
)

print(
    "  21 prior-ranked context slots"
)

print(
    "  x 4 values:"
)

print(
    "    prior relation"
)

print(
    "    detector max 0..10"
)

print(
    "    detector seen fraction 0..10"
)

print(
    "    base attention at t10"
)

print(
    "  + goal detector max"
)

print(
    "  + goal seen fraction"
)

print(
    "  + KL recent5"
)

print()
print(
    "Target/room/scene IDs as features: NO"
)

print(
    "action_probs as features          : NO"
)

print()
print(
    "Saved:",
    OUT,
)

print()
print(
    "ACA-v2 STRUCTURED FEATURES: PASS"
)
