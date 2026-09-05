import json
from pathlib import Path
from collections import Counter


PATH = Path(
    "results/lcr_causal_dataset/"
    "prefixes_labeled.jsonl"
)

OUT = Path(
    "results/lcr_target_folds.json"
)


# ============================================================
# Load ONE final prefix per region from the 64 training scenes.
# ============================================================

rows = [
    json.loads(x)
    for x in open(
        PATH,
        encoding="utf-8",
    )
    if x.strip()
]

rows = [
    r
    for r in rows
    if (
        r["data_split"] == "train"
        and r.get(
            "is_final_prefix",
            False,
        )
    )
]


counts = Counter(
    r["target"]
    for r in rows
)

targets = sorted(
    counts.keys()
)

assert len(targets) == 16, (
    "Expected 16 seen targets, got",
    len(targets),
    targets,
)


print("=" * 88)
print("LCR TARGET-HELD-OUT FOLD CONSTRUCTION")
print("=" * 88)

print()
print("Target region counts:")

for target in sorted(
    targets,
    key=lambda x: (
        -counts[x],
        x,
    ),
):
    print(
        f"{target:<16}: "
        f"{counts[target]}"
    )


# ============================================================
# Step 1:
# Capacity-constrained largest-first greedy assignment.
#
# IMPORTANT:
# each fold MUST contain exactly 4 target categories.
# ============================================================

folds = [
    {
        "targets": [],
        "n_regions": 0,
    }
    for _ in range(4)
]


ordered_targets = sorted(
    targets,
    key=lambda x: (
        -counts[x],
        x,
    ),
)


for target in ordered_targets:

    eligible = [
        i
        for i in range(4)
        if len(
            folds[i]["targets"]
        ) < 4
    ]

    assert eligible

    idx = min(
        eligible,
        key=lambda i: (
            folds[i]["n_regions"],
            len(
                folds[i]["targets"]
            ),
            i,
        ),
    )

    folds[idx][
        "targets"
    ].append(
        target
    )

    folds[idx][
        "n_regions"
    ] += counts[target]


# ============================================================
# Step 2:
# Deterministic pairwise swap refinement.
#
# Keep exactly four targets/fold, but improve balance in the
# number of validation regions.
# ============================================================

def objective(current_folds):
    total = sum(
        f["n_regions"]
        for f in current_folds
    )

    mean = (
        total
        / len(current_folds)
    )

    # Primary:
    # minimize squared imbalance.
    sq = sum(
        (
            f["n_regions"]
            - mean
        ) ** 2
        for f in current_folds
    )

    # Secondary:
    # minimize largest fold difference.
    values = [
        f["n_regions"]
        for f in current_folds
    ]

    spread = (
        max(values)
        - min(values)
    )

    return (
        sq,
        spread,
    )


while True:

    current_obj = objective(
        folds
    )

    best = None

    for i in range(4):
        for j in range(
            i + 1,
            4,
        ):

            for ti in list(
                folds[i][
                    "targets"
                ]
            ):

                for tj in list(
                    folds[j][
                        "targets"
                    ]
                ):

                    ni = (
                        folds[i][
                            "n_regions"
                        ]
                        - counts[ti]
                        + counts[tj]
                    )

                    nj = (
                        folds[j][
                            "n_regions"
                        ]
                        - counts[tj]
                        + counts[ti]
                    )

                    trial = [
                        {
                            "targets":
                                list(
                                    f[
                                        "targets"
                                    ]
                                ),

                            "n_regions":
                                f[
                                    "n_regions"
                                ],
                        }
                        for f in folds
                    ]

                    trial[i][
                        "n_regions"
                    ] = ni

                    trial[j][
                        "n_regions"
                    ] = nj

                    trial_obj = (
                        objective(
                            trial
                        )
                    )

                    if (
                        trial_obj
                        < current_obj
                    ):

                        candidate = (
                            trial_obj,
                            i,
                            j,
                            ti,
                            tj,
                        )

                        if (
                            best is None
                            or candidate
                            < best
                        ):
                            best = (
                                candidate
                            )

    if best is None:
        break

    _, i, j, ti, tj = best

    folds[i][
        "targets"
    ].remove(
        ti
    )

    folds[j][
        "targets"
    ].remove(
        tj
    )

    folds[i][
        "targets"
    ].append(
        tj
    )

    folds[j][
        "targets"
    ].append(
        ti
    )

    folds[i][
        "n_regions"
    ] = sum(
        counts[t]
        for t in folds[i][
            "targets"
        ]
    )

    folds[j][
        "n_regions"
    ] = sum(
        counts[t]
        for t in folds[j][
            "targets"
        ]
    )


# ============================================================
# Validation
# ============================================================

all_assigned = []

for i, fold in enumerate(
    folds
):
    fold[
        "targets"
    ] = sorted(
        fold[
            "targets"
        ]
    )

    assert (
        len(
            fold[
                "targets"
            ]
        )
        == 4
    ), (
        i,
        fold,
    )

    expected_n = sum(
        counts[t]
        for t in fold[
            "targets"
        ]
    )

    assert (
        expected_n
        ==
        fold[
            "n_regions"
        ]
    )

    all_assigned.extend(
        fold[
            "targets"
        ]
    )


assert (
    sorted(
        all_assigned
    )
    ==
    sorted(
        targets
    )
)

assert (
    len(
        all_assigned
    )
    ==
    len(
        set(
            all_assigned
        )
    )
    ==
    16
)


total_regions = sum(
    counts.values()
)

mean_regions = (
    total_regions / 4.0
)


# ============================================================
# Save
# ============================================================

output = {
    "method":
        (
            "capacity-constrained "
            "largest-first greedy "
            "plus pairwise swap refinement"
        ),

    "source":
        str(PATH),

    "scene_scope":
        (
            "64 training scenes only; "
            "one final prefix per region"
        ),

    "num_targets":
        16,

    "num_folds":
        4,

    "targets_per_fold":
        4,

    "total_regions":
        total_regions,

    "ideal_regions_per_fold":
        mean_regions,

    "target_counts":
        dict(
            sorted(
                counts.items()
            )
        ),

    "folds": [
        {
            "fold":
                i,

            "heldout_targets":
                fold[
                    "targets"
                ],

            "heldout_regions":
                fold[
                    "n_regions"
                ],

            "deviation_from_mean":
                (
                    fold[
                        "n_regions"
                    ]
                    - mean_regions
                ),
        }
        for i, fold
        in enumerate(
            folds
        )
    ],
}


with open(
    OUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


# ============================================================
# Print
# ============================================================

print()
print("-" * 88)
print("FINAL 4-FOLD ASSIGNMENT")
print("-" * 88)

for fold in output[
    "folds"
]:
    print()
    print(
        "Fold",
        fold["fold"],
    )

    print(
        "  targets :",
        fold[
            "heldout_targets"
        ],
    )

    print(
        "  regions :",
        fold[
            "heldout_regions"
        ],
    )

    print(
        "  deviation:",
        round(
            fold[
                "deviation_from_mean"
            ],
            2,
        ),
    )


values = [
    f[
        "heldout_regions"
    ]
    for f in output[
        "folds"
    ]
]


print()
print(
    "Total regions          :",
    total_regions,
)

print(
    "Ideal regions / fold   :",
    round(
        mean_regions,
        2,
    ),
)

print(
    "Min / max fold regions :",
    min(values),
    max(values),
)

print(
    "Fold spread            :",
    max(values)
    - min(values),
)

print()
print(
    "Saved:",
    OUT,
)

print()
print(
    "LCR TARGET FOLDS: PASS"
)
