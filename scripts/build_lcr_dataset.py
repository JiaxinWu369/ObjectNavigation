import json
import os
import math
import pickle
from pathlib import Path
from collections import Counter, defaultdict

from reliability.local_region_memory import (
    LocalRegionMemory,
)


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

DATASET_NAME = os.environ.get(
    "AKGVP_LCR_DATASET_NAME",
    "lcr_pilot",
).strip()

RESULT_ROOT = Path(
    os.environ.get(
        "AKGVP_LCR_RESULT_ROOT",
        "results/lcr_pilot_base500",
    )
)

SPLIT_ROOT = Path(
    "test_val_split"
)

OUTPUT_ROOT = Path(
    os.environ.get(
        "AKGVP_LCR_OUTPUT_ROOT",
        "results/lcr_pilot_dataset",
    )
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

ALL_PATH = (
    OUTPUT_ROOT
    / "regions_all.jsonl"
)

LABELED_PATH = (
    OUTPUT_ROOT
    / "regions_labeled.jsonl"
)

STATS_PATH = (
    OUTPUT_ROOT
    / "stats.json"
)


MIN_VIEWS = 3

POSITIVE_DISTANCE = 1.5
NEGATIVE_DISTANCE = 2.5


memory_helper = LocalRegionMemory(
    num_classes=22,
    cell_size=1.0,
    position_bin=0.5,
    yaw_bin=90.0,
    horizon_bin=30.0,
    min_views=MIN_VIEWS,
)


def parse_state(state):
    return (
        memory_helper
        .parse_state(state)
    )


def normalize_state(s):
    """
    Normalize AKGVP state representations.

    Split file:
        "1.50|0.50|180|0"

    Episode logger:
        {
            "x": 1.5,
            "z": 0.5,
            "rotation": {"y": 180},
            "horizon": 0
        }
    """
    if isinstance(s, str):

        parts = s.split("|")

        if len(parts) < 4:
            raise ValueError(
                "Unexpected string state: {}".format(
                    s
                )
            )

        return (
            round(
                float(parts[0]),
                4,
            ),
            round(
                float(parts[1]),
                4,
            ),
            int(
                round(
                    float(parts[2])
                )
            ) % 360,
            int(
                round(
                    float(parts[3])
                )
            ),
        )

    if isinstance(s, dict):

        return (
            round(
                float(s["x"]),
                4,
            ),
            round(
                float(s["z"]),
                4,
            ),
            int(
                round(
                    float(
                        s["rotation"]["y"]
                    )
                )
            ) % 360,
            int(
                round(
                    float(
                        s["horizon"]
                    )
                )
            ),
        )

    raise TypeError(
        "Unknown state type: {}".format(
            type(s)
        )
    )


def parse_target_positions(
    task_data,
):
    """
    Example object id:
        Kettle|+01.04|+00.90|-02.60
    """
    positions = []

    for object_id in task_data:

        parts = str(
            object_id
        ).split("|")

        if len(parts) < 4:
            continue

        try:
            x = float(parts[-3])
            y = float(parts[-2])
            z = float(parts[-1])
        except ValueError:
            continue

        positions.append(
            (x, y, z)
        )

    return positions


def distance_xz(
    x,
    z,
    targets,
):
    return min(
        math.sqrt(
            (x - tx) ** 2
            +
            (z - tz) ** 2
        )
        for tx, _, tz
        in targets
    )


def train_or_val(scene):
    n = int(
        scene.replace(
            "FloorPlan",
            "",
        )
    )

    if 1 <= n <= 20:
        local = n

    elif 201 <= n <= 220:
        local = n - 200

    elif 301 <= n <= 320:
        local = n - 300

    elif 401 <= n <= 420:
        local = n - 400

    else:
        raise ValueError(
            "Unexpected training scene: "
            + scene
        )

    return (
        "val"
        if local >= 17
        else "train"
    )


all_records = []
labeled_records = []

episode_count = 0
regions_total = 0
regions_ge3 = 0

label_counter = Counter()
room_counter = Counter()
target_counter = Counter()
scene_counter = Counter()
split_counter = Counter()

view_lengths = []

print("=" * 96)
print("BUILD LCR LOCAL REGION DATASET")
print("=" * 96)


for room in ROOMS:

    split_path = (
        SPLIT_ROOT
        / f"{room}_{DATASET_NAME}_22.pkl"
    )

    episode_path = (
        RESULT_ROOT
        / f"episodes_{room}.jsonl"
    )

    with open(
        split_path,
        "rb",
    ) as f:
        specs = pickle.load(f)

    episodes = [
        json.loads(x)
        for x in open(
            episode_path,
            encoding="utf-8",
        )
        if x.strip()
    ]

    expected_episodes = len(specs)

    assert expected_episodes > 0, (
        room,
        "empty split",
    )

    assert len(episodes) == expected_episodes, (
        room,
        "result/spec count mismatch",
        len(episodes),
        expected_episodes,
    )

    by_eval_index = {
        int(ep["eval_index"]): ep
        for ep in episodes
    }

    assert set(
        by_eval_index.keys()
    ) == set(
        range(
            expected_episodes
        )
    ), (
        room,
        "eval_index mismatch",
        len(by_eval_index),
        expected_episodes,
    )

    for eval_index, spec in enumerate(
        specs
    ):

        ep = by_eval_index[
            eval_index
        ]

        assert (
            ep["scene"]
            ==
            spec["scene"]
        )

        assert (
            ep["target"]
            ==
            spec["goal_object_type"]
        )

        assert (
            normalize_state(
                ep["start_state"]
            )
            ==
            normalize_state(
                spec["state"]
            )
        ), (
            room,
            eval_index,
            "start_state",
            normalize_state(
                ep["start_state"]
            ),
            normalize_state(
                spec["state"]
            ),
        )

        scene = ep["scene"]
        target = ep["target"]

        target_positions = (
            parse_target_positions(
                spec["task_data"]
            )
        )

        if not target_positions:
            raise RuntimeError(
                "No GT target position: "
                f"{room} {eval_index} "
                f"{scene} {target}"
            )

        trace = ep.get(
            "context_trace",
            []
        )

        trace_truncated = (
            len(trace)
            <
            int(
                ep["ep_length"]
            )
        )

        # -------------------------------------
        # Group trajectory observations by
        # local 1m x 1m region.
        # -------------------------------------
        regions = defaultdict(
            list
        )

        seen_viewpoints = defaultdict(
            set
        )

        for obs in trace:

            state = obs["state"]

            region_key = (
                memory_helper
                .region_key(state)
            )

            viewpoint_key = (
                memory_helper
                .viewpoint_key(state)
            )

            # Only retain genuinely new
            # viewpoints within this region.
            if (
                viewpoint_key
                in seen_viewpoints[
                    region_key
                ]
            ):
                continue

            seen_viewpoints[
                region_key
            ].add(
                viewpoint_key
            )

            x, z, yaw, horizon = (
                parse_state(
                    state
                )
            )

            regions[
                region_key
            ].append(
                {
                    "step":
                        int(
                            obs["step"]
                        ),

                    "state":
                        state,

                    "x":
                        float(x),

                    "z":
                        float(z),

                    "yaw":
                        float(yaw),

                    "horizon":
                        float(horizon),

                    "viewpoint_key":
                        list(
                            viewpoint_key
                        ),

                    "goal_index":
                        int(
                            obs[
                                "goal_index"
                            ]
                        ),

                    "detector_scores":
                        [
                            float(v)
                            for v in
                            obs[
                                "detector_scores"
                            ]
                        ],

                    # Kept for diagnostics.
                    # First learned model does not
                    # have to use these.
                    "base_attention":
                        [
                            float(v)
                            for v in
                            obs[
                                "base_attention"
                            ]
                        ],

                    "semantic_kl":
                        float(
                            obs[
                                "semantic_kl"
                            ]
                        ),
                }
            )

        regions_total += len(
            regions
        )

        for region_key, seq in (
            regions.items()
        ):

            num_views = len(seq)

            if num_views < MIN_VIEWS:
                continue

            regions_ge3 += 1
            view_lengths.append(
                num_views
            )

            d_min = min(
                distance_xz(
                    obs["x"],
                    obs["z"],
                    target_positions,
                )
                for obs in seq
            )

            if (
                d_min
                <= POSITIVE_DISTANCE
            ):
                label = 1
                label_name = "positive"

            elif (
                d_min
                >= NEGATIVE_DISTANCE
            ):
                label = 0
                label_name = "negative"

            else:
                label = None
                label_name = "ignore"

            data_split = (
                train_or_val(
                    scene
                )
            )

            record = {
                "room":
                    room,

                "scene":
                    scene,

                "target":
                    target,

                "goal_index":
                    int(
                        seq[0][
                            "goal_index"
                        ]
                    ),

                "eval_index":
                    int(
                        eval_index
                    ),

                "trace_length":
                    int(
                        len(trace)
                    ),

                "episode_length":
                    int(
                        ep["ep_length"]
                    ),

                "trace_truncated":
                    bool(
                        trace_truncated
                    ),

                "lcr_episode_id":
                    spec[
                        "lcr_episode_id"
                    ],

                "source_episode_id":
                    spec[
                        "lcr_source_episode_id"
                    ],

                "region_key":
                    list(
                        region_key
                    ),

                "num_views":
                    int(
                        num_views
                    ),

                "target_positions":
                    [
                        [
                            float(x),
                            float(y),
                            float(z),
                        ]
                        for x, y, z
                        in target_positions
                    ],

                "distance_min":
                    float(
                        d_min
                    ),

                "label":
                    label,

                "label_name":
                    label_name,

                "data_split":
                    data_split,

                "sequence":
                    seq,
            }

            all_records.append(
                record
            )

            label_counter[
                label_name
            ] += 1

            room_counter[
                (room, label_name)
            ] += 1

            target_counter[
                (target, label_name)
            ] += 1

            scene_counter[
                (scene, label_name)
            ] += 1

            split_counter[
                (data_split, label_name)
            ] += 1

            if label is not None:
                labeled_records.append(
                    record
                )

        episode_count += 1


with open(
    ALL_PATH,
    "w",
    encoding="utf-8",
) as f:
    for row in all_records:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


with open(
    LABELED_PATH,
    "w",
    encoding="utf-8",
) as f:
    for row in labeled_records:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


stats = {
    "episodes":
        episode_count,

    "regions_total":
        regions_total,

    "regions_ge3_views":
        regions_ge3,

    "regions_labeled":
        len(
            labeled_records
        ),

    "positive":
        label_counter[
            "positive"
        ],

    "negative":
        label_counter[
            "negative"
        ],

    "ignored":
        label_counter[
            "ignore"
        ],

    "positive_rate_labeled":
        (
            label_counter[
                "positive"
            ]
            /
            len(
                labeled_records
            )
            if labeled_records
            else 0.0
        ),

    "views_mean":
        (
            sum(view_lengths)
            /
            len(view_lengths)
            if view_lengths
            else 0.0
        ),

    "views_min":
        (
            min(view_lengths)
            if view_lengths
            else 0
        ),

    "views_max":
        (
            max(view_lengths)
            if view_lengths
            else 0
        ),

    "train_labeled":
        (
            split_counter[
                ("train", "positive")
            ]
            +
            split_counter[
                ("train", "negative")
            ]
        ),

    "val_labeled":
        (
            split_counter[
                ("val", "positive")
            ]
            +
            split_counter[
                ("val", "negative")
            ]
        ),
}


with open(
    STATS_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        stats,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print("Episodes                :", episode_count)
print("Regions total           :", regions_total)
print("Regions >=3 views       :", regions_ge3)

print(
    "Positive                :",
    label_counter["positive"],
)

print(
    "Negative                :",
    label_counter["negative"],
)

print(
    "Ignored                 :",
    label_counter["ignore"],
)

print(
    "Labeled total           :",
    len(labeled_records),
)

if labeled_records:
    print(
        "Positive rate           :",
        round(
            stats[
                "positive_rate_labeled"
            ],
            4,
        ),
    )

if view_lengths:
    print(
        "Views min/mean/max      :",
        min(view_lengths),
        round(
            sum(view_lengths)
            / len(view_lengths),
            3,
        ),
        max(view_lengths),
    )

print()
print("Scene split:")
print(
    "  train labeled         :",
    stats["train_labeled"],
)
print(
    "  val labeled           :",
    stats["val_labeled"],
)


print()
print("Per-room labels:")

for room in ROOMS:
    print(
        f"{room:<14}",
        "pos=",
        room_counter[
            (room, "positive")
        ],
        "neg=",
        room_counter[
            (room, "negative")
        ],
        "ignore=",
        room_counter[
            (room, "ignore")
        ],
    )


print()
print("Per-target labels:")

targets = sorted(
    set(
        target
        for target, _
        in target_counter
    )
)

for target in targets:
    print(
        f"{target:<16}",
        "pos=",
        target_counter[
            (target, "positive")
        ],
        "neg=",
        target_counter[
            (target, "negative")
        ],
        "ignore=",
        target_counter[
            (target, "ignore")
        ],
    )


print()
print("Saved:")
print(" ", ALL_PATH)
print(" ", LABELED_PATH)
print(" ", STATS_PATH)

print()
print(
    "LCR DATASET BUILD: PASS"
)
