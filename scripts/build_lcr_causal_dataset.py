import json
import math
from pathlib import Path
from collections import Counter


SRC = Path(
    "results/lcr_full_dataset/regions_all.jsonl"
)

OUT = Path(
    "results/lcr_causal_dataset"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PREFIX_PATH = (
    OUT / "prefixes_labeled.jsonl"
)

STATS_PATH = (
    OUT / "stats.json"
)

POS_DISTANCE = 1.5
NEG_DISTANCE = 2.5
MIN_VIEWS = 3


rows = [
    json.loads(x)
    for x in open(
        SRC,
        encoding="utf-8",
    )
    if x.strip()
]


def center_distance(
    region_key,
    target_positions,
):
    i, j = region_key

    # 1m x 1m cell center.
    cx = float(i) + 0.5
    cz = float(j) + 0.5

    return min(
        math.sqrt(
            (cx - float(tx)) ** 2
            +
            (cz - float(tz)) ** 2
        )
        for tx, _, tz
        in target_positions
    )


prefix_records = []

region_counts = Counter()
prefix_counts = Counter()
label_counts = Counter()

labeled_regions = 0
ignored_regions = 0

for row in rows:

    seq = row["sequence"]

    if len(seq) < MIN_VIEWS:
        continue

    d = center_distance(
        row["region_key"],
        row["target_positions"],
    )

    if d <= POS_DISTANCE:
        label = 1
        label_name = "positive"

    elif d >= NEG_DISTANCE:
        label = 0
        label_name = "negative"

    else:
        ignored_regions += 1
        continue

    labeled_regions += 1

    data_split = row["data_split"]

    label_counts[
        (data_split, label_name)
    ] += 1

    region_uid = (
        "{}|{}|{}|{}|{}".format(
            row["room"],
            row["scene"],
            row["eval_index"],
            row["region_key"][0],
            row["region_key"][1],
        )
    )

    T = len(seq)

    # --------------------------------------------------
    # Every prefix with >=3 observed valid viewpoints
    # becomes a causal training example.
    #
    # Prefix k contains only observations 1..k.
    # Prediction is applied to the NEXT navigation step.
    # --------------------------------------------------
    for k in range(
        MIN_VIEWS,
        T + 1,
    ):
        record = {
            "region_uid":
                region_uid,

            "room":
                row["room"],

            "scene":
                row["scene"],

            "target":
                row["target"],

            "goal_index":
                int(
                    row["goal_index"]
                ),

            "eval_index":
                int(
                    row["eval_index"]
                ),

            "region_key":
                row["region_key"],

            "data_split":
                data_split,

            "label":
                int(label),

            "label_name":
                label_name,

            "region_center_distance":
                float(d),

            "prefix_views":
                int(k),

            "total_recorded_views":
                int(T),

            "is_final_prefix":
                bool(k == T),

            "trace_truncated":
                bool(
                    row.get(
                        "trace_truncated",
                        False,
                    )
                ),

            "sequence":
                seq[:k],
        }

        prefix_records.append(
            record
        )

        prefix_counts[
            data_split
        ] += 1

    region_counts[
        data_split
    ] += 1


with open(
    PREFIX_PATH,
    "w",
    encoding="utf-8",
) as f:
    for row in prefix_records:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


stats = {
    "source_regions":
        len(rows),

    "labeled_regions":
        labeled_regions,

    "ignored_regions":
        ignored_regions,

    "train_regions":
        region_counts["train"],

    "val_regions":
        region_counts["val"],

    "train_prefixes":
        prefix_counts["train"],

    "val_prefixes":
        prefix_counts["val"],

    "train_positive_regions":
        label_counts[
            ("train", "positive")
        ],

    "train_negative_regions":
        label_counts[
            ("train", "negative")
        ],

    "val_positive_regions":
        label_counts[
            ("val", "positive")
        ],

    "val_negative_regions":
        label_counts[
            ("val", "negative")
        ],
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
    )


print("=" * 80)
print("LCR CAUSAL PREFIX DATASET")
print("=" * 80)

for k, v in stats.items():
    print(
        f"{k:28s}: {v}"
    )

print()
print(
    "prefix records:",
    len(prefix_records)
)

print()
print(
    "LCR CAUSAL DATASET: PASS"
)
