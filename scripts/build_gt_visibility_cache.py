import gc
import glob
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import h5py

from datasets.constants import AI2THOR_TARGET_CLASSES


CLASSES = list(AI2THOR_TARGET_CLASSES[22])
CLASS_SET = set(CLASSES)

DATA_DIR = Path("datasets/Scene_Data")
CACHE_DIR = Path("reliability/gt_visibility_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# AI2-THOR metadata category -> AKGVP 22-category name.
TYPE_MAP = {
    "SinkBasin": "Sink",
}


def normalize_type(t):
    return TYPE_MAP.get(t, t)


def canonical_state(s):
    """
    x|z|rotation|horizon
    """
    p = str(s).split("|")

    if len(p) != 4:
        raise ValueError(
            "Bad state: " + str(s)
        )

    x = float(p[0])
    z = float(p[1])
    rot = float(p[2])
    hor = float(p[3])

    return (
        f"{x:.2f}|{z:.2f}|"
        f"{int(round(rot))}|"
        f"{int(round(hor))}"
    )


# ---------------------------------------------------------
# Load all 1260 ZS-Val episodes.
# ---------------------------------------------------------
episodes = []

for p in sorted(
    glob.glob(
        "test_val_split/*_zs_val_22.pkl"
    )
):
    with open(p, "rb") as f:
        episodes.extend(
            pickle.load(f)
        )

print("Episodes:", len(episodes))

if len(episodes) != 1260:
    raise RuntimeError(
        f"Expected 1260, got {len(episodes)}"
    )


episodes_by_scene = defaultdict(list)

for ep in episodes:
    episodes_by_scene[
        ep["scene"]
    ].append(ep)


print("Scenes:", len(episodes_by_scene))


# ---------------------------------------------------------
# Build compact visibility cache for ALL metadata states
# in each ZS-Val scene.
#
# cache[state] = {
#     "visible": [...],
#     "min_distance": {category: distance},
# }
# ---------------------------------------------------------
for scene in sorted(episodes_by_scene):

    metadata_path = (
        DATA_DIR
        / scene
        / "metadata.json"
    )

    cache_path = (
        CACHE_DIR
        / f"{scene}.json"
    )

    print()
    print("=" * 80)
    print("BUILD", scene)
    print("=" * 80)
    print("metadata:", metadata_path)

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    cache = {}

    for raw_state, event in metadata.items():

        state = canonical_state(
            raw_state
        )

        visible_types = set()
        min_distance = {}

        for obj in event.get(
            "objects",
            []
        ):
            if not obj.get(
                "visible",
                False
            ):
                continue

            category = normalize_type(
                obj.get(
                    "objectType",
                    ""
                )
            )

            if category not in CLASS_SET:
                continue

            visible_types.add(
                category
            )

            distance = obj.get(
                "distance",
                None
            )

            if distance is not None:
                distance = float(
                    distance
                )

                if (
                    category not in min_distance
                    or
                    distance
                    < min_distance[category]
                ):
                    min_distance[
                        category
                    ] = distance

        cache[state] = {
            "visible": sorted(
                visible_types
            ),
            "min_distance": (
                min_distance
            ),
        }

    with open(
        cache_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cache,
            f,
            ensure_ascii=False,
        )

    print(
        "metadata states:",
        len(metadata)
    )

    print(
        "cache states   :",
        len(cache)
    )

    print(
        "saved          :",
        cache_path
    )

    del metadata
    del cache
    gc.collect()


# ---------------------------------------------------------
# Diagnostic on the 1260 INITIAL STATES.
# ---------------------------------------------------------
stats = Counter()
by_target = defaultdict(Counter)

not_visible_categories = Counter()
visible_categories = Counter()
visible_near_categories = Counter()
visible_far_categories = Counter()

h5_cache = {}
gt_cache = {}


def get_gt(scene):
    if scene not in gt_cache:
        p = (
            CACHE_DIR
            / f"{scene}.json"
        )

        with open(
            p,
            "r",
            encoding="utf-8",
        ) as f:
            gt_cache[scene] = json.load(
                f
            )

    return gt_cache[scene]


def get_det(scene, state):
    if scene not in h5_cache:
        p = (
            DATA_DIR
            / scene
            / "det_feature_22_cates.hdf5"
        )

        h5_cache[scene] = h5py.File(
            p,
            "r",
        )

    return h5_cache[scene][
        state
    ][:]


for ep in episodes:

    scene = ep["scene"]
    raw_state = ep["state"]
    state = canonical_state(
        raw_state
    )
    target = ep[
        "goal_object_type"
    ]

    stats["episodes"] += 1
    by_target[target]["episodes"] += 1

    gt = get_gt(scene)

    if state not in gt:
        stats[
            "state_missing_metadata"
        ] += 1
        continue

    stats[
        "state_found_metadata"
    ] += 1

    gt_info = gt[state]

    visible = set(
        gt_info["visible"]
    )

    min_distance = (
        gt_info["min_distance"]
    )

    if visible:
        stats[
            "state_with_visible_22"
        ] += 1
    else:
        stats[
            "state_without_visible_22"
        ] += 1

    det = get_det(
        scene,
        raw_state,
    )

    scores = det[:, -1]

    episode_has_det = False
    episode_has_visible_overlap = False
    episode_has_not_visible = False

    for idx, score in enumerate(
        scores
    ):
        category = CLASSES[idx]

        if category == target:
            continue

        if float(score) <= 0:
            continue

        episode_has_det = True

        stats[
            "detected_nongoal_rows"
        ] += 1

        by_target[target][
            "detected_nongoal_rows"
        ] += 1

        if category in visible:

            stats[
                "gt_visible_rows"
            ] += 1

            by_target[target][
                "gt_visible_rows"
            ] += 1

            visible_categories[
                category
            ] += 1

            episode_has_visible_overlap = True

            d = min_distance.get(
                category,
                None
            )

            if d is not None:

                if d <= 1.5:
                    stats[
                        "gt_visible_near_rows"
                    ] += 1

                    visible_near_categories[
                        category
                    ] += 1

                else:
                    stats[
                        "gt_visible_far_rows"
                    ] += 1

                    visible_far_categories[
                        category
                    ] += 1

        else:

            stats[
                "gt_not_visible_rows"
            ] += 1

            by_target[target][
                "gt_not_visible_rows"
            ] += 1

            not_visible_categories[
                category
            ] += 1

            episode_has_not_visible = True

    if episode_has_det:
        stats[
            "episodes_with_detection"
        ] += 1

    if episode_has_visible_overlap:
        stats[
            "episodes_with_visible_overlap"
        ] += 1

    if episode_has_not_visible:
        stats[
            "episodes_with_not_visible_det"
        ] += 1


for f in h5_cache.values():
    f.close()


def pct(a, b):
    if b == 0:
        return 0.0

    return 100.0 * a / b


print()
print("=" * 92)
print("RAW METADATA VISIBILITY DIAGNOSTIC")
print("=" * 92)

for k in [
    "episodes",
    "state_found_metadata",
    "state_missing_metadata",
    "state_with_visible_22",
    "state_without_visible_22",
    "episodes_with_detection",
    "episodes_with_visible_overlap",
    "episodes_with_not_visible_det",
    "detected_nongoal_rows",
    "gt_visible_rows",
    "gt_not_visible_rows",
    "gt_visible_near_rows",
    "gt_visible_far_rows",
]:
    print(
        f"{k:34s}:",
        stats[k],
    )


det_n = stats[
    "detected_nongoal_rows"
]

print()
print("=" * 92)
print("KEY RATIOS")
print("=" * 92)

print(
    "GT-visible among DET rows       :",
    f"{pct(stats['gt_visible_rows'], det_n):.2f}%"
)

print(
    "GT-NOT-visible among DET rows   :",
    f"{pct(stats['gt_not_visible_rows'], det_n):.2f}%"
)

print(
    "Visible <=1.5m among DET rows   :",
    f"{pct(stats['gt_visible_near_rows'], det_n):.2f}%"
)

print(
    "Visible >1.5m among DET rows    :",
    f"{pct(stats['gt_visible_far_rows'], det_n):.2f}%"
)

print(
    "Episodes with visible overlap   :",
    f"{pct(stats['episodes_with_visible_overlap'], stats['episodes']):.2f}%"
)

print(
    "Episodes with NOT-visible DET   :",
    f"{pct(stats['episodes_with_not_visible_det'], stats['episodes']):.2f}%"
)


print()
print("=" * 92)
print("PER TARGET")
print("=" * 92)

print(
    f"{'Target':<16}"
    f"{'N':>6}"
    f"{'DET':>8}"
    f"{'Visible':>10}"
    f"{'NotVis':>10}"
)

print("-" * 52)

for target in sorted(
    by_target
):
    s = by_target[target]

    print(
        f"{target:<16}"
        f"{s['episodes']:>6}"
        f"{s['detected_nongoal_rows']:>8}"
        f"{s['gt_visible_rows']:>10}"
        f"{s['gt_not_visible_rows']:>10}"
    )


def top(title, counter):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)

    for c, n in counter.most_common(
        15
    ):
        print(
            f"{c:18s}",
            n
        )


top(
    "TOP GT-NOT-VISIBLE DET CATEGORIES",
    not_visible_categories,
)

top(
    "TOP GT-VISIBLE DET CATEGORIES",
    visible_categories,
)

top(
    "TOP GT-VISIBLE <=1.5m",
    visible_near_categories,
)

top(
    "TOP GT-VISIBLE >1.5m",
    visible_far_categories,
)
