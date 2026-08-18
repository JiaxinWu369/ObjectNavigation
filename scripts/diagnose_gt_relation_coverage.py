import glob
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import h5py

from datasets.constants import (
    AI2THOR_TARGET_CLASSES
)

from reliability.gt_relation_oracle import (
    ground_distance,
)


CLASSES = list(
    AI2THOR_TARGET_CLASSES[22]
)

DATA_DIR = Path(
    "datasets/Scene_Data"
)

RADIUS = 1.5


visible_cache = {}


def load_visible(scene):
    if scene in visible_cache:
        return visible_cache[scene]

    p = (
        DATA_DIR
        / scene
        / "visible_object_map_1.5.json"
    )

    with open(p) as f:
        obj_to_states = json.load(f)

    state_to_ids = defaultdict(list)

    for oid, states in obj_to_states.items():
        for state in states:
            state_to_ids[state].append(oid)

    visible_cache[scene] = dict(
        state_to_ids
    )

    return visible_cache[scene]


def obj_type(oid):
    return oid.split("|")[0]


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

stats = Counter()
by_target = defaultdict(Counter)

detector_only_categories = Counter()
gt_overlap_categories = Counter()
relation_far_categories = Counter()
relation_near_categories = Counter()

h5_cache = {}


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

    return h5_cache[scene][state][:]


for ep in episodes:
    scene = ep["scene"]
    state = ep["state"]
    target = ep["goal_object_type"]
    target_ids = ep["task_data"]

    stats["episodes"] += 1
    by_target[target]["episodes"] += 1

    state_to_ids = load_visible(scene)

    visible_ids = state_to_ids.get(
        state,
        []
    )

    if visible_ids:
        stats["state_with_gt_visible"] += 1
    else:
        stats["state_without_gt_visible"] += 1

    visible_by_class = defaultdict(list)

    for oid in visible_ids:
        visible_by_class[
            obj_type(oid)
        ].append(oid)

    det = get_det(
        scene,
        state,
    )

    scores = det[:, -1]

    episode_has_det = False
    episode_has_overlap = False
    episode_has_detector_only = False
    episode_has_far = False
    episode_has_near = False

    for idx, score in enumerate(scores):
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

        candidate_ids = (
            visible_by_class.get(
                category,
                []
            )
        )

        # -------------------------------------------------
        # Detector says category exists,
        # but GT says no currently visible instance.
        # -------------------------------------------------
        if not candidate_ids:
            stats[
                "detector_only_rows"
            ] += 1

            by_target[target][
                "detector_only_rows"
            ] += 1

            detector_only_categories[
                category
            ] += 1

            episode_has_detector_only = True

            continue

        # -------------------------------------------------
        # Detector / GT-visible overlap
        # -------------------------------------------------
        stats[
            "gt_overlap_rows"
        ] += 1

        by_target[target][
            "gt_overlap_rows"
        ] += 1

        gt_overlap_categories[
            category
        ] += 1

        episode_has_overlap = True

        min_dist = min(
            ground_distance(
                candidate_id,
                target_id,
            )
            for candidate_id
            in candidate_ids
            for target_id
            in target_ids
        )

        if min_dist > RADIUS:
            stats[
                "relation_far_rows"
            ] += 1

            by_target[target][
                "relation_far_rows"
            ] += 1

            relation_far_categories[
                category
            ] += 1

            episode_has_far = True

        else:
            stats[
                "relation_near_rows"
            ] += 1

            by_target[target][
                "relation_near_rows"
            ] += 1

            relation_near_categories[
                category
            ] += 1

            episode_has_near = True

    if episode_has_det:
        stats[
            "episodes_with_detection"
        ] += 1

    if episode_has_overlap:
        stats[
            "episodes_with_gt_overlap"
        ] += 1

    if episode_has_detector_only:
        stats[
            "episodes_with_detector_only"
        ] += 1

    if episode_has_far:
        stats[
            "episodes_relation_far"
        ] += 1

    if episode_has_near:
        stats[
            "episodes_relation_near"
        ] += 1


for f in h5_cache.values():
    f.close()


def pct(a, b):
    if b == 0:
        return 0.0

    return 100.0 * a / b


print()
print("=" * 88)
print("GLOBAL DIAGNOSTIC")
print("=" * 88)

for k in [
    "episodes",
    "state_with_gt_visible",
    "state_without_gt_visible",
    "episodes_with_detection",
    "episodes_with_gt_overlap",
    "episodes_with_detector_only",
    "episodes_relation_far",
    "episodes_relation_near",
    "detected_nongoal_rows",
    "gt_overlap_rows",
    "detector_only_rows",
    "relation_far_rows",
    "relation_near_rows",
]:
    print(
        f"{k:32s}:",
        stats[k],
    )


n_det = stats[
    "detected_nongoal_rows"
]

n_eps = stats[
    "episodes"
]


print()
print("=" * 88)
print("KEY RATIOS")
print("=" * 88)

print(
    "GT overlap among DET rows       :",
    f"{pct(stats['gt_overlap_rows'], n_det):.2f}%"
)

print(
    "Detector-only among DET rows    :",
    f"{pct(stats['detector_only_rows'], n_det):.2f}%"
)

print(
    "Relation-far among DET rows     :",
    f"{pct(stats['relation_far_rows'], n_det):.2f}%"
)

print(
    "Relation-far among overlap rows :",
    f"{pct(stats['relation_far_rows'], stats['gt_overlap_rows']):.2f}%"
)

print(
    "Episodes with GT overlap        :",
    f"{pct(stats['episodes_with_gt_overlap'], n_eps):.2f}%"
)

print(
    "Episodes relation-far eligible  :",
    f"{pct(stats['episodes_relation_far'], n_eps):.2f}%"
)

print(
    "Episodes detector-only present  :",
    f"{pct(stats['episodes_with_detector_only'], n_eps):.2f}%"
)


print()
print("=" * 88)
print("PER TARGET")
print("=" * 88)

print(
    f"{'Target':<16}"
    f"{'N':>6}"
    f"{'DET':>8}"
    f"{'GT':>8}"
    f"{'Only':>8}"
    f"{'Far':>8}"
    f"{'Near':>8}"
)

print("-" * 62)

for target in sorted(by_target):
    s = by_target[target]

    print(
        f"{target:<16}"
        f"{s['episodes']:>6}"
        f"{s['detected_nongoal_rows']:>8}"
        f"{s['gt_overlap_rows']:>8}"
        f"{s['detector_only_rows']:>8}"
        f"{s['relation_far_rows']:>8}"
        f"{s['relation_near_rows']:>8}"
    )


def print_top(title, counter):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    for category, n in counter.most_common(
        15
    ):
        print(
            f"{category:18s}",
            n,
        )


print_top(
    "TOP DETECTOR-ONLY CATEGORIES",
    detector_only_categories,
)

print_top(
    "TOP GT-OVERLAP CATEGORIES",
    gt_overlap_categories,
)

print_top(
    "TOP RELATION-FAR CATEGORIES",
    relation_far_categories,
)

print_top(
    "TOP RELATION-NEAR CATEGORIES",
    relation_near_categories,
)
