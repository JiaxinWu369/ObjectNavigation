import pickle
import random
from pathlib import Path
from collections import Counter, defaultdict


SEED = 20260822
PER_SCENE = 25

ROOT = Path("test_val_split")

ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

UNSEEN_TARGETS = {
    "Bowl",
    "DeskLamp",
    "Laptop",
    "LightSwitch",
    "Plate",
    "StoveBurner",
}


all_total = 0
all_ids = set()

print("=" * 96)
print("BUILD LCR PILOT SPLIT")
print("=" * 96)

for room in ROOMS:

    src = ROOT / f"{room}_aca_train_22.pkl"
    dst = ROOT / f"{room}_lcr_pilot_22.pkl"

    with open(src, "rb") as f:
        rows = pickle.load(f)

    assert len(rows) == 1000
    assert all(isinstance(x, dict) for x in rows)

    by_scene = defaultdict(list)

    for row in rows:
        by_scene[row["scene"]].append(row)

    assert len(by_scene) == 20

    selected = []

    for scene in sorted(
        by_scene,
        key=lambda x: int(
            x.replace("FloorPlan", "")
        ),
    ):
        scene_rows = by_scene[scene]

        assert len(scene_rows) == 50, (
            room,
            scene,
            len(scene_rows),
        )

        # Deterministic but independent scene-level sampling.
        scene_number = int(
            scene.replace("FloorPlan", "")
        )

        rng = random.Random(
            SEED + scene_number
        )

        indices = sorted(
            rng.sample(
                range(50),
                PER_SCENE,
            )
        )

        chosen = [
            scene_rows[i].copy()
            for i in indices
        ]

        assert len(chosen) == PER_SCENE

        for local_idx, row in enumerate(chosen):
            source_id = row["aca_episode_id"]

            row["lcr_source_episode_id"] = source_id

            row["lcr_episode_id"] = (
                f"{room}|{scene}|{local_idx:03d}"
            )

            assert (
                row["goal_object_type"]
                not in UNSEEN_TARGETS
            )

        selected.extend(chosen)

    assert len(selected) == 500

    # Check no duplicate source episodes.
    src_ids = [
        x["lcr_source_episode_id"]
        for x in selected
    ]

    assert len(src_ids) == len(set(src_ids))

    overlap = all_ids.intersection(src_ids)
    assert len(overlap) == 0

    all_ids.update(src_ids)

    with open(dst, "wb") as f:
        pickle.dump(
            selected,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    scene_counts = Counter(
        x["scene"]
        for x in selected
    )

    target_counts = Counter(
        x["goal_object_type"]
        for x in selected
    )

    print()
    print(f"[{room}]")
    print("source episodes :", len(rows))
    print("pilot episodes  :", len(selected))
    print("scenes          :", len(scene_counts))
    print(
        "per scene       :",
        sorted(
            set(scene_counts.values())
        ),
    )

    print("target distribution:")

    for target, n in sorted(
        target_counts.items()
    ):
        print(
            f"  {target:<16} {n:4d}"
        )

    print("saved           :", dst)

    all_total += len(selected)


assert all_total == 2000
assert len(all_ids) == 2000

print()
print("=" * 96)
print("GLOBAL")
print("=" * 96)
print("Rooms            :", len(ROOMS))
print("Scenes           :", 80)
print("Episodes         :", all_total)
print("Episodes / scene :", PER_SCENE)
print("Unique source IDs:", len(all_ids))
print("Unseen targets   : 0")

print()
print("LCR PILOT SPLIT: PASS")
