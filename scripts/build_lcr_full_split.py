import pickle
from pathlib import Path
from collections import Counter


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


total = 0
all_source_ids = set()

print("=" * 88)
print("BUILD LCR FULL SPLIT")
print("=" * 88)

for room in ROOMS:

    src = ROOT / f"{room}_aca_train_22.pkl"
    dst = ROOT / f"{room}_lcr_full_22.pkl"

    with open(src, "rb") as f:
        rows = pickle.load(f)

    assert len(rows) == 1000

    output = []

    for i, row in enumerate(rows):

        row = row.copy()

        assert (
            row["goal_object_type"]
            not in UNSEEN_TARGETS
        )

        source_id = row["aca_episode_id"]

        assert source_id not in all_source_ids
        all_source_ids.add(source_id)

        row["lcr_source_episode_id"] = (
            source_id
        )

        row["lcr_episode_id"] = (
            f"{room}|full|{i:04d}"
        )

        output.append(row)

    scenes = Counter(
        x["scene"]
        for x in output
    )

    assert len(scenes) == 20
    assert set(scenes.values()) == {50}

    with open(dst, "wb") as f:
        pickle.dump(
            output,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print()
    print(room)
    print("episodes :", len(output))
    print("scenes   :", len(scenes))
    print(
        "per scene:",
        sorted(set(scenes.values()))
    )
    print("saved    :", dst)

    total += len(output)


assert total == 4000
assert len(all_source_ids) == 4000

print()
print("Total episodes :", total)
print("Unique IDs     :", len(all_source_ids))
print("Unseen targets : 0")
print()
print("LCR FULL SPLIT: PASS")
