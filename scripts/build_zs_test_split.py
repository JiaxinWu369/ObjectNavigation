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

UNSEEN = {
    "Bowl",
    "DeskLamp",
    "Laptop",
    "LightSwitch",
    "Plate",
    "StoveBurner",
}

grand = Counter()
total = 0

for room in ROOMS:

    src = ROOT / f"{room}_test_22.pkl"
    dst = ROOT / f"{room}_zs_test_22.pkl"

    with open(src, "rb") as f:
        rows = pickle.load(f)

    selected = [
        x for x in rows
        if x["goal_object_type"] in UNSEEN
    ]

    with open(dst, "wb") as f:
        pickle.dump(
            selected,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    counts = Counter(
        x["goal_object_type"]
        for x in selected
    )

    total += len(selected)
    grand.update(counts)

    print(
        f"{room:12s}",
        f"N={len(selected):4d}",
        dict(sorted(counts.items())),
    )


print("=" * 80)
print("TOTAL:", total)
print(
    "TARGETS:",
    dict(sorted(grand.items()))
)

assert total == 1244
