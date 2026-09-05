import pickle
import hashlib
from pathlib import Path

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


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def stable(x):
    """
    Only fields defining an evaluation episode.
    """
    return repr(
        (
            x["scene"],
            x["state"],
            x["goal_object_type"],
            x["task_data"],
        )
    )


all_pass = True

for room in ROOMS:

    val = load(
        ROOT / f"{room}_val_22.pkl"
    )

    zs = load(
        ROOT / f"{room}_zs_val_22.pkl"
    )

    filt = [
        x for x in val
        if x["goal_object_type"] in UNSEEN
    ]

    a = [
        stable(x)
        for x in filt
    ]

    b = [
        stable(x)
        for x in zs
    ]

    sequence_equal = a == b
    set_equal = sorted(a) == sorted(b)

    print(
        room,
        "N=",
        len(zs),
        "sequence_equal=",
        sequence_equal,
        "multiset_equal=",
        set_equal,
    )

    all_pass &= (
        sequence_equal
        and set_equal
    )


print()
print(
    "ZS-VAL EXACT FILTER:",
    "PASS" if all_pass else "FAIL"
)

if not all_pass:
    raise SystemExit(1)
