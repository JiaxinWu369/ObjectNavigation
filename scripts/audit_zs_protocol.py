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


def load(name):
    with open(ROOT / name, "rb") as f:
        return pickle.load(f)


total_zs_val = 0
total_val_unseen = 0
total_test_unseen = 0

for room in ROOMS:

    val = load(
        f"{room}_val_22.pkl"
    )

    test = load(
        f"{room}_test_22.pkl"
    )

    zs_val = load(
        f"{room}_zs_val_22.pkl"
    )

    val_unseen = [
        x for x in val
        if x["goal_object_type"] in UNSEEN
    ]

    test_unseen = [
        x for x in test
        if x["goal_object_type"] in UNSEEN
    ]

    print("=" * 72)
    print(room)

    print(
        "VAL total       :",
        len(val),
    )

    print(
        "VAL unseen      :",
        len(val_unseen),
    )

    print(
        "existing ZS-Val :",
        len(zs_val),
    )

    print(
        "TEST total      :",
        len(test),
    )

    print(
        "TEST unseen     :",
        len(test_unseen),
    )

    print(
        "ZS-Val targets  :",
        dict(
            sorted(
                Counter(
                    x["goal_object_type"]
                    for x in zs_val
                ).items()
            )
        ),
    )

    print(
        "TEST-ZS targets :",
        dict(
            sorted(
                Counter(
                    x["goal_object_type"]
                    for x in test_unseen
                ).items()
            )
        ),
    )

    print(
        "VAL scenes      :",
        sorted(
            set(
                x["scene"]
                for x in val_unseen
            )
        ),
    )

    print(
        "TEST scenes     :",
        sorted(
            set(
                x["scene"]
                for x in test_unseen
            )
        ),
    )

    total_zs_val += len(zs_val)
    total_val_unseen += len(val_unseen)
    total_test_unseen += len(test_unseen)


print("=" * 72)

print(
    "TOTAL existing ZS-Val:",
    total_zs_val,
)

print(
    "TOTAL filtered VAL   :",
    total_val_unseen,
)

print(
    "TOTAL candidate ZS-Test:",
    total_test_unseen,
)
