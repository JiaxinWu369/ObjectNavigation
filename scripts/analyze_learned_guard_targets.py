import json
from pathlib import Path
from collections import Counter, defaultdict


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]


def load(root):
    root = Path(root)
    rows = {}

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        with open(
            p,
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                x = json.loads(line)

                key = (
                    x.get("scene_type", room),
                    int(x["eval_index"]),
                )

                rows[key] = x

    return rows


def compare(name_a, path_a, name_b, path_b):

    A = load(path_a)
    B = load(path_b)

    assert set(A) == set(B)

    by_target = defaultdict(Counter)
    by_room = defaultdict(Counter)
    total = Counter()

    recovered = []
    lost = []

    for key in sorted(A):

        a = A[key]
        b = B[key]

        assert a["scene"] == b["scene"]
        assert a["target"] == b["target"]

        sa = bool(a["success"])
        sb = bool(b["success"])

        if not sa and not sb:
            o = "N00"
        elif not sa and sb:
            o = "N01"
            recovered.append((key, a, b))
        elif sa and not sb:
            o = "N10"
            lost.append((key, a, b))
        else:
            o = "N11"

        total[o] += 1
        by_target[a["target"]][o] += 1
        by_room[
            a.get("scene_type", key[0])
        ][o] += 1

    print()
    print("=" * 88)
    print(f"{name_a}  vs  {name_b}")
    print("=" * 88)

    print(
        "N00/N01/N10/N11:",
        total["N00"],
        total["N01"],
        total["N10"],
        total["N11"],
    )

    print()
    print("TARGET LEVEL")
    print("-" * 88)

    print(
        f"{'Target':<18}"
        f"{name_b+' only':>14}"
        f"{name_a+' only':>14}"
        f"{'Net':>8}"
        f"{'Episodes':>10}"
        f"{'SR A':>10}"
        f"{'SR B':>10}"
    )

    for target in sorted(by_target):

        c = by_target[target]

        n = sum(c.values())

        sr_a = (
            c["N10"] + c["N11"]
        ) / n

        sr_b = (
            c["N01"] + c["N11"]
        ) / n

        net = (
            c["N01"] - c["N10"]
        )

        print(
            f"{target:<18}"
            f"{c['N01']:>14}"
            f"{c['N10']:>14}"
            f"{net:>8}"
            f"{n:>10}"
            f"{100*sr_a:>9.2f}%"
            f"{100*sr_b:>9.2f}%"
        )

    print()
    print("ROOM LEVEL")
    print("-" * 88)

    print(
        f"{'Room':<18}"
        f"{name_b+' only':>14}"
        f"{name_a+' only':>14}"
        f"{'Net':>8}"
    )

    for room in sorted(by_room):

        c = by_room[room]

        print(
            f"{room:<18}"
            f"{c['N01']:>14}"
            f"{c['N10']:>14}"
            f"{c['N01']-c['N10']:>8}"
        )

    print()
    print("RECOVERED BY", name_b)
    print("-" * 88)

    for key, a, b in recovered:

        print(
            key,
            "scene=",
            a["scene"],
            "target=",
            a["target"],
            "A_len=",
            a["ep_length"],
            "B_len=",
            b["ep_length"],
        )

    print()
    print("LOST BY", name_b)
    print("-" * 88)

    for key, a, b in lost:

        print(
            key,
            "scene=",
            a["scene"],
            "target=",
            a["target"],
            "A_len=",
            a["ep_length"],
            "B_len=",
            b["ep_length"],
        )


compare(
    "Base",
    "results/final_test_base500",
    "Learned",
    "results/learned_guard_test_t04",
)

compare(
    "Rule",
    "results/guard_tau02_zstest",
    "Learned",
    "results/learned_guard_test_t04",
)
