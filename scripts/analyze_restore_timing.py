import json
from pathlib import Path
from collections import Counter, defaultdict


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]


def load_run(root):
    root = Path(root)
    rows = {}

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)

                key = (
                    r["scene_type"],
                    int(r["eval_index"]),
                )

                rows[key] = r

    return rows


def outcome(a, b):
    sa = bool(a["success"])
    sb = bool(b["success"])

    if not sa and not sb:
        return "N00"
    if not sa and sb:
        return "N01"
    if sa and not sb:
        return "N10"

    return "N11"


def analyze(name, a, b):
    assert set(a) == set(b)

    total = Counter()
    by_target = defaultdict(Counter)
    by_room = defaultdict(Counter)

    for key in sorted(a):
        x = a[key]
        y = b[key]

        assert x["scene"] == y["scene"]
        assert x["target"] == y["target"]

        o = outcome(x, y)

        total[o] += 1
        by_target[x["target"]][o] += 1
        by_room[x["scene_type"]][o] += 1

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print(
        "N01 restore-help = {}  "
        "N10 restore-harm = {}  "
        "net = {:+d}".format(
            total["N01"],
            total["N10"],
            total["N01"] - total["N10"],
        )
    )

    print()
    print("PER TARGET")

    print(
        f"{'Target':<18}"
        f"{'Help':>8}"
        f"{'Harm':>8}"
        f"{'Net':>8}"
        f"{'Discord':>10}"
    )

    print("-" * 54)

    for g in sorted(by_target):
        c = by_target[g]

        h = c["N01"]
        d = c["N10"]

        print(
            f"{g:<18}"
            f"{h:>8d}"
            f"{d:>8d}"
            f"{h-d:>+8d}"
            f"{h+d:>10d}"
        )

    print()
    print("PER ROOM")

    print(
        f"{'Room':<18}"
        f"{'Help':>8}"
        f"{'Harm':>8}"
        f"{'Net':>8}"
        f"{'Discord':>10}"
    )

    print("-" * 54)

    for g in sorted(by_room):
        c = by_room[g]

        h = c["N01"]
        d = c["N10"]

        print(
            f"{g:<18}"
            f"{h:>8d}"
            f"{d:>8d}"
            f"{h-d:>+8d}"
            f"{h+d:>10d}"
        )


nc = load_run(
    "results/pair_nocontext_base100"
)

for k in [10, 20, 40]:

    run = load_run(
        f"results/pair_early{k}_nocontext_base100"
    )

    analyze(
        f"NOCONTEXT -> RESTORE AT K={k}",
        nc,
        run,
    )
