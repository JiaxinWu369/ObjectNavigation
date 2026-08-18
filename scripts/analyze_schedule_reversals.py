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

    by_target = defaultdict(Counter)
    by_room = defaultdict(Counter)
    by_scene = defaultdict(Counter)

    total = Counter()

    recovered = []
    lost = []

    for key in sorted(a):
        x = a[key]
        y = b[key]

        assert x["scene"] == y["scene"]
        assert x["target"] == y["target"]

        o = outcome(x, y)

        total[o] += 1
        by_target[x["target"]][o] += 1
        by_room[x["scene_type"]][o] += 1
        by_scene[x["scene"]][o] += 1

        record = {
            "scene_type": x["scene_type"],
            "eval_index": int(x["eval_index"]),
            "scene": x["scene"],
            "target": x["target"],
            "a_success": int(x["success"]),
            "b_success": int(y["success"]),
            "a_spl": x["spl"],
            "b_spl": y["spl"],
        }

        if o == "N01":
            recovered.append(record)

        if o == "N10":
            lost.append(record)

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print(
        "N00={} N01={} N10={} N11={} net={:+d}".format(
            total["N00"],
            total["N01"],
            total["N10"],
            total["N11"],
            total["N01"] - total["N10"],
        )
    )

    def print_table(title, groups):
        print()
        print(title)

        print(
            f"{'Group':<20}"
            f"{'N01 rec':>10}"
            f"{'N10 lost':>11}"
            f"{'Net':>8}"
            f"{'Discord':>10}"
        )

        print("-" * 60)

        for g in sorted(groups):
            c = groups[g]

            n01 = c["N01"]
            n10 = c["N10"]

            print(
                f"{g:<20}"
                f"{n01:>10d}"
                f"{n10:>11d}"
                f"{n01-n10:>+8d}"
                f"{n01+n10:>10d}"
            )

    print_table(
        "PER TARGET",
        by_target,
    )

    print_table(
        "PER ROOM",
        by_room,
    )

    print_table(
        "PER SCENE",
        by_scene,
    )

    return recovered, lost


base = load_run(
    "results/pair_base100"
)

k40 = load_run(
    "results/pair_early40_nocontext_base100"
)

nc = load_run(
    "results/pair_nocontext_base100"
)


rec_base, lost_base = analyze(
    "BASE -> EARLY40",
    base,
    k40,
)

rec_nc, lost_nc = analyze(
    "NOCONTEXT -> EARLY40",
    nc,
    k40,
)


out_dir = Path(
    "results/context_reversal_100k"
)

for name, rows in [
    ("base_to_k40_recovered.json", rec_base),
    ("base_to_k40_lost.json", lost_base),
    ("nc_to_k40_recovered.json", rec_nc),
    ("nc_to_k40_lost.json", lost_nc),
]:
    with open(
        out_dir / name,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rows,
            f,
            indent=2,
        )
