import csv
import json
from pathlib import Path
from collections import Counter, defaultdict


BASE_DIR = Path("results/pair_base100")
SUPPRESS_DIR = Path("results/pair_nocontext_base100")

OUT_DIR = Path("results/context_reversal_100k")
OUT_DIR.mkdir(parents=True, exist_ok=True)


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]


def load_run(root):
    rows = {}

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        if not p.exists():
            raise FileNotFoundError(p)

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)

                key = (
                    row["scene_type"],
                    int(row["eval_index"]),
                )

                if key in rows:
                    raise RuntimeError(
                        f"Duplicate episode key: {key}"
                    )

                rows[key] = row

    return rows


def canonical_start_state(s):
    return (
        round(float(s["x"]), 4),
        round(float(s["z"]), 4),
        int(round(float(s["rotation"]["y"]))),
        int(round(float(s["horizon"]))),
    )


def validate_pair(key, a, b):
    fields = [
        "scene",
        "scene_type",
        "target",
        "eval_index",
    ]

    for field in fields:
        if a[field] != b[field]:
            raise RuntimeError(
                f"Pair mismatch at {key}: "
                f"{field}: {a[field]} != {b[field]}"
            )

    sa = canonical_start_state(a["start_state"])
    sb = canonical_start_state(b["start_state"])

    if sa != sb:
        raise RuntimeError(
            f"Start-state mismatch at {key}: "
            f"{sa} != {sb}"
        )


def outcome_name(base_success, suppress_success):
    if not base_success and not suppress_success:
        return "N00_both_fail"

    if not base_success and suppress_success:
        return "N01_suppress_only"

    if base_success and not suppress_success:
        return "N10_base_only"

    return "N11_both_success"


base = load_run(BASE_DIR)
suppress = load_run(SUPPRESS_DIR)

if set(base) != set(suppress):
    only_base = sorted(set(base) - set(suppress))
    only_suppress = sorted(set(suppress) - set(base))

    raise RuntimeError(
        "Episode key mismatch.\n"
        f"Only Base: {only_base[:10]}\n"
        f"Only Suppress: {only_suppress[:10]}"
    )


keys = sorted(base)

print("=" * 100)
print("PAIR VALIDATION")
print("=" * 100)
print("Base episodes    :", len(base))
print("Suppress episodes:", len(suppress))
print("Paired episodes  :", len(keys))


records = []

counts = Counter()
target_counts = defaultdict(Counter)
room_counts = defaultdict(Counter)
scene_counts = defaultdict(Counter)

different_trajectories = 0


for key in keys:
    b = base[key]
    s = suppress[key]

    validate_pair(key, b, s)

    bs = bool(b["success"])
    ss = bool(s["success"])

    outcome = outcome_name(bs, ss)

    counts[outcome] += 1
    target_counts[b["target"]][outcome] += 1
    room_counts[b["scene_type"]][outcome] += 1
    scene_counts[b["scene"]][outcome] += 1

    if b["actions"] != s["actions"]:
        different_trajectories += 1

    record = {
        "scene_type": b["scene_type"],
        "eval_index": int(b["eval_index"]),
        "scene": b["scene"],
        "target": b["target"],

        "start_x": b["start_state"]["x"],
        "start_y": b["start_state"]["y"],
        "start_z": b["start_state"]["z"],
        "start_rotation": b["start_state"]["rotation"]["y"],
        "start_horizon": b["start_state"]["horizon"],

        "outcome": outcome,

        "base_success": int(bs),
        "suppress_success": int(ss),

        "base_spl": float(b["spl"]),
        "suppress_spl": float(s["spl"]),

        "base_ep_length": int(b.get("ep_length", len(b["actions"]))),
        "suppress_ep_length": int(
            s.get("ep_length", len(s["actions"]))
        ),

        "base_dts": float(b.get("dts", float("nan"))),
        "suppress_dts": float(
            s.get("dts", float("nan"))
        ),

        "trajectory_different": int(
            b["actions"] != s["actions"]
        ),

        "base_actions": json.dumps(
            b["actions"],
            separators=(",", ":"),
        ),
        "suppress_actions": json.dumps(
            s["actions"],
            separators=(",", ":"),
        ),
    }

    records.append(record)


N = len(records)

n00 = counts["N00_both_fail"]
n01 = counts["N01_suppress_only"]
n10 = counts["N10_base_only"]
n11 = counts["N11_both_success"]

base_successes = n10 + n11
suppress_successes = n01 + n11
union_successes = n01 + n10 + n11

base_sr = 100.0 * base_successes / N
suppress_sr = 100.0 * suppress_successes / N
union_sr = 100.0 * union_successes / N

best_fixed_sr = max(base_sr, suppress_sr)
headroom = union_sr - best_fixed_sr

reversal_n = n01 + n10
reversal_rate = 100.0 * reversal_n / N


print()
print("=" * 100)
print("BASE vs NOCONTEXT REVERSAL SUMMARY")
print("=" * 100)

print(
    f"N00 Both fail                    : "
    f"{n00:4d}  ({100*n00/N:6.2f}%)"
)
print(
    f"N01 Suppress only                : "
    f"{n01:4d}  ({100*n01/N:6.2f}%)"
)
print(
    f"N10 Base only                    : "
    f"{n10:4d}  ({100*n10/N:6.2f}%)"
)
print(
    f"N11 Both success                 : "
    f"{n11:4d}  ({100*n11/N:6.2f}%)"
)

print()
print(
    f"Base successes                   : "
    f"{base_successes:4d} / {N} = {base_sr:6.2f}%"
)
print(
    f"NoContext successes              : "
    f"{suppress_successes:4d} / {N} = "
    f"{suppress_sr:6.2f}%"
)
print(
    f"Hindsight union successes        : "
    f"{union_successes:4d} / {N} = "
    f"{union_sr:6.2f}%"
)
print(
    f"Oracle headroom over best fixed  : "
    f"{headroom:+6.2f} pp"
)
print(
    f"Reversal episodes N01+N10        : "
    f"{reversal_n:4d} / {N} = "
    f"{reversal_rate:6.2f}%"
)
print(
    f"Different action trajectories    : "
    f"{different_trajectories:4d} / {N} = "
    f"{100*different_trajectories/N:6.2f}%"
)


def print_group_table(title, grouped):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    header = (
        f"{'Group':<20}"
        f"{'N':>7}"
        f"{'N00':>8}"
        f"{'N01 Sup':>10}"
        f"{'N10 Base':>11}"
        f"{'N11':>8}"
        f"{'Net Sup':>10}"
        f"{'Rev%':>9}"
    )

    print(header)
    print("-" * len(header))

    for group in sorted(grouped):
        c = grouped[group]

        a00 = c["N00_both_fail"]
        a01 = c["N01_suppress_only"]
        a10 = c["N10_base_only"]
        a11 = c["N11_both_success"]

        total = a00 + a01 + a10 + a11
        net = a01 - a10
        rev = (
            100.0 * (a01 + a10) / total
            if total > 0
            else 0.0
        )

        print(
            f"{group:<20}"
            f"{total:>7d}"
            f"{a00:>8d}"
            f"{a01:>10d}"
            f"{a10:>11d}"
            f"{a11:>8d}"
            f"{net:>+10d}"
            f"{rev:>8.2f}%"
        )


print_group_table(
    "PER TARGET REVERSAL",
    target_counts,
)

print_group_table(
    "PER ROOM REVERSAL",
    room_counts,
)

print_group_table(
    "PER SCENE REVERSAL",
    scene_counts,
)


fieldnames = list(records[0].keys())


def write_csv(filename, subset):
    p = OUT_DIR / filename

    with open(
        p,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(subset)

    print(
        f"saved {filename:<24}: "
        f"{len(subset)}"
    )


print()
print("=" * 100)
print("WRITE EPISODE LISTS")
print("=" * 100)

write_csv(
    "all_pairs.csv",
    records,
)

write_csv(
    "suppress_only.csv",
    [
        r for r in records
        if r["outcome"] == "N01_suppress_only"
    ],
)

write_csv(
    "base_only.csv",
    [
        r for r in records
        if r["outcome"] == "N10_base_only"
    ],
)

write_csv(
    "both_success.csv",
    [
        r for r in records
        if r["outcome"] == "N11_both_success"
    ],
)

write_csv(
    "both_fail.csv",
    [
        r for r in records
        if r["outcome"] == "N00_both_fail"
    ],
)


summary = {
    "N": N,
    "N00_both_fail": n00,
    "N01_suppress_only": n01,
    "N10_base_only": n10,
    "N11_both_success": n11,
    "base_sr": base_sr,
    "suppress_sr": suppress_sr,
    "hindsight_union_sr": union_sr,
    "oracle_headroom_over_best_fixed_pp": headroom,
    "reversal_count": reversal_n,
    "reversal_rate_percent": reversal_rate,
    "different_action_trajectories":
        different_trajectories,
}

with open(
    OUT_DIR / "summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print()
print(
    "Summary:",
    OUT_DIR / "summary.json"
)
