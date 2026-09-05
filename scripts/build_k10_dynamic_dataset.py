import csv
import json
from pathlib import Path
from collections import Counter, defaultdict


TRACE_DIR = Path(
    "results/diag_nocontext_trace100"
)

K10_DIR = Path(
    "results/pair_early10_nocontext_base100"
)

OUT_DIR = Path(
    "results/k10_dynamic_dataset_100k"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]


def canonical_start_state(s):
    return (
        round(float(s["x"]), 4),
        round(float(s["z"]), 4),
        int(round(float(s["rotation"]["y"]))),
        int(round(float(s["horizon"]))),
    )


def load_run(root):

    out = {}

    for room in ROOMS:

        p = root / f"episodes_{room}.jsonl"

        if not p.exists():
            raise FileNotFoundError(p)

        with open(
            p,
            encoding="utf-8"
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                r = json.loads(line)

                key = (
                    room,
                    int(r["eval_index"]),
                )

                if key in out:
                    raise RuntimeError(
                        f"Duplicate key: {key}"
                    )

                out[key] = r

    return out


nc = load_run(TRACE_DIR)
k10 = load_run(K10_DIR)

assert len(nc) == 1260
assert len(k10) == 1260
assert set(nc) == set(k10)


counts = Counter()
target_counts = defaultdict(Counter)
room_counts = defaultdict(Counter)

all_rows = []
reversal_rows = []


for key in sorted(nc):

    a = nc[key]
    b = k10[key]

    # --------------------------------------------------
    # Strict episode pairing.
    # --------------------------------------------------

    assert a["scene"] == b["scene"]
    assert a["scene_type"] == b["scene_type"]
    assert a["target"] == b["target"]

    assert (
        canonical_start_state(
            a["start_state"]
        )
        ==
        canonical_start_state(
            b["start_state"]
        )
    )

    nc_success = int(
        bool(a["success"])
    )

    restore_success = int(
        bool(b["success"])
    )

    # --------------------------------------------------
    # Potential-outcome group.
    #
    # NoContext = continue suppressing after t=10
    # Early10   = restore Context at t=10
    # --------------------------------------------------

    if (
        nc_success == 0
        and restore_success == 0
    ):
        outcome = "N00_both_fail"
        label = None

    elif (
        nc_success == 0
        and restore_success == 1
    ):
        outcome = "N01_restore_help"
        label = 1

    elif (
        nc_success == 1
        and restore_success == 0
    ):
        outcome = "N10_restore_harm"
        label = 0

    else:
        outcome = "N11_both_success"
        label = None

    counts[outcome] += 1
    target_counts[a["target"]][outcome] += 1
    room_counts[a["scene_type"]][outcome] += 1

    trace = a.get(
        "context_trace",
        []
    )

    row = {
        "scene_type":
            a["scene_type"],

        "eval_index":
            int(a["eval_index"]),

        "scene":
            a["scene"],

        "target":
            a["target"],

        "start_state":
            a["start_state"],

        "nc_success":
            nc_success,

        "restore_success":
            restore_success,

        "outcome":
            outcome,

        "restore_label":
            label,

        "ep_length":
            int(a["ep_length"]),

        "trace":
            trace,
    }

    all_rows.append(row)

    # --------------------------------------------------
    # Informative reversal episodes.
    # --------------------------------------------------

    if label is not None:

        # Every informative episode must reach
        # the t=10 intervention state.
        if len(trace) < 11:
            raise RuntimeError(
                f"Discordant episode "
                f"does not reach t=10: "
                f"{key}, trace_len={len(trace)}"
            )

        if int(
            trace[10]["step"]
        ) != 10:
            raise RuntimeError(
                f"Invalid t=10 trace "
                f"for {key}"
            )

        reversal_rows.append(row)


# ======================================================
# Save JSONL datasets.
# ======================================================

with open(
    OUT_DIR / "all_1260.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for r in all_rows:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False
            )
            + "\n"
        )


with open(
    OUT_DIR / "reversal_79.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for r in reversal_rows:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False
            )
            + "\n"
        )


# ======================================================
# Lightweight metadata CSV.
# Do NOT flatten trace yet.
# ======================================================

csv_fields = [
    "scene_type",
    "eval_index",
    "scene",
    "target",
    "nc_success",
    "restore_success",
    "outcome",
    "restore_label",
    "ep_length",
]

with open(
    OUT_DIR / "reversal_79_metadata.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields
    )

    writer.writeheader()

    for r in reversal_rows:

        writer.writerow(
            {
                k: r[k]
                for k in csv_fields
            }
        )


# ======================================================
# Summary.
# ======================================================

summary = {
    "total_episodes":
        len(all_rows),

    "reversal_episodes":
        len(reversal_rows),

    "outcomes":
        dict(counts),

    "per_target":
        {
            k: dict(v)
            for k, v
            in sorted(
                target_counts.items()
            )
        },

    "per_room":
        {
            k: dict(v)
            for k, v
            in sorted(
                room_counts.items()
            )
        },
}


with open(
    OUT_DIR / "summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ======================================================
# Print validation summary.
# ======================================================

print("=" * 90)
print("K10 MATCHED DYNAMIC DATASET")
print("=" * 90)

print(
    "Total episodes :",
    len(all_rows)
)

print(
    "Reversals      :",
    len(reversal_rows)
)

print()

for name in [
    "N00_both_fail",
    "N01_restore_help",
    "N10_restore_harm",
    "N11_both_success",
]:
    print(
        f"{name:<22}: "
        f"{counts[name]}"
    )


print()
print("PER TARGET")
print(
    f"{'Target':<16}"
    f"{'Help':>8}"
    f"{'Harm':>8}"
    f"{'Total':>8}"
)

print("-" * 40)

for target in sorted(
    target_counts
):

    c = target_counts[
        target
    ]

    help_n = c[
        "N01_restore_help"
    ]

    harm_n = c[
        "N10_restore_harm"
    ]

    print(
        f"{target:<16}"
        f"{help_n:>8d}"
        f"{harm_n:>8d}"
        f"{help_n + harm_n:>8d}"
    )


print()
print("PER ROOM")
print(
    f"{'Room':<16}"
    f"{'Help':>8}"
    f"{'Harm':>8}"
    f"{'Total':>8}"
)

print("-" * 40)

for room in sorted(
    room_counts
):

    c = room_counts[
        room
    ]

    help_n = c[
        "N01_restore_help"
    ]

    harm_n = c[
        "N10_restore_harm"
    ]

    print(
        f"{room:<16}"
        f"{help_n:>8d}"
        f"{harm_n:>8d}"
        f"{help_n + harm_n:>8d}"
    )


# Known matched K10 outcome totals.
assert counts[
    "N00_both_fail"
] == 826

assert counts[
    "N01_restore_help"
] == 36

assert counts[
    "N10_restore_harm"
] == 43

assert counts[
    "N11_both_success"
] == 355

assert len(
    reversal_rows
) == 79

print()
print(
    "K10 DYNAMIC DATASET: PASS"
)
