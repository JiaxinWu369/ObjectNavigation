import csv
import json
import math
from pathlib import Path

ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

K10_ROOT = Path("results/early10_500_zsval")
K20_ROOT = Path("results/early20_500_zsval")

OUT_DIR = Path("figures/fig1_candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(root):
    data = {}

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)

                key = (
                    room,
                    int(r["eval_index"]),
                )

                data[key] = r

    assert len(data) == 1260
    return data


def xz(state):
    """
    AKGVP state:
        x | z | rotation | horizon
    Only x,z are needed for top-down trajectories.
    """
    p = str(state).split("|")

    return (
        float(p[0]),
        float(p[1]),
    )


def max_separation(a, b, start=10):
    n = min(len(a), len(b))

    if n <= start:
        return 0.0

    ds = []

    for i in range(start, n):
        ax, az = xz(a[i])
        bx, bz = xz(b[i])

        ds.append(
            math.hypot(
                ax - bx,
                az - bz,
            )
        )

    return max(ds) if ds else 0.0


def final_separation(a, b):
    ax, az = xz(a[-1])
    bx, bz = xz(b[-1])

    return math.hypot(
        ax - bx,
        az - bz,
    )


A = load(K10_ROOT)
B = load(K20_ROOT)

assert set(A) == set(B)

rows = []

for key in sorted(A):
    a = A[key]
    b = B[key]

    assert a["scene"] == b["scene"]
    assert a["target"] == b["target"]

    ya = int(bool(a["success"]))
    yb = int(bool(b["success"]))

    # Only outcome reversals are useful for Fig. 1.
    if ya == yb:
        continue

    aa = a["full_actions"]
    ba = b["full_actions"]

    sa = a["full_states"]
    sb = b["full_states"]

    if len(sa) <= 10 or len(sb) <= 10:
        continue

    # Matched intervention check.
    assert aa[:10] == ba[:10]
    assert sa[10] == sb[10]

    if ya == 1 and yb == 0:
        case_type = "restore_help"

    elif ya == 0 and yb == 1:
        case_type = "delay_help"

    else:
        continue

    max_sep = max_separation(
        sa,
        sb,
        start=10,
    )

    final_sep = final_separation(
        sa,
        sb,
    )

    len10 = len(aa)
    len20 = len(ba)

    # Only used for visual-case ranking.
    # Strong branch divergence is preferred.
    visual_score = (
        max_sep
        + 0.25 * final_sep
        + 0.002 * min(
            len10 + len20,
            200,
        )
    )

    rows.append({
        "case_type": case_type,
        "room": key[0],
        "eval_index": key[1],
        "scene": a["scene"],
        "target": a["target"],

        "k10_success": ya,
        "k20_success": yb,

        "k10_length": a["ep_length"],
        "k20_length": b["ep_length"],

        "max_branch_separation":
            round(max_sep, 4),

        "final_separation":
            round(final_sep, 4),

        "visual_score":
            round(visual_score, 4),

        "decision_state":
            sa[10],
    })


assert sum(
    x["case_type"] == "restore_help"
    for x in rows
) == 32

assert sum(
    x["case_type"] == "delay_help"
    for x in rows
) == 25


rows = sorted(
    rows,
    key=lambda x: (
        x["case_type"],
        -x["visual_score"],
    ),
)


out = OUT_DIR / "fig1_candidate_cases.csv"

with open(
    out,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    w.writeheader()
    w.writerows(rows)


for case_type in [
    "restore_help",
    "delay_help",
]:

    subset = [
        x for x in rows
        if x["case_type"] == case_type
    ]

    print()
    print("=" * 110)
    print(case_type.upper())
    print("=" * 110)

    print(
        f"{'rank':<5}"
        f"{'room':<14}"
        f"{'scene':<15}"
        f"{'target':<16}"
        f"{'idx':>6}"
        f"{'K10':>6}"
        f"{'K20':>6}"
        f"{'maxSep':>10}"
        f"{'score':>10}"
    )

    print("-" * 90)

    for i, x in enumerate(
        subset[:12],
        1,
    ):
        print(
            f"{i:<5}"
            f"{x['room']:<14}"
            f"{x['scene']:<15}"
            f"{x['target']:<16}"
            f"{x['eval_index']:>6}"
            f"{x['k10_length']:>6}"
            f"{x['k20_length']:>6}"
            f"{x['max_branch_separation']:>10.3f}"
            f"{x['visual_score']:>10.3f}"
        )


print()
print("Saved:", out)
print("Total :", len(rows))
print("FIG1 CASE SELECTION: PASS")
