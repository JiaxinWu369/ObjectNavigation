import json
from glob import glob
from collections import defaultdict

import numpy as np


BASE_ROOT = "results/base500_zsval"
LEARNED_ROOT = "results/lcr_learned500_zsval"


def room_name(path):
    x = path.split("/")[-1]
    return x[
        len("episodes_"):
        -len(".jsonl")
    ]


def load(root):
    rows = {}

    for path in sorted(
        glob(
            root
            + "/episodes_*.jsonl"
        )
    ):
        room = room_name(path)

        with open(
            path,
            encoding="utf-8",
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                r = json.loads(line)

                key = (
                    room,
                    int(
                        r["eval_index"]
                    ),
                )

                rows[key] = r

    return rows


base = load(BASE_ROOT)
learned = load(LEARNED_ROOT)

assert set(base) == set(learned)
assert len(base) == 1260


def outcome_group(b, l):

    bs = int(
        bool(
            b["success"]
        )
    )

    ls = int(
        bool(
            l["success"]
        )
    )

    if bs == 0 and ls == 1:
        return "repair"

    if bs == 1 and ls == 0:
        return "harm"

    if bs == 1 and ls == 1:
        return "both_success"

    return "both_fail"


def trace_summary(ep):

    trace = ep.get(
        "context_trace",
        []
    )

    gates = []
    Rs = []
    qs = []

    active_steps = []
    active_regions = set()

    for x in trace:

        s = x.get(
            "lcr_stats"
        )

        if not s:
            continue

        gate = float(
            s["gate_min"]
        )

        R = float(
            s["R_before"]
        )

        q = s.get("q")

        gates.append(gate)
        Rs.append(R)

        if q is not None:
            qs.append(
                float(q)
            )

        if gate < 1.0 - 1e-6:

            active_steps.append(
                int(
                    x["step"]
                )
            )

            active_regions.add(
                tuple(
                    s[
                        "region_key"
                    ]
                )
            )

    if not gates:
        return None

    return {
        "first_active":
            (
                min(active_steps)
                if active_steps
                else np.nan
            ),

        "active_steps":
            len(active_steps),

        "active_regions":
            len(active_regions),

        "mean_R":
            float(
                np.mean(Rs)
            ),

        "min_R":
            float(
                np.min(Rs)
            ),

        "exposure":
            float(
                np.sum(
                    1.0
                    - np.asarray(
                        gates
                    )
                )
            ),

        "mean_q":
            (
                float(
                    np.mean(qs)
                )
                if qs
                else np.nan
            ),

        "min_q":
            (
                float(
                    np.min(qs)
                )
                if qs
                else np.nan
            ),

        "first_q":
            (
                float(qs[0])
                if qs
                else np.nan
            ),
    }


groups = defaultdict(list)

for key in sorted(base):

    g = outcome_group(
        base[key],
        learned[key],
    )

    s = trace_summary(
        learned[key]
    )

    if s is not None:
        groups[g].append(s)


metrics = [
    "first_active",
    "active_steps",
    "active_regions",
    "mean_R",
    "min_R",
    "exposure",
    "first_q",
    "mean_q",
    "min_q",
]


print("=" * 88)
print("LEARNED-LCR REPAIR / HARM AUDIT")
print("=" * 88)

for g in [
    "repair",
    "harm",
    "both_success",
    "both_fail",
]:

    rows = groups[g]

    print()
    print(
        "{}  N={}".format(
            g.upper(),
            len(rows),
        )
    )

    for metric in metrics:

        v = np.asarray(
            [
                r[metric]
                for r in rows
            ],
            dtype=float,
        )

        v = v[
            np.isfinite(v)
        ]

        if len(v) == 0:
            print(
                f"{metric:<16}: NA"
            )
            continue

        print(
            "{:<16}: mean={:.4f} "
            "median={:.4f} "
            "min={:.4f} "
            "max={:.4f}".format(
                metric,
                np.mean(v),
                np.median(v),
                np.min(v),
                np.max(v),
            )
        )


print()
print("=" * 88)
print("PAIRWISE REPAIR vs HARM EFFECT SIZE")
print("=" * 88)

for metric in metrics:

    a = np.asarray(
        [
            r[metric]
            for r in groups[
                "repair"
            ]
        ],
        dtype=float,
    )

    b = np.asarray(
        [
            r[metric]
            for r in groups[
                "harm"
            ]
        ],
        dtype=float,
    )

    a = a[
        np.isfinite(a)
    ]

    b = b[
        np.isfinite(b)
    ]

    if (
        len(a) < 2
        or len(b) < 2
    ):
        continue

    pooled = np.sqrt(
        (
            (
                len(a) - 1
            )
            * np.var(
                a,
                ddof=1,
            )
            +
            (
                len(b) - 1
            )
            * np.var(
                b,
                ddof=1,
            )
        )
        /
        (
            len(a)
            + len(b)
            - 2
        )
    )

    d = (
        (
            np.mean(a)
            - np.mean(b)
        )
        / pooled
        if pooled > 1e-12
        else 0.0
    )

    print(
        "{:<16}: "
        "repair={:.4f} "
        "harm={:.4f} "
        "Cohen_d={:+.3f}".format(
            metric,
            np.mean(a),
            np.mean(b),
            d,
        )
    )
