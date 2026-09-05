import json
from glob import glob
from collections import Counter, defaultdict

import numpy as np


BASE_ROOT = "results/base500_zsval"
LEARNED_ROOT = "results/lcr_learned500_zsval"

WINDOW = 10


def room_name(path):
    name = path.split("/")[-1]
    return name[
        len("episodes_"):
        -len(".jsonl")
    ]


def load(root):
    rows = {}

    for path in sorted(
        glob(
            root + "/episodes_*.jsonl"
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


def success(r):
    return int(
        bool(
            r["success"]
        )
    )


def group_name(b, l):

    a = success(b)
    x = success(l)

    if a == 0 and x == 1:
        return "repair"

    if a == 1 and x == 0:
        return "harm"

    if a == 1 and x == 1:
        return "both_success"

    return "both_fail"


def get_target(row):

    for key in [
        "target",
        "target_object",
        "goal",
        "goal_object_type",
    ]:
        if key in row:
            return str(
                row[key]
            )

    return "UNKNOWN"


def summarize_first_window(ep):

    trace = ep.get(
        "context_trace",
        []
    )

    valid = []

    for item in trace:

        s = item.get(
            "lcr_stats"
        )

        if not s:
            continue

        gate = float(
            s["gate_min"]
        )

        valid.append(
            {
                "step":
                    int(
                        item["step"]
                    ),

                "gate":
                    gate,

                "q":
                    (
                        None
                        if s.get("q")
                        is None
                        else float(
                            s["q"]
                        )
                    ),

                "R":
                    float(
                        s["R_before"]
                    ),

                "new_view":
                    bool(
                        s[
                            "new_viewpoint"
                        ]
                    ),

                "action_probs":
                    item.get(
                        "action_probs"
                    ),
            }
        )

    active = [
        x
        for x in valid
        if x["gate"]
        < 1.0 - 1e-6
    ]

    if not active:
        return None

    first = active[0]

    t0 = first["step"]

    window = [
        x
        for x in valid
        if (
            x["step"] >= t0
            and
            x["step"]
            < t0 + WINDOW
        )
    ]

    if not window:
        return None

    gates = np.asarray(
        [
            x["gate"]
            for x in window
        ],
        dtype=float,
    )

    active_mask = (
        gates
        < 1.0 - 1e-6
    )

    probs = first[
        "action_probs"
    ]

    if probs is not None:

        p = np.asarray(
            probs,
            dtype=float,
        ).reshape(-1)

        p_sorted = np.sort(
            p
        )[::-1]

        action_top1 = float(
            p_sorted[0]
        )

        action_margin = (
            float(
                p_sorted[0]
                - p_sorted[1]
            )
            if len(p_sorted) >= 2
            else np.nan
        )

    else:
        action_top1 = np.nan
        action_margin = np.nan

    return {
        "first_active":
            float(t0),

        "first_gate":
            float(
                first["gate"]
            ),

        "first_q":
            (
                np.nan
                if first["q"]
                is None
                else float(
                    first["q"]
                )
            ),

        "first_R":
            float(
                first["R"]
            ),

        "window_steps":
            int(
                len(window)
            ),

        "window_active_steps":
            int(
                active_mask.sum()
            ),

        "window_mean_gate":
            float(
                gates.mean()
            ),

        "window_min_gate":
            float(
                gates.min()
            ),

        "window_exposure":
            float(
                np.sum(
                    1.0 - gates
                )
            ),

        "first_action_top1":
            action_top1,

        "first_action_margin":
            action_margin,
    }


def effect_size(a, b):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
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
        return np.nan

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

    if pooled < 1e-12:
        return 0.0

    return float(
        (
            np.mean(a)
            - np.mean(b)
        )
        / pooled
    )


base = load(
    BASE_ROOT
)

learned = load(
    LEARNED_ROOT
)

assert set(base) == set(learned)
assert len(base) == 1260


groups = defaultdict(list)

target_counts = {
    "repair":
        Counter(),

    "harm":
        Counter(),
}

room_counts = {
    "repair":
        Counter(),

    "harm":
        Counter(),
}


for key in sorted(base):

    g = group_name(
        base[key],
        learned[key],
    )

    s = summarize_first_window(
        learned[key]
    )

    if s is not None:
        groups[g].append(
            s
        )

    if g in (
        "repair",
        "harm",
    ):
        room = key[0]

        target = get_target(
            learned[key]
        )

        room_counts[
            g
        ][room] += 1

        target_counts[
            g
        ][target] += 1


metrics = [
    "first_active",
    "first_gate",
    "first_q",
    "first_R",
    "window_steps",
    "window_active_steps",
    "window_mean_gate",
    "window_min_gate",
    "window_exposure",
    "first_action_top1",
    "first_action_margin",
]


print("=" * 88)
print(
    "LEARNED-LCR FIXED-WINDOW AUDIT"
)
print(
    "window:",
    WINDOW,
    "steps from first active step",
)
print("=" * 88)


for g in [
    "repair",
    "harm",
]:

    print()
    print(
        "{} N={}".format(
            g.upper(),
            len(
                groups[g]
            ),
        )
    )

    for metric in metrics:

        v = np.asarray(
            [
                x[metric]
                for x in groups[g]
            ],
            dtype=float,
        )

        v = v[
            np.isfinite(v)
        ]

        if len(v) == 0:
            continue

        print(
            "{:<20}: "
            "mean={:.4f} "
            "median={:.4f} "
            "min={:.4f} "
            "max={:.4f}".format(
                metric,
                v.mean(),
                np.median(v),
                v.min(),
                v.max(),
            )
        )


print()
print("=" * 88)
print(
    "REPAIR vs HARM EFFECT SIZE"
)
print("=" * 88)


for metric in metrics:

    a = [
        x[metric]
        for x in groups[
            "repair"
        ]
    ]

    b = [
        x[metric]
        for x in groups[
            "harm"
        ]
    ]

    aa = np.asarray(
        a,
        dtype=float,
    )

    bb = np.asarray(
        b,
        dtype=float,
    )

    aa = aa[
        np.isfinite(aa)
    ]

    bb = bb[
        np.isfinite(bb)
    ]

    if (
        len(aa) < 2
        or len(bb) < 2
    ):
        continue

    d = effect_size(
        aa,
        bb,
    )

    print(
        "{:<20}: "
        "repair={:.4f} "
        "harm={:.4f} "
        "d={:+.3f}".format(
            metric,
            aa.mean(),
            bb.mean(),
            d,
        )
    )


print()
print("=" * 88)
print("TARGET BREAKDOWN")
print("=" * 88)

targets = sorted(
    set(
        target_counts[
            "repair"
        ]
    )
    |
    set(
        target_counts[
            "harm"
        ]
    )
)

for target in targets:

    print(
        "{:<16} repair={:<3} harm={:<3} net={:+d}".format(
            target,
            target_counts[
                "repair"
            ][target],
            target_counts[
                "harm"
            ][target],
            (
                target_counts[
                    "repair"
                ][target]
                -
                target_counts[
                    "harm"
                ][target]
            ),
        )
    )


print()
print("=" * 88)
print("ROOM BREAKDOWN")
print("=" * 88)

rooms = sorted(
    set(
        room_counts[
            "repair"
        ]
    )
    |
    set(
        room_counts[
            "harm"
        ]
    )
)

for room in rooms:

    print(
        "{:<16} repair={:<3} harm={:<3} net={:+d}".format(
            room,
            room_counts[
                "repair"
            ][room],
            room_counts[
                "harm"
            ][room],
            (
                room_counts[
                    "repair"
                ][room]
                -
                room_counts[
                    "harm"
                ][room]
            ),
        )
    )
