import json
from glob import glob

import numpy as np


ROOTS = {
    "Base": "results/base500_zsval",
    "Rule": "results/lcr_rule500_zsval",
    "Learned": "results/lcr_learned500_zsval",
}

SEED = 2026
N_BOOT = 20000


def room_from_path(path):
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
        room = room_from_path(path)

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
                    int(r["eval_index"]),
                )

                rows[key] = r

    return rows


def success(r):
    return int(
        bool(r["success"])
    )


def spl(r):
    return float(
        r["spl"]
    )


def length(r):
    for k in [
        "ep_length",
        "episode_length",
        "length",
    ]:
        if k in r:
            return float(r[k])

    raise KeyError(
        "No length key: "
        + str(sorted(r.keys()))
    )


data = {
    name: load(root)
    for name, root in ROOTS.items()
}

keys = sorted(
    data["Base"]
)

assert len(keys) == 1260
assert (
    set(keys)
    == set(data["Rule"])
    == set(data["Learned"])
)


rng = np.random.default_rng(
    SEED
)


def bootstrap_mean(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    n = len(x)

    boot = np.empty(
        N_BOOT,
        dtype=float,
    )

    for i in range(
        N_BOOT
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = (
            x[idx].mean()
        )

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5],
    )

    return (
        float(x.mean()),
        float(lo),
        float(hi),
    )


def analyze(A_name, B_name):

    A = data[A_name]
    B = data[B_name]

    groups = {
        "N00": [],
        "N01": [],
        "N10": [],
        "N11": [],
    }

    all_delta_spl = []

    for k in keys:

        a = success(A[k])
        b = success(B[k])

        if a == 0 and b == 0:
            g = "N00"

        elif a == 0 and b == 1:
            g = "N01"

        elif a == 1 and b == 0:
            g = "N10"

        else:
            g = "N11"

        dspl = (
            spl(B[k])
            - spl(A[k])
        )

        dlen = (
            length(B[k])
            - length(A[k])
        )

        groups[g].append(
            {
                "key": k,
                "delta_spl": dspl,
                "delta_length": dlen,
                "spl_A": spl(A[k]),
                "spl_B": spl(B[k]),
                "len_A": length(A[k]),
                "len_B": length(B[k]),
            }
        )

        all_delta_spl.append(
            dspl
        )

    print()
    print("=" * 92)
    print(
        "{} -> {}".format(
            A_name,
            B_name,
        )
    )
    print("=" * 92)

    for g in [
        "N00",
        "N01",
        "N10",
        "N11",
    ]:
        print(
            "{}: {}".format(
                g,
                len(groups[g]),
            )
        )

    print()

    total_delta = (
        np.mean(
            all_delta_spl
        )
    )

    print(
        "Overall Delta SPL : "
        "{:+.4f} pp".format(
            total_delta
            * 100.0
        )
    )

    print()
    print(
        "CONTRIBUTION TO "
        "OVERALL DELTA SPL"
    )

    contribution_sum = 0.0

    for g in [
        "N00",
        "N01",
        "N10",
        "N11",
    ]:

        rows = groups[g]

        contribution = (
            sum(
                x["delta_spl"]
                for x in rows
            )
            / len(keys)
        )

        contribution_sum += (
            contribution
        )

        group_mean = (
            np.mean(
                [
                    x["delta_spl"]
                    for x in rows
                ]
            )
            if rows
            else 0.0
        )

        print(
            "{:<4} "
            "N={:<4} "
            "mean within group={:+.4f} pp  "
            "contribution={:+.4f} pp".format(
                g,
                len(rows),
                group_mean * 100.0,
                contribution * 100.0,
            )
        )

    print(
        "Contribution sum   : "
        "{:+.4f} pp".format(
            contribution_sum
            * 100.0
        )
    )

    # ----------------------------------------------------
    # Most important analysis:
    # Both methods succeeded on exactly the same episodes.
    # ----------------------------------------------------
    common = groups[
        "N11"
    ]

    common_dspl = [
        x["delta_spl"]
        for x in common
    ]

    common_dlen = [
        x["delta_length"]
        for x in common
    ]

    spl_ci = bootstrap_mean(
        common_dspl
    )

    len_ci = bootstrap_mean(
        common_dlen
    )

    print()
    print(
        "COMMON-SUCCESS ANALYSIS "
        "(N11)"
    )
    print(
        "N:",
        len(common),
    )

    print(
        "Delta SPL    : "
        "{:+.4f} pp "
        "95% CI "
        "[{:+.4f}, {:+.4f}]".format(
            spl_ci[0] * 100.0,
            spl_ci[1] * 100.0,
            spl_ci[2] * 100.0,
        )
    )

    print(
        "Delta Length : "
        "{:+.4f} "
        "95% CI "
        "[{:+.4f}, {:+.4f}]".format(
            len_ci[0],
            len_ci[1],
            len_ci[2],
        )
    )

    positive = sum(
        x > 1e-12
        for x in common_dspl
    )

    negative = sum(
        x < -1e-12
        for x in common_dspl
    )

    equal = (
        len(common_dspl)
        - positive
        - negative
    )

    print()
    print(
        "Common-success SPL:"
    )

    print(
        "B better:",
        positive,
    )

    print(
        "A better:",
        negative,
    )

    print(
        "equal   :",
        equal,
    )


for pair in [
    ("Base", "Rule"),
    ("Rule", "Learned"),
    ("Base", "Learned"),
]:
    analyze(
        pair[0],
        pair[1],
    )
