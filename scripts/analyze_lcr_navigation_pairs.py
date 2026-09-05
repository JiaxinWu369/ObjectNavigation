import argparse
import json
import os
from glob import glob

import numpy as np
from scipy.stats import binomtest


parser = argparse.ArgumentParser()

parser.add_argument(
    "--base",
    required=True,
)

parser.add_argument(
    "--rule",
    required=True,
)

parser.add_argument(
    "--learned",
    required=True,
)

args = parser.parse_args()


def room_from_path(path):
    name = os.path.basename(path)

    assert (
        name.startswith(
            "episodes_"
        )
        and
        name.endswith(
            ".jsonl"
        )
    )

    return name[
        len("episodes_"):
        -len(".jsonl")
    ]


def load_root(root):

    rows = {}

    files = sorted(
        glob(
            os.path.join(
                root,
                "episodes_*.jsonl",
            )
        )
    )

    if not files:
        raise RuntimeError(
            "No episode files: "
            + root
        )

    for path in files:

        room = room_from_path(
            path
        )

        with open(
            path,
            encoding="utf-8",
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                r = json.loads(
                    line
                )

                if (
                    "eval_index"
                    not in r
                ):
                    raise RuntimeError(
                        "eval_index missing. "
                        "Available keys: "
                        + str(
                            sorted(
                                r.keys()
                            )
                        )
                    )

                if (
                    "success"
                    not in r
                ):
                    raise RuntimeError(
                        "success missing. "
                        "Available keys: "
                        + str(
                            sorted(
                                r.keys()
                            )
                        )
                    )

                key = (
                    room,
                    int(
                        r[
                            "eval_index"
                        ]
                    ),
                )

                if key in rows:
                    raise RuntimeError(
                        "Duplicate key: "
                        + str(key)
                    )

                rows[key] = r

    return rows


def value(
    row,
    key,
    default=None,
):
    x = row.get(
        key,
        default,
    )

    if x is None:
        return default

    return x


def success(row):
    return int(
        bool(
            row["success"]
        )
    )


def spl(row):
    x = row.get(
        "spl",
        np.nan,
    )
    return float(x)


def length(row):

    for k in [
        "ep_length",
        "episode_length",
        "length",
    ]:
        if k in row:
            return float(
                row[k]
            )

    return np.nan


def pair_stats(
    name_a,
    A,
    name_b,
    B,
):

    keys_a = set(A)
    keys_b = set(B)

    if keys_a != keys_b:

        print(
            "A-only:",
            len(
                keys_a - keys_b
            ),
        )

        print(
            "B-only:",
            len(
                keys_b - keys_a
            ),
        )

        raise RuntimeError(
            "Episode mapping mismatch"
        )

    keys = sorted(
        keys_a
    )

    assert len(keys) == 1260, (
        len(keys)
    )

    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    repairs = []
    harms = []

    for key in keys:

        a = success(
            A[key]
        )

        b = success(
            B[key]
        )

        if a == 0 and b == 0:
            n00 += 1

        elif a == 0 and b == 1:
            n01 += 1
            repairs.append(
                key
            )

        elif a == 1 and b == 0:
            n10 += 1
            harms.append(
                key
            )

        elif a == 1 and b == 1:
            n11 += 1

    discord = (
        n01 + n10
    )

    p = (
        1.0
        if discord == 0
        else
        float(
            binomtest(
                n01,
                discord,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )

    sr_a = (
        n10 + n11
    ) / 1260.0

    sr_b = (
        n01 + n11
    ) / 1260.0

    union = (
        n01
        + n10
        + n11
    ) / 1260.0

    oracle_headroom = (
        union - sr_b
    )

    spl_a = np.asarray(
        [
            spl(
                A[k]
            )
            for k in keys
        ],
        dtype=float,
    )

    spl_b = np.asarray(
        [
            spl(
                B[k]
            )
            for k in keys
        ],
        dtype=float,
    )

    len_a = np.asarray(
        [
            length(
                A[k]
            )
            for k in keys
        ],
        dtype=float,
    )

    len_b = np.asarray(
        [
            length(
                B[k]
            )
            for k in keys
        ],
        dtype=float,
    )

    print()
    print("=" * 78)
    print(
        "{}  vs  {}".format(
            name_a,
            name_b,
        )
    )
    print("=" * 78)

    print(
        "N00 both fail       :",
        n00,
    )

    print(
        "N01 B-only / repair :",
        n01,
    )

    print(
        "N10 A-only / harm   :",
        n10,
    )

    print(
        "N11 both success    :",
        n11,
    )

    print(
        "discordant          :",
        discord,
    )

    print(
        "McNemar exact p     :",
        p,
    )

    print()

    print(
        "{} SR              : {:.6f}".format(
            name_a,
            sr_a,
        )
    )

    print(
        "{} SR              : {:.6f}".format(
            name_b,
            sr_b,
        )
    )

    print(
        "Delta B-A           : {:+.4f} pp".format(
            (
                sr_b
                - sr_a
            )
            * 100.0
        )
    )

    print(
        "Union oracle SR     : {:.4f}%".format(
            union * 100.0
        )
    )

    print(
        "Oracle headroom B   : {:.4f} pp".format(
            oracle_headroom
            * 100.0
        )
    )

    if (
        np.all(
            np.isfinite(
                spl_a
            )
        )
        and
        np.all(
            np.isfinite(
                spl_b
            )
        )
    ):
        print()
        print(
            "{} SPL            : {:.6f}".format(
                name_a,
                spl_a.mean(),
            )
        )

        print(
            "{} SPL            : {:.6f}".format(
                name_b,
                spl_b.mean(),
            )
        )

        print(
            "Delta SPL B-A      : {:+.4f} pp".format(
                (
                    spl_b.mean()
                    - spl_a.mean()
                )
                * 100.0
            )
        )

    if (
        np.all(
            np.isfinite(
                len_a
            )
        )
        and
        np.all(
            np.isfinite(
                len_b
            )
        )
    ):
        print()
        print(
            "{} Length         : {:.4f}".format(
                name_a,
                len_a.mean(),
            )
        )

        print(
            "{} Length         : {:.4f}".format(
                name_b,
                len_b.mean(),
            )
        )

        print(
            "Delta Length B-A   : {:+.4f}".format(
                len_b.mean()
                - len_a.mean()
            )
        )

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "p": p,
        "repairs": repairs,
        "harms": harms,
    }


base = load_root(
    args.base
)

rule = load_root(
    args.rule
)

learned = load_root(
    args.learned
)


print(
    "Base episodes   :",
    len(base),
)

print(
    "Rule episodes   :",
    len(rule),
)

print(
    "Learned episodes:",
    len(learned),
)


r1 = pair_stats(
    "Base",
    base,
    "Learned",
    learned,
)

r2 = pair_stats(
    "Rule",
    rule,
    "Learned",
    learned,
)


print()
print("=" * 78)
print("LEARNED-LCR NAVIGATION GATE")
print("=" * 78)

delta_sr = (
    (
        sum(
            success(x)
            for x in learned.values()
        )
        -
        sum(
            success(x)
            for x in base.values()
        )
    )
    / 1260.0
    * 100.0
)

base_spl = np.mean(
    [
        spl(x)
        for x in base.values()
    ]
)

learned_spl = np.mean(
    [
        spl(x)
        for x in learned.values()
    ]
)

delta_spl = (
    learned_spl
    - base_spl
) * 100.0


go_sr = (
    delta_sr >= 0.5
)

go_eff = (
    delta_spl >= 1.0
    and
    delta_sr >= -0.2
)

print(
    "Delta SR  : {:+.4f} pp".format(
        delta_sr
    )
)

print(
    "Delta SPL : {:+.4f} pp".format(
        delta_spl
    )
)

print(
    "SR criterion        :",
    go_sr,
)

print(
    "Efficiency criterion:",
    go_eff,
)

print()

if (
    go_sr
    or go_eff
):
    print(
        "FINAL NAVIGATION GATE: GO"
    )
else:
    print(
        "FINAL NAVIGATION GATE: NO-GO"
    )
