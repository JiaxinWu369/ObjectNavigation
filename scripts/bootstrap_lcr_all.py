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
    return name[len("episodes_"):-len(".jsonl")]


def load(root):
    rows = {}

    for path in sorted(
        glob(root + "/episodes_*.jsonl")
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

                if key in rows:
                    raise RuntimeError(
                        "duplicate key: {}".format(
                            key
                        )
                    )

                rows[key] = r

    return rows


def get_length(r):
    for key in [
        "ep_length",
        "episode_length",
        "length",
    ]:
        if key in r:
            return float(r[key])

    raise KeyError(
        "length key missing: {}".format(
            sorted(r.keys())
        )
    )


def success(r):
    return float(bool(r["success"]))


data = {
    name: load(root)
    for name, root in ROOTS.items()
}

key_sets = [
    set(x)
    for x in data.values()
]

assert all(
    x == key_sets[0]
    for x in key_sets[1:]
)

keys = sorted(key_sets[0])

assert len(keys) == 1260, len(keys)


def arrays(name):
    rows = data[name]

    return {
        "success": np.asarray(
            [
                success(rows[k])
                for k in keys
            ],
            dtype=float,
        ),

        "spl": np.asarray(
            [
                float(rows[k]["spl"])
                for k in keys
            ],
            dtype=float,
        ),

        "length": np.asarray(
            [
                get_length(rows[k])
                for k in keys
            ],
            dtype=float,
        ),
    }


arr = {
    name: arrays(name)
    for name in ROOTS
}


print("=" * 84)
print("METHOD SUMMARY")
print("=" * 84)

for name in [
    "Base",
    "Rule",
    "Learned",
]:
    x = arr[name]

    print(
        "{:<10} "
        "SR={:.4f}%  "
        "SPL={:.4f}%  "
        "Length={:.4f}".format(
            name,
            x["success"].mean() * 100.0,
            x["spl"].mean() * 100.0,
            x["length"].mean(),
        )
    )


rng = np.random.default_rng(SEED)


def bootstrap_delta(
    a,
    b,
):
    """
    B - A
    """

    delta = b - a
    n = len(delta)

    boot = np.empty(
        N_BOOT,
        dtype=float,
    )

    for i in range(N_BOOT):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = delta[idx].mean()

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5],
    )

    return (
        float(delta.mean()),
        float(lo),
        float(hi),
    )


def exact_pair_counts(a, b):

    a = a.astype(int)
    b = b.astype(int)

    n00 = int(
        np.sum(
            (a == 0)
            &
            (b == 0)
        )
    )

    n01 = int(
        np.sum(
            (a == 0)
            &
            (b == 1)
        )
    )

    n10 = int(
        np.sum(
            (a == 1)
            &
            (b == 0)
        )
    )

    n11 = int(
        np.sum(
            (a == 1)
            &
            (b == 1)
        )
    )

    return n00, n01, n10, n11


pairs = [
    ("Base", "Rule"),
    ("Rule", "Learned"),
    ("Base", "Learned"),
]


for A, B in pairs:

    print()
    print("=" * 84)
    print("{} -> {}".format(A, B))
    print("=" * 84)

    sr = bootstrap_delta(
        arr[A]["success"],
        arr[B]["success"],
    )

    spl = bootstrap_delta(
        arr[A]["spl"],
        arr[B]["spl"],
    )

    length = bootstrap_delta(
        arr[A]["length"],
        arr[B]["length"],
    )

    n00, n01, n10, n11 = (
        exact_pair_counts(
            arr[A]["success"],
            arr[B]["success"],
        )
    )

    print(
        "Delta SR     : {:+.4f} pp "
        "95% CI [{:+.4f}, {:+.4f}]".format(
            sr[0] * 100.0,
            sr[1] * 100.0,
            sr[2] * 100.0,
        )
    )

    print(
        "Delta SPL    : {:+.4f} pp "
        "95% CI [{:+.4f}, {:+.4f}]".format(
            spl[0] * 100.0,
            spl[1] * 100.0,
            spl[2] * 100.0,
        )
    )

    print(
        "Delta Length : {:+.4f} "
        "95% CI [{:+.4f}, {:+.4f}]".format(
            length[0],
            length[1],
            length[2],
        )
    )

    print()
    print(
        "N00={} N01={} N10={} N11={}".format(
            n00,
            n01,
            n10,
            n11,
        )
    )

    print(
        "B-only repair:",
        n01,
    )

    print(
        "A-only harm  :",
        n10,
    )


print()
print("=" * 84)
print("EFFICIENCY TREND")
print("=" * 84)

base_spl = arr[
    "Base"
]["spl"].mean()

rule_spl = arr[
    "Rule"
]["spl"].mean()

learned_spl = arr[
    "Learned"
]["spl"].mean()

print(
    "Base    SPL:",
    base_spl * 100.0,
)

print(
    "Rule    SPL:",
    rule_spl * 100.0,
)

print(
    "Learned SPL:",
    learned_spl * 100.0,
)

if (
    base_spl
    < rule_spl
    < learned_spl
):
    print(
        "Numerical trend: "
        "Base < Rule < Learned"
    )
else:
    print(
        "No monotonic SPL trend"
    )
