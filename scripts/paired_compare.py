import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]


def load(root):
    root = Path(root)
    rows = []

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        cur = [
            json.loads(x)
            for x in open(p, encoding="utf-8")
            if x.strip()
        ]

        for i, x in enumerate(cur):
            x["_room"] = room
            x["_index"] = i

        rows.extend(cur)

    return rows


def bootstrap_ci(x, n_boot=20000, seed=0):
    x = np.asarray(x, dtype=np.float64)

    rng = np.random.default_rng(seed)
    n = len(x)

    vals = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )
        vals[i] = x[idx].mean()

    return np.percentile(
        vals,
        [2.5, 97.5],
    )


ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--name-a", default="A")
ap.add_argument("--name-b", default="B")
args = ap.parse_args()

A = load(args.a)
B = load(args.b)

assert len(A) == len(B), (
    len(A),
    len(B),
)

for i, (a, b) in enumerate(zip(A, B)):
    assert (
        a["_room"],
        a["_index"],
    ) == (
        b["_room"],
        b["_index"],
    )

print("PAIR ALIGNMENT: PASS", len(A))

sa = np.array(
    [int(x["success"]) for x in A]
)

sb = np.array(
    [int(x["success"]) for x in B]
)

spla = np.array(
    [float(x["spl"]) for x in A]
)

splb = np.array(
    [float(x["spl"]) for x in B]
)

la = np.array(
    [float(x["ep_length"]) for x in A]
)

lb = np.array(
    [float(x["ep_length"]) for x in B]
)

n00 = int(((sa == 0) & (sb == 0)).sum())
n01 = int(((sa == 0) & (sb == 1)).sum())
n10 = int(((sa == 1) & (sb == 0)).sum())
n11 = int(((sa == 1) & (sb == 1)).sum())

print()
print(args.name_a, "vs", args.name_b)
print("N00/N01/N10/N11:",
      n00, n01, n10, n11)

if n01 + n10:
    p = binomtest(
        min(n01, n10),
        n01 + n10,
        0.5,
    ).pvalue
else:
    p = 1.0

print("McNemar exact p:", p)

dsr = (sb.mean() - sa.mean()) * 100

dspl = (splb - spla) * 100
dlen = lb - la

ci_spl = bootstrap_ci(dspl)
ci_len = bootstrap_ci(dlen)

print()
print(
    "Delta SR (pp):",
    round(dsr, 4),
)

print(
    "Delta SPL (pp):",
    round(dspl.mean(), 4),
    "95% CI",
    np.round(ci_spl, 4),
)

print(
    "Delta Len:",
    round(dlen.mean(), 4),
    "95% CI",
    np.round(ci_len, 4),
)

common = (
    (sa == 1)
    & (sb == 1)
)

print()
print(
    "Common-success N:",
    int(common.sum()),
)

if common.sum():

    cs_spl = dspl[common]
    cs_len = dlen[common]

    print(
        "Common-success Delta SPL (pp):",
        round(cs_spl.mean(), 4),
        "95% CI",
        np.round(
            bootstrap_ci(cs_spl),
            4,
        ),
    )

    print(
        "Common-success Delta Len:",
        round(cs_len.mean(), 4),
        "95% CI",
        np.round(
            bootstrap_ci(cs_len),
            4,
        ),
    )
