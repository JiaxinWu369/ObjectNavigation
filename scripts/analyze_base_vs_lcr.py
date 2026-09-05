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

BASE = Path(
    "results/base500_zsval"
)

LCR = Path(
    "results/lcr_constant500_zsval"
)

N_BOOT = 20000
SEED = 20260828


def load(root):

    out = {}

    for room in ROOMS:

        p = (
            root
            / f"episodes_{room}.jsonl"
        )

        if not p.exists():
            raise FileNotFoundError(p)

        for line in open(
            p,
            encoding="utf-8",
        ):

            if not line.strip():
                continue

            r = json.loads(line)

            key = (
                room,
                int(r["eval_index"]),
            )

            out[key] = r

    return out


def state_key(s):

    return (
        round(float(s["x"]), 4),
        round(float(s["z"]), 4),
        int(round(
            float(
                s["rotation"]["y"]
            )
        )),
        int(round(
            float(s["horizon"])
        )),
    )


base = load(BASE)
lcr = load(LCR)

assert set(base) == set(lcr)

keys = sorted(base)

for k in keys:

    a = base[k]
    b = lcr[k]

    assert a["scene"] == b["scene"]
    assert a["target"] == b["target"]

    assert (
        state_key(
            a["start_state"]
        )
        ==
        state_key(
            b["start_state"]
        )
    )

print(
    "PAIR ALIGNMENT: PASS",
    len(keys),
)


def arr(
    rows,
    field,
):

    return np.asarray(
        [
            float(
                rows[k][field]
            )
            for k in keys
        ]
    )


sa = np.asarray(
    [
        int(
            bool(
                base[k]["success"]
            )
        )
        for k in keys
    ]
)

sb = np.asarray(
    [
        int(
            bool(
                lcr[k]["success"]
            )
        )
        for k in keys
    ]
)

spl_a = arr(base, "spl")
spl_b = arr(lcr, "spl")

len_a = arr(base, "ep_length")
len_b = arr(lcr, "ep_length")


n00 = int(
    np.sum(
        (sa == 0)
        &
        (sb == 0)
    )
)

n01 = int(
    np.sum(
        (sa == 0)
        &
        (sb == 1)
    )
)

n10 = int(
    np.sum(
        (sa == 1)
        &
        (sb == 0)
    )
)

n11 = int(
    np.sum(
        (sa == 1)
        &
        (sb == 1)
    )
)

discord = n01 + n10

p = (
    binomtest(
        min(n01, n10),
        discord,
        p=0.5,
    ).pvalue
    if discord
    else 1.0
)

print()
print(
    "N00/N01/N10/N11:",
    n00,
    n01,
    n10,
    n11,
)

print(
    "McNemar exact p:",
    p,
)

print(
    "Delta SR (pp):",
    100
    * (
        sb.mean()
        -
        sa.mean()
    ),
)


rng = np.random.default_rng(
    SEED
)


def bootstrap(
    a,
    b,
    scale=1.0,
):

    d = b - a

    n = len(d)

    boot = np.empty(
        N_BOOT
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
            d[idx].mean()
        )

    return (
        scale * d.mean(),
        scale * np.percentile(
            boot,
            2.5,
        ),
        scale * np.percentile(
            boot,
            97.5,
        ),
    )


dspl = bootstrap(
    spl_a,
    spl_b,
    100.0,
)

dlen = bootstrap(
    len_a,
    len_b,
)


print(
    "Delta SPL (pp): "
    "{:+.4f} "
    "95% CI "
    "[{:+.4f}, {:+.4f}]".format(
        *dspl
    )
)

print(
    "Delta Len: "
    "{:+.4f} "
    "95% CI "
    "[{:+.4f}, {:+.4f}]".format(
        *dlen
    )
)


common = (
    (sa == 1)
    &
    (sb == 1)
)

print(
    "Common-success N:",
    common.sum(),
)

cs_spl = bootstrap(
    spl_a[common],
    spl_b[common],
    100.0,
)

cs_len = bootstrap(
    len_a[common],
    len_b[common],
)


print(
    "Common-success "
    "Delta SPL (pp): "
    "{:+.4f} "
    "95% CI "
    "[{:+.4f}, {:+.4f}]".format(
        *cs_spl
    )
)

print(
    "Common-success "
    "Delta Len: "
    "{:+.4f} "
    "95% CI "
    "[{:+.4f}, {:+.4f}]".format(
        *cs_len
    )
)

print()
print(
    "BASE VS FINAL LCR: PASS"
)
