import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


DATA = Path(
    "results/k10_dynamic_dataset_100k/"
    "reversal_79.jsonl"
)

TARGETS = [
    "Laptop",
    "LightSwitch",
]

N_PERM = 10000
N_BOOT = 10000
SEED = 20260819


def auc_score(y, x):
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)

    pos = y == 1
    n1 = int(pos.sum())
    n0 = int((~pos).sum())

    ranks = rankdata(
        x,
        method="average"
    )

    u = (
        ranks[pos].sum()
        -
        n1 * (n1 + 1) / 2
    )

    return float(
        u / (n1 * n0)
    )


def bootstrap_ci(y, x, seed):
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)

    xp = x[y == 1]
    xn = x[y == 0]

    rng = np.random.default_rng(seed)

    vals = []

    for _ in range(N_BOOT):
        bp = rng.choice(
            xp,
            len(xp),
            replace=True
        )

        bn = rng.choice(
            xn,
            len(xn),
            replace=True
        )

        yy = np.concatenate([
            np.ones(len(bp), dtype=int),
            np.zeros(len(bn), dtype=int),
        ])

        xx = np.concatenate([
            bp,
            bn
        ])

        vals.append(
            auc_score(yy, xx)
        )

    return np.quantile(
        vals,
        [0.025, 0.975]
    )


def permutation_p(y, x, seed):
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)

    obs = auc_score(y, x)
    effect = abs(obs - 0.5)

    rng = np.random.default_rng(seed)

    extreme = 0

    for _ in range(N_PERM):
        yp = rng.permutation(y)

        a = auc_score(
            yp,
            x
        )

        if abs(a - 0.5) >= effect - 1e-12:
            extreme += 1

    return (
        extreme + 1
    ) / (
        N_PERM + 1
    )


rows = []

with open(
    DATA,
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        r = json.loads(line)

        trace = r["trace"]

        assert len(trace) >= 11

        kl10 = float(
            trace[10]["semantic_kl"]
        )

        kl_recent5 = float(
            np.mean([
                float(
                    trace[t]["semantic_kl"]
                )
                for t in range(6, 11)
            ])
        )

        rows.append({
            "target":
                r["target"],

            "label":
                int(r["restore_label"]),

            "kl_t10":
                kl10,

            "kl_mean_recent5":
                kl_recent5,
        })


for target in TARGETS:

    group = [
        r
        for r in rows
        if r["target"] == target
    ]

    y = np.asarray([
        r["label"]
        for r in group
    ])

    print()
    print("=" * 80)
    print(target)
    print("=" * 80)

    print(
        "N={} help={} harm={}".format(
            len(group),
            int(y.sum()),
            int((y == 0).sum()),
        )
    )

    for i, feature in enumerate([
        "kl_t10",
        "kl_mean_recent5",
    ]):

        x = np.asarray([
            r[feature]
            for r in group
        ])

        auc = auc_score(
            y,
            x
        )

        lo, hi = bootstrap_ci(
            y,
            x,
            SEED + i
        )

        p = permutation_p(
            y,
            x,
            SEED + 100 + i
        )

        print(
            "{:<20} "
            "help={:.6f} "
            "harm={:.6f} "
            "AUC={:.3f} "
            "CI=[{:.3f},{:.3f}] "
            "p={:.4f}".format(
                feature,
                x[y == 1].mean(),
                x[y == 0].mean(),
                auc,
                lo,
                hi,
                p,
            )
        )

print()
print(
    "K10 KL SUPPLEMENT: PASS"
)
