import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]


PAIRS = [
    (
        "VAL Base vs Learned",
        Path("results/base500_zsval"),
        Path("results/learned_guard_val_t04"),
    ),
    (
        "VAL Rule vs Learned",
        Path("results/guard_tau02_zsval"),
        Path("results/learned_guard_val_t04"),
    ),
    (
        "TEST Base vs Learned",
        Path("results/final_test_base500"),
        Path("results/learned_guard_test_t04"),
    ),
    (
        "TEST Rule vs Learned",
        Path("results/guard_tau02_zstest"),
        Path("results/learned_guard_test_t04"),
    ),
]


def load(root):
    rows = {}

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        if not p.exists():
            return None

        with open(
            p,
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                x = json.loads(line)

                key = (
                    x.get("scene_type", room),
                    int(x["eval_index"]),
                )

                rows[key] = x

    return rows


def exact_mcnemar(n01, n10):
    n = n01 + n10

    if n == 0:
        return 1.0

    k = min(n01, n10)

    p = 2.0 * sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(1.0, p)


def bootstrap_ci(
    values,
    seed=0,
    B=20000,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)

    n = len(x)

    means = np.empty(
        B,
        dtype=np.float64,
    )

    for i in range(B):
        ids = rng.integers(
            0,
            n,
            size=n,
        )

        means[i] = x[ids].mean()

    return np.quantile(
        means,
        [0.025, 0.975],
    )


def analyze(
    name,
    root_a,
    root_b,
):
    A = load(root_a)
    B = load(root_b)

    if A is None or B is None:
        print()
        print(name)
        print("SKIP: missing episode files")
        print("A:", root_a)
        print("B:", root_b)
        return None

    assert set(A) == set(B)

    n00 = n01 = n10 = n11 = 0

    dspl = []
    dlen = []

    common_spl = []
    common_len = []

    targets = defaultdict(Counter)
    rooms = defaultdict(Counter)

    for key in sorted(A):
        a = A[key]
        b = B[key]

        assert a["scene"] == b["scene"]
        assert a["target"] == b["target"]

        sa = bool(a["success"])
        sb = bool(b["success"])

        if not sa and not sb:
            o = "N00"
            n00 += 1

        elif not sa and sb:
            o = "N01"
            n01 += 1

        elif sa and not sb:
            o = "N10"
            n10 += 1

        else:
            o = "N11"
            n11 += 1

        targets[a["target"]][o] += 1
        rooms[key[0]][o] += 1

        dspl.append(
            float(b["spl"])
            - float(a["spl"])
        )

        dlen.append(
            float(b["ep_length"])
            - float(a["ep_length"])
        )

        if sa and sb:
            common_spl.append(
                float(b["spl"])
                - float(a["spl"])
            )

            common_len.append(
                float(b["ep_length"])
                - float(a["ep_length"])
            )

    n = len(A)

    sr_a = (
        n10 + n11
    ) / n

    sr_b = (
        n01 + n11
    ) / n

    spl_ci = bootstrap_ci(dspl)
    len_ci = bootstrap_ci(dlen)

    cspl_ci = bootstrap_ci(
        common_spl
    )

    clen_ci = bootstrap_ci(
        common_len
    )

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    print("episodes:", n)

    print(
        "N00/N01/N10/N11:",
        n00,
        n01,
        n10,
        n11,
    )

    print(
        "SR A / B:",
        f"{100*sr_a:.4f}",
        f"{100*sr_b:.4f}",
    )

    print(
        "Delta SR:",
        f"{100*(sr_b-sr_a):+.4f} pp",
    )

    print(
        "McNemar p:",
        f"{exact_mcnemar(n01,n10):.8f}",
    )

    dspl = np.asarray(dspl)
    dlen = np.asarray(dlen)

    print(
        "Delta SPL:",
        f"{100*dspl.mean():+.4f} pp",
    )

    print(
        "SPL CI:",
        "[{:+.4f}, {:+.4f}] pp".format(
            100*spl_ci[0],
            100*spl_ci[1],
        ),
    )

    print(
        "Delta Len:",
        f"{dlen.mean():+.4f}",
    )

    print(
        "Len CI:",
        "[{:+.4f}, {:+.4f}]".format(
            len_ci[0],
            len_ci[1],
        ),
    )

    print(
        "Common success:",
        len(common_spl),
    )

    print(
        "Common Delta SPL:",
        f"{100*np.mean(common_spl):+.4f} pp",
    )

    print(
        "Common SPL CI:",
        "[{:+.4f}, {:+.4f}] pp".format(
            100*cspl_ci[0],
            100*cspl_ci[1],
        ),
    )

    print(
        "Common Delta Len:",
        f"{np.mean(common_len):+.4f}",
    )

    print(
        "Common Len CI:",
        "[{:+.4f}, {:+.4f}]".format(
            clen_ci[0],
            clen_ci[1],
        ),
    )

    print()
    print("TARGET REVERSALS")
    print(
        f"{'Target':<18}"
        f"{'B-only':>9}"
        f"{'A-only':>9}"
        f"{'Net':>8}"
    )

    for target in sorted(targets):
        c = targets[target]

        print(
            f"{target:<18}"
            f"{c['N01']:>9}"
            f"{c['N10']:>9}"
            f"{c['N01']-c['N10']:>8}"
        )

    print()
    print("ROOM REVERSALS")
    print(
        f"{'Room':<18}"
        f"{'B-only':>9}"
        f"{'A-only':>9}"
        f"{'Net':>8}"
    )

    for room in sorted(rooms):
        c = rooms[room]

        print(
            f"{room:<18}"
            f"{c['N01']:>9}"
            f"{c['N10']:>9}"
            f"{c['N01']-c['N10']:>8}"
        )

    return {
        "name": name,
        "episodes": n,
        "N00": n00,
        "N01": n01,
        "N10": n10,
        "N11": n11,
        "sr_a": sr_a,
        "sr_b": sr_b,
        "delta_sr_pp":
            100 * (sr_b - sr_a),
        "mcnemar_p":
            exact_mcnemar(
                n01,
                n10,
            ),
        "delta_spl_pp":
            100 * dspl.mean(),
        "spl_ci_pp": [
            100 * spl_ci[0],
            100 * spl_ci[1],
        ],
        "delta_len":
            dlen.mean(),
        "len_ci": [
            len_ci[0],
            len_ci[1],
        ],
    }


results = []

for item in PAIRS:
    r = analyze(*item)

    if r is not None:
        results.append(r)


out = Path(
    "results/learned_guard_protocol/"
    "final_offline_summary.json"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    out,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        indent=2,
    )

print()
print("=" * 78)
print("FINAL OFFLINE ANALYSIS COMPLETE")
print("=" * 78)
print("saved:", out)
