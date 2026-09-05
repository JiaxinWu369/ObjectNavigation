import json
import math
from pathlib import Path

import numpy as np


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

LEARNED = Path(
    "results/learned_guard_test_t04"
)

TARGETS = {
    "Base": {
        "success": 0.6535369775,
        "spl": 0.2810600510,
        "ep_length": 42.51527331,
    },
    "RuleGuard02": {
        "success": 0.6615755627,
        "spl": 0.2813296407,
        "ep_length": 41.99678457,
    },
}


def count_episodes(root):
    total = 0

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        if not p.exists():
            return -1

        with open(
            p,
            encoding="utf-8",
        ) as f:
            total += sum(
                1
                for line in f
                if line.strip()
            )

    return total


def find_runs(target):

    hits = []

    for p in Path("results").rglob(
        "metrics.json"
    ):

        try:
            x = json.load(open(p))
        except Exception:
            continue

        if (
            abs(
                float(x.get("success", -999))
                - target["success"]
            ) < 2e-6
            and
            abs(
                float(x.get("spl", -999))
                - target["spl"]
            ) < 2e-6
            and
            abs(
                float(x.get("ep_length", -999))
                - target["ep_length"]
            ) < 2e-4
        ):
            root = p.parent

            if count_episodes(root) == 1244:
                hits.append(root)

    return sorted(
        set(hits),
        key=lambda x: str(x),
    )


def load(root):

    rows = {}

    for room in ROOMS:

        p = root / f"episodes_{room}.jsonl"

        with open(
            p,
            encoding="utf-8",
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                x = json.loads(line)

                key = (
                    x.get(
                        "scene_type",
                        room,
                    ),
                    int(
                        x["eval_index"]
                    ),
                )

                if key in rows:
                    raise RuntimeError(
                        f"duplicate key {key}"
                    )

                rows[key] = x

    return rows


def exact_mcnemar(n01, n10):

    n = n01 + n10

    if n == 0:
        return 1.0

    k = min(n01, n10)

    tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(
        1.0,
        2.0 * tail,
    )


def ci_bootstrap(d, seed=0, B=20000):

    d = np.asarray(
        d,
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)

    n = len(d)

    vals = np.empty(
        B,
        dtype=np.float64,
    )

    for i in range(B):

        ids = rng.integers(
            0,
            n,
            size=n,
        )

        vals[i] = d[ids].mean()

    return np.quantile(
        vals,
        [0.025, 0.975],
    )


def compare(name_a, root_a, name_b, root_b):

    A = load(root_a)
    B = load(root_b)

    assert set(A) == set(B)

    keys = sorted(A)

    n00 = n01 = n10 = n11 = 0

    dspl = []
    dlen = []

    common_spl = []
    common_len = []

    for key in keys:

        a = A[key]
        b = B[key]

        assert (
            a["scene"]
            == b["scene"]
        )

        assert (
            a["target"]
            == b["target"]
        )

        sa = bool(a["success"])
        sb = bool(b["success"])

        if not sa and not sb:
            n00 += 1

        elif not sa and sb:
            n01 += 1

        elif sa and not sb:
            n10 += 1

        else:
            n11 += 1

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

    dspl = np.asarray(dspl)
    dlen = np.asarray(dlen)

    spl_ci = ci_bootstrap(dspl)
    len_ci = ci_bootstrap(dlen)

    common_spl = np.asarray(
        common_spl,
        dtype=np.float64,
    )

    common_len = np.asarray(
        common_len,
        dtype=np.float64,
    )

    cspl_ci = ci_bootstrap(
        common_spl
    )

    clen_ci = ci_bootstrap(
        common_len
    )

    sr_a = (
        n10 + n11
    ) / len(keys)

    sr_b = (
        n01 + n11
    ) / len(keys)

    print()
    print("=" * 72)
    print(
        f"{name_a}  vs  {name_b}"
    )
    print("=" * 72)

    print("A:", root_a)
    print("B:", root_b)

    print()
    print("N00 both fail :", n00)
    print(
        f"N01 {name_b}-only :",
        n01,
    )
    print(
        f"N10 {name_a}-only :",
        n10,
    )
    print("N11 both success:", n11)

    print()
    print(
        f"{name_a} SR:",
        f"{100*sr_a:.4f}%",
    )

    print(
        f"{name_b} SR:",
        f"{100*sr_b:.4f}%",
    )

    print(
        "Delta SR B-A:",
        f"{100*(sr_b-sr_a):+.4f} pp",
    )

    print(
        "McNemar exact p:",
        exact_mcnemar(
            n01,
            n10,
        ),
    )

    print()
    print(
        "Delta SPL B-A:",
        f"{100*dspl.mean():+.4f} pp",
    )

    print(
        "95% CI SPL:",
        "[{:+.4f}, {:+.4f}] pp".format(
            100*spl_ci[0],
            100*spl_ci[1],
        ),
    )

    print(
        "Delta Len B-A:",
        f"{dlen.mean():+.4f}",
    )

    print(
        "95% CI Len:",
        "[{:+.4f}, {:+.4f}]".format(
            len_ci[0],
            len_ci[1],
        ),
    )

    print()
    print(
        "common success N:",
        len(common_spl),
    )

    print(
        "common Delta SPL:",
        f"{100*common_spl.mean():+.4f} pp",
    )

    print(
        "common SPL 95% CI:",
        "[{:+.4f}, {:+.4f}] pp".format(
            100*cspl_ci[0],
            100*cspl_ci[1],
        ),
    )

    print(
        "common Delta Len:",
        f"{common_len.mean():+.4f}",
    )

    print(
        "common Len 95% CI:",
        "[{:+.4f}, {:+.4f}]".format(
            clen_ci[0],
            clen_ci[1],
        ),
    )


assert count_episodes(
    LEARNED
) == 1244

print("=" * 72)
print("LOCATING TEST RUNS")
print("=" * 72)

found = {}

for name, target in TARGETS.items():

    hits = find_runs(target)

    print()
    print(name)

    for x in hits:
        print("  ", x)

    if not hits:
        raise RuntimeError(
            f"No complete Test run found for {name}"
        )

    # If duplicates have identical aggregate metrics,
    # compare each candidate; the first is used below.
    found[name] = hits[0]


compare(
    "Base",
    found["Base"],
    "Learned-0.4",
    LEARNED,
)

compare(
    "RuleGuard02",
    found["RuleGuard02"],
    "Learned-0.4",
    LEARNED,
)
