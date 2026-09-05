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

RUNS = {
    "CG": Path(
        "results/lcr_constant_global500_zsval"
    ),
    "LG": Path(
        "results/lcr_learned_global500_zsval"
    ),
    "CR": Path(
        "results/lcr_constant500_zsval"
    ),
    "LR": Path(
        "results/lcr_learned500_zsval"
    ),
}

N_BOOT = 20000
SEED = 20260828


def canonical_state(s):
    return (
        round(float(s["x"]), 4),
        round(float(s["z"]), 4),
        int(round(float(
            s["rotation"]["y"]
        ))),
        int(round(float(
            s["horizon"]
        ))),
    )


def load_run(root):
    rows = {}

    for room in ROOMS:
        p = (
            root
            / f"episodes_{room}.jsonl"
        )

        if not p.exists():
            raise FileNotFoundError(p)

        with open(
            p,
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
                        f"duplicate {key}"
                    )

                rows[key] = r

    return rows


def validate(runs):

    names = list(runs)

    ref = runs[names[0]]

    ref_keys = set(ref)

    for name in names[1:]:
        cur = runs[name]

        if set(cur) != ref_keys:
            raise RuntimeError(
                f"episode keys differ: {name}"
            )

    for key in sorted(ref_keys):

        first = ref[key]

        for name in names[1:]:

            cur = runs[name][key]

            for field in [
                "scene",
                "scene_type",
                "target",
            ]:
                if (
                    cur[field]
                    != first[field]
                ):
                    raise RuntimeError(
                        f"{key}: "
                        f"{field} mismatch"
                    )

            if (
                canonical_state(
                    cur["start_state"]
                )
                !=
                canonical_state(
                    first["start_state"]
                )
            ):
                raise RuntimeError(
                    f"{key}: "
                    "start-state mismatch"
                )

    print(
        "PAIR ALIGNMENT: PASS",
        "N=",
        len(ref_keys),
    )


def arrays(rows, keys):

    success = np.asarray(
        [
            int(bool(
                rows[k]["success"]
            ))
            for k in keys
        ],
        dtype=np.float64,
    )

    spl = np.asarray(
        [
            float(
                rows[k]["spl"]
            )
            for k in keys
        ],
        dtype=np.float64,
    )

    length = np.asarray(
        [
            float(
                rows[k]["ep_length"]
            )
            for k in keys
        ],
        dtype=np.float64,
    )

    return {
        "success": success,
        "spl": spl,
        "length": length,
    }


def ci_mean_difference(
    a,
    b,
    rng,
):
    """
    Return B - A paired bootstrap CI.
    """
    d = b - a
    n = len(d)

    vals = np.empty(
        N_BOOT,
        dtype=np.float64,
    )

    for i in range(N_BOOT):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        vals[i] = (
            d[idx].mean()
        )

    return (
        float(d.mean()),
        float(np.percentile(
            vals,
            2.5,
        )),
        float(np.percentile(
            vals,
            97.5,
        )),
    )


def compare(
    name_a,
    a,
    name_b,
    b,
    rng,
):

    print()
    print("=" * 88)
    print(
        name_a,
        "->",
        name_b,
    )
    print("=" * 88)

    sa = a["success"].astype(int)
    sb = b["success"].astype(int)

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

    discord = (
        n01 + n10
    )

    if discord:
        p = binomtest(
            min(n01, n10),
            discord,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    else:
        p = 1.0

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

    dsr = (
        100.0
        *
        (
            b["success"].mean()
            -
            a["success"].mean()
        )
    )

    print(
        "Delta SR (pp):",
        f"{dsr:+.4f}",
    )

    dspl = ci_mean_difference(
        100.0 * a["spl"],
        100.0 * b["spl"],
        rng,
    )

    dlen = ci_mean_difference(
        a["length"],
        b["length"],
        rng,
    )

    print(
        "Delta SPL (pp): "
        f"{dspl[0]:+.4f} "
        f"95% CI "
        f"[{dspl[1]:+.4f}, "
        f"{dspl[2]:+.4f}]"
    )

    print(
        "Delta Len: "
        f"{dlen[0]:+.4f} "
        f"95% CI "
        f"[{dlen[1]:+.4f}, "
        f"{dlen[2]:+.4f}]"
    )

    common = (
        (sa == 1)
        &
        (sb == 1)
    )

    print(
        "Common-success N:",
        int(common.sum()),
    )

    if common.any():

        cs_spl = ci_mean_difference(
            100.0
            * a["spl"][common],
            100.0
            * b["spl"][common],
            rng,
        )

        cs_len = ci_mean_difference(
            a["length"][common],
            b["length"][common],
            rng,
        )

        print(
            "Common-success "
            "Delta SPL (pp): "
            f"{cs_spl[0]:+.4f} "
            f"95% CI "
            f"[{cs_spl[1]:+.4f}, "
            f"{cs_spl[2]:+.4f}]"
        )

        print(
            "Common-success "
            "Delta Len: "
            f"{cs_len[0]:+.4f} "
            f"95% CI "
            f"[{cs_len[1]:+.4f}, "
            f"{cs_len[2]:+.4f}]"
        )


def factorial_bootstrap(
    arr,
    metric,
    scale,
    rng,
):

    cg = arr["CG"][metric]
    lg = arr["LG"][metric]
    cr = arr["CR"][metric]
    lr = arr["LR"][metric]

    n = len(cg)

    # Learned - Constant,
    # averaged over calibration style.
    reliability = 0.5 * (
        (lg - cg)
        +
        (lr - cr)
    )

    # Relation - Global,
    # averaged over reliability type.
    calibration = 0.5 * (
        (cr - cg)
        +
        (lr - lg)
    )

    # Difference-in-differences.
    interaction = (
        (lr - lg)
        -
        (cr - cg)
    )

    effects = {
        "Learned-Constant":
            reliability,
        "Relation-Global":
            calibration,
        "Interaction":
            interaction,
    }

    print()
    print(
        "-" * 88
    )
    print(
        "FACTORIAL EFFECT:",
        metric,
    )
    print(
        "-" * 88
    )

    for name, d in effects.items():

        boots = np.empty(
            N_BOOT,
            dtype=np.float64,
        )

        for i in range(N_BOOT):

            idx = rng.integers(
                0,
                n,
                size=n,
            )

            boots[i] = (
                d[idx].mean()
            )

        mean = (
            scale
            * d.mean()
        )

        lo = (
            scale
            * np.percentile(
                boots,
                2.5,
            )
        )

        hi = (
            scale
            * np.percentile(
                boots,
                97.5,
            )
        )

        print(
            f"{name:20s}: "
            f"{mean:+.4f} "
            f"[{lo:+.4f}, "
            f"{hi:+.4f}]"
        )


runs = {
    name: load_run(path)
    for name, path
    in RUNS.items()
}

validate(runs)

keys = sorted(
    next(
        iter(runs.values())
    ).keys()
)

arr = {
    name: arrays(rows, keys)
    for name, rows
    in runs.items()
}

print()
print("=" * 88)
print("METHOD SUMMARY")
print("=" * 88)

for name in [
    "CG",
    "LG",
    "CR",
    "LR",
]:

    x = arr[name]

    print(
        f"{name:3s}",
        "SR={:.4f}%".format(
            100
            * x["success"].mean()
        ),
        "SPL={:.4f}%".format(
            100
            * x["spl"].mean()
        ),
        "Len={:.4f}".format(
            x["length"].mean()
        ),
    )


rng = np.random.default_rng(
    SEED
)

# Reliability effect under
# relation-specific calibration.
compare(
    "CR",
    arr["CR"],
    "LR",
    arr["LR"],
    rng,
)

# Reliability effect under
# global calibration.
compare(
    "CG",
    arr["CG"],
    "LG",
    arr["LG"],
    rng,
)

# Calibration effect with
# constant reliability.
compare(
    "CG",
    arr["CG"],
    "CR",
    arr["CR"],
    rng,
)

# Calibration effect with
# learned reliability.
compare(
    "LG",
    arr["LG"],
    "LR",
    arr["LR"],
    rng,
)

factorial_bootstrap(
    arr,
    "success",
    100.0,
    rng,
)

factorial_bootstrap(
    arr,
    "spl",
    100.0,
    rng,
)

factorial_bootstrap(
    arr,
    "length",
    1.0,
    rng,
)

print()
print(
    "2x2 LCR FACTORIAL ANALYSIS: PASS"
)
