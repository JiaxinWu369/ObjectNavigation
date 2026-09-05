import json
from pathlib import Path
from collections import Counter

from scipy.stats import binomtest


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

RUNS = {
    "K0_Base":
        Path("results/base500_zsval"),

    "K10":
        Path("results/early10_500_zsval"),

    "K20":
        Path("results/early20_500_zsval"),

    "K30":
        Path("results/early30_500_zsval"),

    "K40":
        Path("results/early40_500_zsval"),

    "Kinf":
        Path("results/nocontext500_zsval"),
}


def load(root):

    out = {}

    for room in ROOMS:

        p = root / f"episodes_{room}.jsonl"

        if not p.exists():
            raise FileNotFoundError(p)

        with open(
            p,
            encoding="utf-8"
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                r = json.loads(line)

                key = (
                    room,
                    int(r["eval_index"]),
                )

                if key in out:
                    raise RuntimeError(
                        f"duplicate {key}"
                    )

                out[key] = r

    assert len(out) == 1260

    return out


data = {
    k: load(v)
    for k, v in RUNS.items()
}


keys = set(
    data["K0_Base"]
)

for name in data:
    assert set(data[name]) == keys


# ----------------------------------------------------------
# Each tuple:
#
# boundary step,
# Restore NOW,
# Delay restoration
#
# At K=40, Delay = never restore.
# ----------------------------------------------------------

PAIRS = [
    (0,  "K0_Base", "K10"),
    (10, "K10",     "K20"),
    (20, "K20",     "K30"),
    (30, "K30",     "K40"),
    (40, "K40",     "Kinf"),
]


for K, restore_name, delay_name in PAIRS:

    restore = data[restore_name]
    delay = data[delay_name]

    c = Counter()

    prefix_mismatch = 0
    state_mismatch = 0
    early_end_mismatch = 0
    reached = 0

    target_counts = {}

    for key in sorted(keys):

        r = restore[key]
        d = delay[key]

        assert r["scene"] == d["scene"]
        assert r["target"] == d["target"]

        yr = int(bool(r["success"]))
        yd = int(bool(d["success"]))

        # R1D0 = restore-now only
        # R0D1 = delay only
        c[(yr, yd)] += 1

        target = r["target"]

        if target not in target_counts:
            target_counts[target] = Counter()

        target_counts[target][(yr, yd)] += 1

        ra = r["full_actions"]
        da = d["full_actions"]

        rs = r["full_states"]
        ds = d["full_states"]

        # K=0: same initial state is sufficient.
        if K == 0:

            assert len(rs) >= 1
            assert len(ds) >= 1

            if rs[0] != ds[0]:
                state_mismatch += 1

            reached += 1
            continue

        # If one ends before intervention,
        # both histories/outcomes must be identical.
        if len(ra) <= K or len(da) <= K:

            if (
                ra != da
                or rs != ds
                or yr != yd
            ):
                early_end_mismatch += 1

            continue

        reached += 1

        if ra[:K] != da[:K]:
            prefix_mismatch += 1

        if rs[K] != ds[K]:
            state_mismatch += 1


    restore_only = c[(1, 0)]
    delay_only = c[(0, 1)]

    both_success = c[(1, 1)]
    both_fail = c[(0, 0)]

    discordant = (
        restore_only
        +
        delay_only
    )

    if discordant:

        p = binomtest(
            restore_only,
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    else:
        p = 1.0

    sr_restore = (
        restore_only
        +
        both_success
    ) / 1260.0

    sr_delay = (
        delay_only
        +
        both_success
    ) / 1260.0

    union = (
        restore_only
        +
        delay_only
        +
        both_success
    ) / 1260.0

    headroom = (
        union
        -
        max(
            sr_restore,
            sr_delay
        )
    )

    print()
    print("=" * 94)

    print(
        f"K={K}: "
        f"RestoreNow={restore_name} "
        f"vs Delay={delay_name}"
    )

    print("=" * 94)

    print(
        "Both fail          :",
        both_fail
    )

    print(
        "Restore-now only   :",
        restore_only
    )

    print(
        "Delay only         :",
        delay_only
    )

    print(
        "Both success       :",
        both_success
    )

    print(
        "Discordant         :",
        discordant
    )

    print(
        "Restore-now SR     :",
        f"{100*sr_restore:.3f}%"
    )

    print(
        "Delay SR           :",
        f"{100*sr_delay:.3f}%"
    )

    print(
        "Restore - Delay    :",
        f"{100*(sr_restore-sr_delay):+.3f} pp"
    )

    print(
        "Exact paired p     :",
        f"{p:.6g}"
    )

    print(
        "Oracle union SR    :",
        f"{100*union:.3f}%"
    )

    print(
        "Adaptive headroom  :",
        f"{100*headroom:.3f} pp"
    )

    print(
        "Reached boundary   :",
        reached
    )

    print(
        "Prefix mismatch    :",
        prefix_mismatch
    )

    print(
        "Boundary state diff:",
        state_mismatch
    )

    print(
        "Early-end mismatch :",
        early_end_mismatch
    )

    assert prefix_mismatch == 0
    assert state_mismatch == 0
    assert early_end_mismatch == 0

    print()
    print(
        f"{'Target':<16}"
        f"{'Restore':>10}"
        f"{'Delay':>10}"
        f"{'Net':>10}"
    )

    print("-" * 48)

    for target in sorted(
        target_counts
    ):

        tc = target_counts[target]

        ro = tc[(1, 0)]
        do = tc[(0, 1)]

        if ro + do == 0:
            continue

        print(
            f"{target:<16}"
            f"{ro:>10d}"
            f"{do:>10d}"
            f"{ro-do:>10d}"
        )


print()
print(
    "500K ACTIVATION LADDER MATCHING: PASS"
)
