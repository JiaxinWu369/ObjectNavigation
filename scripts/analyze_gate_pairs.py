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


def load_run(root):
    root = Path(root)
    records = {}

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        if not p.exists():
            raise FileNotFoundError(p)

        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)

                # Deterministic pairing key.
                key = (
                    r["scene_type"],
                    int(r["eval_index"]),
                )

                if key in records:
                    raise RuntimeError(
                        f"Duplicate key: {key}"
                    )

                records[key] = r

    return records


def bootstrap_ci(values, n_boot=10000, seed=1234):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)

    means = np.empty(
        n_boot,
        dtype=np.float64,
    )

    n = len(values)

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        means[i] = values[idx].mean()

    return np.quantile(
        means,
        [0.025, 0.975],
    )


def summarize(name, data):
    rows = list(data.values())

    sr = np.mean([
        bool(r["success"])
        for r in rows
    ])

    spl = np.mean([
        float(r["spl"])
        for r in rows
    ])

    ep_len = np.mean([
        float(r["ep_length"])
        for r in rows
    ])

    dts = np.mean([
        float(r["dts"])
        for r in rows
    ])

    print(
        f"{name:<12}"
        f"N={len(rows):4d}  "
        f"SR={100*sr:6.2f}%  "
        f"SPL={100*spl:6.2f}%  "
        f"LEN={ep_len:7.2f}  "
        f"DTS={dts:7.3f}"
    )


def compare(name, base, other):
    keys = sorted(base.keys())

    if set(keys) != set(other.keys()):
        raise RuntimeError(
            f"{name}: pairing keys do not match"
        )

    b_success = np.array([
        bool(base[k]["success"])
        for k in keys
    ])

    o_success = np.array([
        bool(other[k]["success"])
        for k in keys
    ])

    b_spl = np.array([
        float(base[k]["spl"])
        for k in keys
    ])

    o_spl = np.array([
        float(other[k]["spl"])
        for k in keys
    ])

    n00 = int(np.sum(
        (~b_success) &
        (~o_success)
    ))

    n01 = int(np.sum(
        (~b_success) &
        o_success
    ))

    n10 = int(np.sum(
        b_success &
        (~o_success)
    ))

    n11 = int(np.sum(
        b_success &
        o_success
    ))

    discordant = n01 + n10

    if discordant > 0:
        p_value = binomtest(
            n01,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    else:
        p_value = 1.0

    sr_delta = (
        o_success.astype(float) -
        b_success.astype(float)
    )

    spl_delta = o_spl - b_spl

    sr_ci = bootstrap_ci(
        sr_delta * 100.0
    )

    spl_ci = bootstrap_ci(
        spl_delta * 100.0
    )

    action_changed = sum(
        base[k].get("actions") !=
        other[k].get("actions")
        for k in keys
    )

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("N00 Base fail -> Other fail :", n00)
    print("N01 Base fail -> Other succ :", n01)
    print("N10 Base succ -> Other fail :", n10)
    print("N11 Base succ -> Other succ :", n11)

    print()
    print(
        "Net recovered successes     :",
        n01 - n10
    )

    print(
        "McNemar exact p             :",
        f"{p_value:.6g}"
    )

    print(
        "SR delta                    :",
        f"{100*sr_delta.mean():+.3f} pp"
    )

    print(
        "SR paired bootstrap 95% CI  :",
        f"[{sr_ci[0]:+.3f}, "
        f"{sr_ci[1]:+.3f}] pp"
    )

    print(
        "SPL delta                   :",
        f"{100*spl_delta.mean():+.3f} pp"
    )

    print(
        "SPL paired bootstrap 95% CI :",
        f"[{spl_ci[0]:+.3f}, "
        f"{spl_ci[1]:+.3f}] pp"
    )

    print(
        "Different action trajectories:",
        action_changed,
        "/",
        len(keys),
    )


def target_table(base, strong, random_run):
    targets = sorted({
        r["target"]
        for r in base.values()
    })

    print()
    print("=" * 96)
    print("PER-TARGET SR")
    print("=" * 96)

    print(
        f"{'Target':<16}"
        f"{'N':>7}"
        f"{'Base':>11}"
        f"{'Strong':>11}"
        f"{'Random':>11}"
        f"{'S-B':>11}"
        f"{'R-B':>11}"
    )

    print("-" * 78)

    keys = sorted(base.keys())

    for target in targets:
        kk = [
            k for k in keys
            if base[k]["target"] == target
        ]

        def sr(data):
            return np.mean([
                bool(data[k]["success"])
                for k in kk
            ])

        b = sr(base)
        s = sr(strong)
        r = sr(random_run)

        print(
            f"{target:<16}"
            f"{len(kk):>7d}"
            f"{100*b:>10.2f}%"
            f"{100*s:>10.2f}%"
            f"{100*r:>10.2f}%"
            f"{100*(s-b):>+10.2f}"
            f"{100*(r-b):>+10.2f}"
        )


def main():
    base = load_run(
        "results/pair_base50"
    )

    strong = load_run(
        "results/pair_strong01"
    )

    random_run = load_run(
        "results/pair_random01"
    )

    assert len(base) == 1260
    assert len(strong) == 1260
    assert len(random_run) == 1260

    print("=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)

    summarize(
        "Base",
        base,
    )

    summarize(
        "Strong-0.1",
        strong,
    )

    summarize(
        "Random-0.1",
        random_run,
    )

    compare(
        "BASE vs STRONG-0.1",
        base,
        strong,
    )

    compare(
        "BASE vs RANDOM-0.1",
        base,
        random_run,
    )

    compare(
        "RANDOM-0.1 vs STRONG-0.1",
        random_run,
        strong,
    )

    target_table(
        base,
        strong,
        random_run,
    )


if __name__ == "__main__":
    main()
