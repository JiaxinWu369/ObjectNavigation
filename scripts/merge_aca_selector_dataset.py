import json
from pathlib import Path
from collections import Counter, defaultdict

from scipy.stats import binomtest


B1 = Path(
    "results/aca_train500/paired/all_4000.jsonl"
)

B2 = Path(
    "results/aca_train500/batch2/paired/all_4431.jsonl"
)

OUT = Path(
    "results/aca_train500/selector"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def load_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


b1_all = load_jsonl(B1)

# Only episodes that actually reached the
# intervention state are selector samples.
b1 = [
    r
    for r in b1_all
    if r["reached_k10"]
]

b2 = load_jsonl(B2)


for r in b1:
    r["batch"] = 1

for r in b2:
    r["batch"] = 2


assert len(b1) == 2224
assert len(b2) == 4431


rows = b1 + b2

assert len(rows) == 6655


# --------------------------------------------------
# No duplicated start state across the two batches.
# --------------------------------------------------

keys = [
    (
        r["scene"],
        str(r["start_state"]),
    )
    for r in rows
]

assert len(keys) == len(set(keys))


# --------------------------------------------------
# Every selector sample needs its common-history
# Early20 trace.
# --------------------------------------------------

trace_missing = sum(
    r.get("context_trace") is None
    for r in rows
)

assert trace_missing == 0


counts = Counter(
    r["pair_state"]
    for r in rows
)

room_counts = defaultdict(Counter)
target_counts = defaultdict(Counter)

for r in rows:
    room_counts[
        r["room"]
    ][
        r["pair_state"]
    ] += 1

    target_counts[
        r["target"]
    ][
        r["pair_state"]
    ] += 1


n00 = counts["N00"]
n01 = counts["N01"]
n10 = counts["N10"]
n11 = counts["N11"]

n = len(rows)

nd = n01 + n10

sr10 = (
    n10 + n11
) / n

sr20 = (
    n01 + n11
) / n

oracle = (
    n01 + n10 + n11
) / n

headroom = (
    oracle
    - max(sr10, sr20)
)

p_exact = binomtest(
    n10,
    nd,
    p=0.5,
    alternative="two-sided",
).pvalue


# --------------------------------------------------
# Preference labels.
#
# 1 = RestoreNow / Early10 helpful
# 0 = Delay / Early20 helpful
# --------------------------------------------------

preference = []

for r in rows:

    if r["pair_state"] not in {
        "N10",
        "N01",
    }:
        continue

    x = dict(r)

    x["restore_now_label"] = int(
        r["pair_state"] == "N10"
    )

    preference.append(x)


assert len(preference) == nd


all_path = OUT / "eligible_all_6655.jsonl"

with open(
    all_path,
    "w",
    encoding="utf-8",
) as f:
    for r in rows:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


pref_path = OUT / "discordant_all_240.jsonl"

with open(
    pref_path,
    "w",
    encoding="utf-8",
) as f:
    for r in preference:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


summary = {
    "eligible": n,

    "batch1_eligible": len(b1),
    "batch2_eligible": len(b2),

    "N00": n00,
    "N01_delay_only": n01,
    "N10_restore_only": n10,
    "N11": n11,

    "discordant": nd,
    "discordant_rate": nd / n,

    "early10_sr": sr10,
    "early20_sr": sr20,

    "early10_minus_early20_pp":
        100 * (sr10 - sr20),

    "paired_exact_p":
        p_exact,

    "oracle_sr":
        oracle,

    "adaptive_headroom_pp":
        100 * headroom,

    "trace_missing":
        trace_missing,
}


with open(
    OUT / "summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print("=" * 84)
print("ACA SELECTOR MERGED DATASET")
print("=" * 84)

print("Eligible            :", n)
print("Batch1              :", len(b1))
print("Batch2              :", len(b2))

print()
print("N00                 :", n00)
print("N01 Delay only      :", n01)
print("N10 RestoreNow only :", n10)
print("N11                 :", n11)

print("Discordant          :", nd)
print(
    "Discordant rate     :",
    f"{100*nd/n:.3f}%"
)

print()
print(
    "Early10 SR          :",
    f"{100*sr10:.3f}%"
)

print(
    "Early20 SR          :",
    f"{100*sr20:.3f}%"
)

print(
    "Early10 - Early20   :",
    f"{100*(sr10-sr20):+.3f} pp"
)

print(
    "Paired exact p      :",
    f"{p_exact:.8f}"
)

print(
    "Oracle SR           :",
    f"{100*oracle:.3f}%"
)

print(
    "Adaptive headroom   :",
    f"{100*headroom:.3f} pp"
)

print()
print(
    "Trace missing       :",
    trace_missing
)


print()
print("=" * 84)
print("BY TARGET")
print("=" * 84)

print(
    f"{'Target':<18}"
    f"{'N':>6}"
    f"{'Restore':>10}"
    f"{'Delay':>10}"
    f"{'Discord':>10}"
)

print("-" * 54)

for target in sorted(target_counts):

    c = target_counts[target]

    nn = sum(c.values())

    print(
        f"{target:<18}"
        f"{nn:>6d}"
        f"{c['N10']:>10d}"
        f"{c['N01']:>10d}"
        f"{c['N10']+c['N01']:>10d}"
    )


print()
print("=" * 84)
print("BY ROOM")
print("=" * 84)

for room in [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]:
    c = room_counts[room]

    print(
        f"{room:<14}"
        f"N00={c['N00']:4d}  "
        f"N01={c['N01']:4d}  "
        f"N10={c['N10']:4d}  "
        f"N11={c['N11']:4d}"
    )


print()
print("All        :", all_path)
print("Preference :", pref_path)

print()
print("ACA SELECTOR MERGE: PASS")
