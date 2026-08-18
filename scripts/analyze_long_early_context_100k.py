from scripts.analyze_gate_pairs import (
    load_run,
    summarize,
    compare,
)

base = load_run(
    "results/pair_base100"
)

k10 = load_run(
    "results/pair_early10_nocontext_base100"
)

k20 = load_run(
    "results/pair_early20_nocontext_base100"
)

k40 = load_run(
    "results/pair_early40_nocontext_base100"
)

full = load_run(
    "results/pair_nocontext_base100"
)


print("=" * 100)
print("LONG EARLY CONTEXT SUPPRESSION — 100K")
print("=" * 100)

summarize("Base", base)
summarize("Early-10", k10)
summarize("Early-20", k20)
summarize("Early-40", k40)
summarize("NoContext", full)


comparisons = [
    ("Base", base, "Early-10", k10),
    ("Base", base, "Early-20", k20),
    ("Base", base, "Early-40", k40),

    ("NoContext", full, "Early-10", k10),
    ("NoContext", full, "Early-20", k20),
    ("NoContext", full, "Early-40", k40),

    ("Early-20", k20, "Early-40", k40),
]

for name_a, a, name_b, b in comparisons:

    print()
    print("#" * 100)
    print(f"{name_a} vs {name_b}")
    print("#" * 100)

    compare(
        f"{name_a} vs {name_b}",
        a,
        b,
    )
