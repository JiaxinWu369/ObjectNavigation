from scripts.analyze_gate_pairs import (
    load_run,
    summarize,
    compare,
)

base = load_run(
    "results/pair_base100"
)

e1 = load_run(
    "results/pair_early1_nocontext_base100"
)

e3 = load_run(
    "results/pair_early3_nocontext_base100"
)

e5 = load_run(
    "results/pair_early5_nocontext_base100"
)

full = load_run(
    "results/pair_nocontext_base100"
)


print("=" * 100)
print("EARLY-ONLY CONTEXT SUPPRESSION — 100K")
print("=" * 100)

summarize("Base", base)
summarize("Early-1", e1)
summarize("Early-3", e3)
summarize("Early-5", e5)
summarize("NoContext-all", full)


for name, run in [
    ("Early-1", e1),
    ("Early-3", e3),
    ("Early-5", e5),
]:
    print()
    print("#" * 100)
    print("BASE vs", name)
    print("#" * 100)

    compare(
        f"Base vs {name}",
        base,
        run,
    )


for name, run in [
    ("Early-1", e1),
    ("Early-3", e3),
    ("Early-5", e5),
]:
    print()
    print("#" * 100)
    print(name, "vs NoContext-all")
    print("#" * 100)

    compare(
        f"{name} vs NoContext-all",
        run,
        full,
    )
