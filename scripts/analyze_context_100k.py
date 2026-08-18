from scripts.analyze_gate_pairs import (
    load_run,
    summarize,
    compare,
)

base = load_run(
    "results/pair_base100"
)

uniform = load_run(
    "results/pair_uniform01_base100"
)

nocontext = load_run(
    "results/pair_nocontext_base100"
)


print("=" * 92)
print("100K CONTEXT ROBUSTNESS SUMMARY")
print("=" * 92)

summarize(
    "Base-100k",
    base,
)

summarize(
    "Uniform-0.1",
    uniform,
)

summarize(
    "NoContext",
    nocontext,
)


print()
print("#" * 92)
print("BASE vs UNIFORM-0.1")
print("#" * 92)

compare(
    "BASE-100K vs UNIFORM-0.1",
    base,
    uniform,
)


print()
print("#" * 92)
print("BASE vs NOCONTEXT")
print("#" * 92)

compare(
    "BASE-100K vs NOCONTEXT",
    base,
    nocontext,
)


print()
print("#" * 92)
print("UNIFORM-0.1 vs NOCONTEXT")
print("#" * 92)

compare(
    "UNIFORM-0.1 vs NOCONTEXT",
    uniform,
    nocontext,
)
