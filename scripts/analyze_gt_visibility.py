from scripts.analyze_gate_pairs import (
    load_run,
    summarize,
    compare,
)

base = load_run(
    "results/pair_base50"
)

uniform = load_run(
    "results/pair_uniform01"
)

strong = load_run(
    "results/pair_strong01"
)

gt = load_run(
    "results/pair_gt_visibility01"
)


print("=" * 88)
print("GT VISIBILITY ORACLE SUMMARY")
print("=" * 88)

summarize(
    "Base",
    base,
)

summarize(
    "Strong",
    strong,
)

summarize(
    "Uniform",
    uniform,
)

summarize(
    "GT-Visible",
    gt,
)


compare(
    "BASE vs GT-VISIBILITY",
    base,
    gt,
)

compare(
    "UNIFORM vs GT-VISIBILITY",
    uniform,
    gt,
)

compare(
    "STRONG vs GT-VISIBILITY",
    strong,
    gt,
)
