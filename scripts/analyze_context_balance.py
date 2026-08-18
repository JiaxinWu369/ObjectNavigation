from scripts.analyze_gate_pairs import (
    load_run,
    summarize,
    compare,
)

runs = {
    "Base":
        "results/pair_base50",

    "Uniform-0.5":
        "results/pair_uniform05",

    "Uniform-0.1":
        "results/pair_uniform01",

    "NoContext-0":
        "results/pair_uniform00",

    "AllSemantic-0.1":
        "results/pair_allsemantic01",
}

data = {
    k: load_run(v)
    for k, v in runs.items()
}

print("=" * 92)
print("CONTEXT BALANCE SUMMARY")
print("=" * 92)

for name in runs:
    summarize(name, data[name])

print()
print("#" * 92)
print("AGAINST BASE")
print("#" * 92)

for name in [
    "Uniform-0.5",
    "Uniform-0.1",
    "NoContext-0",
    "AllSemantic-0.1",
]:
    compare(
        f"BASE vs {name}",
        data["Base"],
        data[name],
    )

print()
print("#" * 92)
print("AGAINST UNIFORM-0.1")
print("#" * 92)

for name in [
    "Uniform-0.5",
    "NoContext-0",
    "AllSemantic-0.1",
]:
    compare(
        f"{name} vs UNIFORM-0.1",
        data[name],
        data["Uniform-0.1"],
    )
