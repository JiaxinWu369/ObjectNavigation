from scripts.analyze_gate_pairs import (
    load_run,
    summarize,
    compare,
)

runs = {
    "Base":
        "results/pair_base50",

    "Uniform-0":
        "results/pair_uniform01",

    "Uniform-5":
        "results/pair_uniform01_s5",

    "Uniform-10":
        "results/pair_uniform01_s10",

    "Uniform-20":
        "results/pair_uniform01_s20",
}

data = {
    name: load_run(path)
    for name, path in runs.items()
}

print("=" * 92)
print("DELAYED UNIFORM SUMMARY")
print("=" * 92)

for name in runs:
    summarize(
        name,
        data[name],
    )

print()
print("#" * 92)
print("AGAINST BASE")
print("#" * 92)

for name in [
    "Uniform-0",
    "Uniform-5",
    "Uniform-10",
    "Uniform-20",
]:
    compare(
        f"BASE vs {name}",
        data["Base"],
        data[name],
    )

print()
print("#" * 92)
print("DIRECT COMPARISONS AGAINST UNIFORM-0")
print("#" * 92)

for name in [
    "Uniform-5",
    "Uniform-10",
    "Uniform-20",
]:
    compare(
        f"UNIFORM-0 vs {name}",
        data["Uniform-0"],
        data[name],
    )
