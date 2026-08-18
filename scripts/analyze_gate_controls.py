from scripts.analyze_gate_pairs import (
    load_run,
    summarize,
    compare,
)

runs = {
    "Base":
        "results/pair_base50",

    "Strong":
        "results/pair_strong01",

    "RandomStep":
        "results/pair_random01",

    "Uniform":
        "results/pair_uniform01",

    "RandomEpisode":
        "results/pair_episode_random01",
}

data = {
    name: load_run(path)
    for name, path in runs.items()
}

print("=" * 90)
print("CONTROL SUMMARY")
print("=" * 90)

for name in runs:
    summarize(
        name,
        data[name],
    )

for name in [
    "Strong",
    "RandomStep",
    "Uniform",
    "RandomEpisode",
]:
    compare(
        f"BASE vs {name}",
        data["Base"],
        data[name],
    )

print()
print("#" * 90)
print("DIRECT COMPARISONS AGAINST STRONG")
print("#" * 90)

for name in [
    "RandomStep",
    "Uniform",
    "RandomEpisode",
]:
    compare(
        f"{name} vs STRONG",
        data[name],
        data["Strong"],
    )
