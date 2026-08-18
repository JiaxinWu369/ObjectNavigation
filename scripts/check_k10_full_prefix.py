import json
from pathlib import Path
from collections import Counter


NC_DIR = Path(
    "results/pair_nocontext_base100"
)

K10_DIR = Path(
    "results/pair_early10_nocontext_base100"
)


def find_vis(root):
    files = list(
        root.rglob(
            "visualization_metrics.json"
        )
    )

    if len(files) != 1:
        raise RuntimeError(
            f"{root}: expected exactly one "
            f"visualization_metrics.json, "
            f"found {files}"
        )

    return files[0]


def state_tuple(s):
    """
    AKGVP offline state:
    x|z|rotation|horizon
    """
    parts = str(s).split("|")

    if len(parts) != 4:
        raise ValueError(
            f"Unexpected state: {s}"
        )

    return (
        round(float(parts[0]), 4),
        round(float(parts[1]), 4),
        int(round(float(parts[2]))),
        int(round(float(parts[3]))),
    )


def make_key(r):
    states = r.get("states", [])

    if not states:
        raise RuntimeError(
            f"No states in record: {r.keys()}"
        )

    return (
        r["scene"],
        str(r["target"]),
        state_tuple(states[0]),
    )


def load_vis(root):
    p = find_vis(root)

    with open(
        p,
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    out = {}

    duplicate = Counter()

    for r in data:
        key = make_key(r)

        if key in out:
            duplicate[key] += 1
        else:
            out[key] = r

    if duplicate:
        print(
            "Duplicate episode identifiers:"
        )

        for k, v in duplicate.most_common(20):
            print(k, v + 1)

        raise RuntimeError(
            "Cannot safely pair by "
            "(scene,target,start_state)."
        )

    return out


nc = load_vis(NC_DIR)
k10 = load_vis(K10_DIR)

print("=" * 80)
print("FULL ACTION PREFIX VALIDATION")
print("=" * 80)

print("NoContext records:", len(nc))
print("Early10 records  :", len(k10))

only_nc = set(nc) - set(k10)
only_k10 = set(k10) - set(nc)

print("only NoContext   :", len(only_nc))
print("only Early10     :", len(only_k10))

if only_nc:
    print(
        "example only NC:",
        list(only_nc)[:5]
    )

if only_k10:
    print(
        "example only K10:",
        list(only_k10)[:5]
    )

assert set(nc) == set(k10)

prefix_mismatch = []
short_episodes = []
success_discordant = []
discordant_short = []

length_mismatch = 0

for key in sorted(nc):

    a = nc[key]
    b = k10[key]

    aa = [
        int(x)
        for x in a["action_list"]
    ]

    bb = [
        int(x)
        for x in b["action_list"]
    ]

    # Number of recorded states should
    # correspond to number of decisions.
    if len(a["states"]) != len(aa):
        length_mismatch += 1

    if len(b["states"]) != len(bb):
        length_mismatch += 1

    # Episode ended before restoration point.
    if min(len(aa), len(bb)) < 10:

        short_episodes.append(
            (
                key,
                len(aa),
                len(bb),
            )
        )

        # Since both policies are identical
        # before step 10, complete trajectories
        # should also be identical if both
        # terminate before step 10.
        if aa != bb:
            prefix_mismatch.append(
                (
                    key,
                    aa,
                    bb,
                )
            )

    else:

        if aa[:10] != bb[:10]:
            prefix_mismatch.append(
                (
                    key,
                    aa[:10],
                    bb[:10],
                )
            )

    sa = bool(a["success"])
    sb = bool(b["success"])

    if sa != sb:
        success_discordant.append(key)

        if min(len(aa), len(bb)) < 10:
            discordant_short.append(
                (
                    key,
                    len(aa),
                    len(bb),
                    sa,
                    sb,
                )
            )


print()
print("Prefix mismatch first 10 :",
      len(prefix_mismatch))

print("Episodes ending <10      :",
      len(short_episodes))

print("Success discordant        :",
      len(success_discordant))

print("Discordant ending <10     :",
      len(discordant_short))

print("state/action len mismatch :",
      length_mismatch)


if prefix_mismatch:
    print()
    print("PREFIX MISMATCH EXAMPLES")

    for x in prefix_mismatch[:10]:
        print(x)


if discordant_short:
    print()
    print(
        "DISCORDANT SHORT EXAMPLES"
    )

    for x in discordant_short[:10]:
        print(x)


assert len(success_discordant) == 79

assert len(prefix_mismatch) == 0, (
    "NoContext and Early10 are not "
    "identical before step 10."
)

assert len(discordant_short) == 0, (
    "Found success reversal before "
    "the intervention point."
)

print()
print(
    "K10 FULL MATCHED PREFIX: PASS"
)
