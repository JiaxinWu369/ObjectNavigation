import json
from pathlib import Path
from collections import Counter
from scipy.stats import binomtest


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

ROOTS = {
    "Base": Path("results/base250_zsval"),
    "NoContext": Path("results/nocontext250_zsval"),
    "Early40": Path("results/early40_250_zsval"),
}


def load(root):
    out = {}

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        if not p.exists():
            raise FileNotFoundError(p)

        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)

                key = (
                    room,
                    int(r["eval_index"]),
                )

                if key in out:
                    raise RuntimeError(
                        f"Duplicate key: {key}"
                    )

                out[key] = r

    assert len(out) == 1260
    return out


runs = {
    name: load(root)
    for name, root in ROOTS.items()
}

keys = set(runs["Base"])

assert keys == set(runs["NoContext"])
assert keys == set(runs["Early40"])


def compare(a_name, b_name):
    """
    N01 = A fails, B succeeds = B-only
    N10 = A succeeds, B fails = A-only
    """

    A = runs[a_name]
    B = runs[b_name]

    c = Counter()
    target = {}

    for key in sorted(keys):
        a = A[key]
        b = B[key]

        assert a["scene"] == b["scene"]
        assert a["target"] == b["target"]

        ya = int(bool(a["success"]))
        yb = int(bool(b["success"]))

        state = f"N{ya}{yb}"
        c[state] += 1

        t = a["target"]

        if t not in target:
            target[t] = Counter()

        target[t][state] += 1

    discordant = (
        c["N01"] + c["N10"]
    )

    if discordant:
        p = binomtest(
            c["N01"],
            discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    else:
        p = 1.0

    sr_a = (
        c["N10"] + c["N11"]
    ) / 1260

    sr_b = (
        c["N01"] + c["N11"]
    ) / 1260

    oracle = (
        c["N01"]
        + c["N10"]
        + c["N11"]
    ) / 1260

    print()
    print("=" * 86)
    print(f"{a_name}  vs  {b_name}")
    print("=" * 86)

    print("N00 both fail :", c["N00"])
    print(
        f"N01 {b_name}-only :",
        c["N01"]
    )
    print(
        f"N10 {a_name}-only :",
        c["N10"]
    )
    print("N11 both success:", c["N11"])

    print()
    print(
        f"{a_name} SR : "
        f"{100*sr_a:.3f}%"
    )

    print(
        f"{b_name} SR : "
        f"{100*sr_b:.3f}%"
    )

    print(
        f"Delta {b_name}-{a_name}: "
        f"{100*(sr_b-sr_a):+.3f} pp"
    )

    print(
        "Discordant:",
        discordant
    )

    print(
        "Exact paired p:",
        f"{p:.6g}"
    )

    print(
        "Hindsight union SR:",
        f"{100*oracle:.3f}%"
    )

    print(
        "Headroom over best fixed:",
        f"{100*(oracle-max(sr_a,sr_b)):.3f} pp"
    )

    print()
    print(
        f"{'Target':<16}"
        f"{b_name+' only':>14}"
        f"{a_name+' only':>14}"
        f"{'Net B-A':>10}"
    )

    print("-" * 56)

    for t in sorted(target):
        b_only = target[t]["N01"]
        a_only = target[t]["N10"]

        if b_only + a_only == 0:
            continue

        print(
            f"{t:<16}"
            f"{b_only:>14d}"
            f"{a_only:>14d}"
            f"{b_only-a_only:>10d}"
        )


compare("Base", "NoContext")
compare("Base", "Early40")
compare("NoContext", "Early40")


# ============================================================
# K40 matched-prefix verification:
# NoContext and Early40 both suppress actions 0..39.
# First restored action for Early40 is action index 40.
# ============================================================

NC = runs["NoContext"]
E40 = runs["Early40"]

prefix_mismatch = 0
state40_mismatch = 0
pre40_termination_mismatch = 0
reached_state40 = 0

examples = []

for key in sorted(keys):
    a = NC[key]
    b = E40[key]

    aa = a["full_actions"]
    bb = b["full_actions"]

    sa = a["full_states"]
    sb = b["full_states"]

    # If either episode terminates before the
    # first restored action, the histories/outcomes
    # must be identical.
    if len(aa) <= 40 or len(bb) <= 40:

        if (
            aa != bb
            or sa != sb
            or bool(a["success"])
               != bool(b["success"])
        ):
            pre40_termination_mismatch += 1

            if len(examples) < 5:
                examples.append(
                    (
                        key,
                        "pre40",
                        len(aa),
                        len(bb),
                    )
                )

        continue

    reached_state40 += 1

    # Actions 0..39 must be identical.
    if aa[:40] != bb[:40]:
        prefix_mismatch += 1

        if len(examples) < 5:
            examples.append(
                (
                    key,
                    "actions",
                )
            )

    # states[40] is the pre-action state
    # for the first restored action.
    if sa[40] != sb[40]:
        state40_mismatch += 1

        if len(examples) < 5:
            examples.append(
                (
                    key,
                    "state40",
                )
            )


print()
print("=" * 86)
print("K40 MATCHED INTERVENTION CHECK")
print("=" * 86)

print(
    "Episodes:",
    len(keys)
)

print(
    "Reached state40:",
    reached_state40
)

print(
    "First-40 action mismatch:",
    prefix_mismatch
)

print(
    "t=40 state mismatch:",
    state40_mismatch
)

print(
    "Pre-restoration termination mismatch:",
    pre40_termination_mismatch
)

if examples:
    print(
        "Examples:",
        examples
    )

assert prefix_mismatch == 0
assert state40_mismatch == 0
assert pre40_termination_mismatch == 0

print()
print(
    "K40 FULL MATCHED INTERVENTION: PASS"
)
