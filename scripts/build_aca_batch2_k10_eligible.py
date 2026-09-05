import json
import pickle
from pathlib import Path
from collections import Counter


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

ROOT = Path(
    "results/"
    "aca_train500_batch2_early20"
)


def load_log(room):

    p = ROOT / (
        f"episodes_{room}.jsonl"
    )

    rows = {}

    with open(
        p,
        encoding="utf-8",
    ) as f:

        for file_idx, line in enumerate(f):

            if not line.strip():
                continue

            r = json.loads(line)

            idx = int(
                r.get(
                    "eval_index",
                    file_idx,
                )
            )

            rows[idx] = r

    return rows


def actions(r):

    for key in [
        "full_actions",
        "action_list",
        "actions",
    ]:
        if key in r:
            return r[key]

    raise KeyError(
        "No full action trajectory"
    )


total = 0
eligible_total = 0


for room in ROOMS:

    source_path = Path(
        f"test_val_split/"
        f"{room}_aca_train2_22.pkl"
    )

    specs = pickle.load(
        open(source_path, "rb")
    )

    logs = load_log(room)

    assert len(specs) == 2000
    assert len(logs) == 2000

    eligible = []

    target_counter = Counter()

    for idx in range(2000):

        r = logs[idx]

        aa = actions(r)

        # Action index 10 exists only if
        # len(actions) > 10.
        if len(aa) <= 10:
            continue

        ep = dict(specs[idx])

        ep[
            "aca_source_eval_index"
        ] = idx

        eligible.append(ep)

        target_counter[
            ep["goal_object_type"]
        ] += 1

    out = Path(
        f"test_val_split/"
        f"{room}_aca_train2_k10_22.pkl"
    )

    with open(
        out,
        "wb",
    ) as f:

        pickle.dump(
            eligible,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    total += len(specs)
    eligible_total += len(eligible)

    print()
    print(room)
    print(
        "Source   :",
        len(specs),
    )
    print(
        "Eligible :",
        len(eligible),
    )
    print(
        "Rate     :",
        f"{100*len(eligible)/len(specs):.3f}%"
    )
    print(
        "Targets  :",
        dict(
            sorted(
                target_counter.items()
            )
        )
    )


print()
print("=" * 72)
print("ACA BATCH2 K10 ELIGIBILITY")
print("=" * 72)

print(
    "Source episodes:",
    total,
)

print(
    "Eligible       :",
    eligible_total,
)

print(
    "Eligibility    :",
    f"{100*eligible_total/total:.3f}%"
)

print()
print(
    "ACA BATCH2 K10 ELIGIBLE: PASS"
)
