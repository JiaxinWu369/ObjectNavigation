import json
from pathlib import Path
from collections import Counter

ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

ROOT = Path("results/guard_tau02_zsval")


def load():
    rows = []

    for room in ROOMS:
        p = ROOT / f"episodes_{room}.jsonl"

        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))

    return rows


def max_stall_streak(x):

    actions = x["full_actions"]
    states = x["full_states"]

    cur = 0
    best = 0

    for t, a in enumerate(actions):

        # MoveAhead = 0
        stalled = (
            a == 0
            and
            t + 1 < len(states)
            and
            states[t + 1] == states[t]
        )

        if stalled:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return best


rows = load()

print("episodes:", len(rows))

for k in [2, 3, 4, 5, 6, 8, 10]:

    succ = 0
    fail = 0

    for x in rows:

        triggered = (
            max_stall_streak(x) >= k
        )

        if not triggered:
            continue

        if x["success"]:
            succ += 1
        else:
            fail += 1

    print(
        f"k={k:<2}",
        f"success_trigger={succ:<4}",
        f"failure_trigger={fail:<4}",
    )

print()
print("STREAK DISTRIBUTION")

succ_c = Counter()
fail_c = Counter()

for x in rows:

    s = max_stall_streak(x)

    if x["success"]:
        succ_c[s] += 1
    else:
        fail_c[s] += 1

for s in sorted(
    set(succ_c) | set(fail_c)
):
    if s >= 1:
        print(
            f"streak={s:<3}",
            f"success={succ_c[s]:<4}",
            f"fail={fail_c[s]:<4}",
        )
