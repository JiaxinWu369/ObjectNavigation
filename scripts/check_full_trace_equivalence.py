import json
import re
from pathlib import Path
from collections import defaultdict

OLD = Path(
    "results/pair_nocontext_base100"
)

NEW = Path(
    "results/diag_nocontext_trace100"
)

ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]


def room_from_scene(scene):
    n = int(
        re.search(
            r"\d+",
            scene
        ).group()
    )

    if n < 200:
        return "kitchen"
    if n < 300:
        return "living_room"
    if n < 400:
        return "bedroom"
    return "bathroom"


# Old complete visualization trajectories.
vis_file = (
    OLD
    / "visualization_files"
    / "visualization_metrics.json"
)

old_vis = json.loads(
    vis_file.read_text()
)

old_vis_room = defaultdict(list)

for r in old_vis:
    old_vis_room[
        room_from_scene(r["scene"])
    ].append(r)


# Old JSONL metadata.
old_json = {}

for room in ROOMS:
    p = OLD / f"episodes_{room}.jsonl"

    for line in p.read_text().splitlines():
        if not line.strip():
            continue

        r = json.loads(line)

        old_json[
            (
                room,
                int(r["eval_index"])
            )
        ] = r


# New JSONL.
new_json = {}

for room in ROOMS:
    p = NEW / f"episodes_{room}.jsonl"

    for line in p.read_text().splitlines():
        if not line.strip():
            continue

        r = json.loads(line)

        new_json[
            (
                room,
                int(r["eval_index"])
            )
        ] = r


assert set(old_json) == set(new_json)

action_mismatch = []
state_mismatch = []
outcome_mismatch = []
length_mismatch = []

for key in sorted(new_json):

    room, idx = key

    old_meta = old_json[key]
    new = new_json[key]

    vis = old_vis_room[
        room
    ][idx]

    assert (
        old_meta["scene"]
        ==
        new["scene"]
    )

    assert (
        old_meta["target"]
        ==
        new["target"]
    )

    if (
        bool(old_meta["success"])
        !=
        bool(new["success"])
    ):
        outcome_mismatch.append(key)

    if (
        int(old_meta["ep_length"])
        !=
        int(new["ep_length"])
    ):
        length_mismatch.append(
            (
                key,
                old_meta["ep_length"],
                new["ep_length"],
            )
        )

    old_actions = [
        int(x)
        for x in vis["action_list"]
    ]

    if (
        old_actions
        !=
        new["full_actions"]
    ):
        action_mismatch.append(key)

    if (
        vis["states"]
        !=
        new["full_states"]
    ):
        state_mismatch.append(key)


print("Episodes         :", len(new_json))
print("Outcome mismatch :", len(outcome_mismatch))
print("Length mismatch  :", len(length_mismatch))
print("Action mismatch  :", len(action_mismatch))
print("State mismatch   :", len(state_mismatch))

assert len(new_json) == 1260
assert not outcome_mismatch
assert not length_mismatch
assert not action_mismatch
assert not state_mismatch

print()
print(
    "FULL TRACE NON-INVASIVE EQUIVALENCE: PASS"
)
