import json
import re
from pathlib import Path
from collections import defaultdict


OLD = Path(
    "results/pair_nocontext_base100"
)

NEW = Path(
    "results/smoke_nocontext_trace100"
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


# ---------------------------------------------------------
# Old authoritative complete trajectories:
# visualization action_list/states.
# ---------------------------------------------------------

vis_file = (
    OLD
    / "visualization_files"
    / "visualization_metrics.json"
)

vis = json.loads(
    vis_file.read_text()
)

old_by_room = defaultdict(list)

for row in vis:
    old_by_room[
        room_from_scene(
            row["scene"]
        )
    ].append(row)


# ---------------------------------------------------------
# Old JSONL outcomes.
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Compare new smoke.
# ---------------------------------------------------------

checked = 0

for room in ROOMS:

    p = NEW / f"episodes_{room}.jsonl"

    for line in p.read_text().splitlines():

        if not line.strip():
            continue

        new = json.loads(line)

        idx = int(
            new["eval_index"]
        )

        key = (
            room,
            idx
        )

        old_meta = old_json[key]
        old_vis = old_by_room[
            room
        ][idx]

        assert (
            new["scene"]
            ==
            old_meta["scene"]
        )

        assert (
            new["target"]
            ==
            old_meta["target"]
        )

        assert (
            bool(new["success"])
            ==
            bool(old_meta["success"])
        )

        assert (
            int(new["ep_length"])
            ==
            int(old_meta["ep_length"])
        )

        assert (
            new["full_actions"]
            ==
            [
                int(x)
                for x
                in old_vis["action_list"]
            ]
        )

        assert (
            new["full_states"]
            ==
            old_vis["states"]
        )

        checked += 1


print(
    "Compared episodes:",
    checked
)

print(
    "TRACE NON-INVASIVE EQUIVALENCE: PASS"
)
