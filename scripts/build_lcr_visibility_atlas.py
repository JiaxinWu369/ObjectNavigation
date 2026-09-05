import argparse
import json
import math
import os
from collections import defaultdict
from glob import glob

from ai2thor.controller import Controller


SEEN_TARGETS = [
    "AlarmClock",
    "Book",
    "CellPhone",
    "Chair",
    "CoffeeMachine",
    "FloorLamp",
    "Fridge",
    "GarbageCan",
    "Kettle",
    "Microwave",
    "Pan",
    "Pot",
    "RemoteControl",
    "Sink",
    "Television",
    "Toaster",
]


def region_key(x, z):
    return (
        int(math.floor(float(x))),
        int(math.floor(float(z))),
    )


def region_str(region):
    return "{},{}".format(
        int(region[0]),
        int(region[1]),
    )


def parse_state(state):
    if isinstance(state, dict):
        if "position" in state:
            p = state["position"]
            return (
                float(p["x"]),
                float(p["z"]),
            )

        if "x" in state and "z" in state:
            return (
                float(state["x"]),
                float(state["z"]),
            )

    parts = str(state).split("|")

    if len(parts) < 2:
        raise ValueError(
            "Cannot parse state: {}".format(
                state
            )
        )

    return (
        float(parts[0]),
        float(parts[1]),
    )


def get_scene(row):
    for key in [
        "scene",
        "scene_name",
        "sceneName",
    ]:
        if key in row:
            return str(row[key])

    raise KeyError(
        "Cannot find scene. keys={}".format(
            sorted(row.keys())
        )
    )


def get_target(row):
    for key in [
        "target",
        "target_object",
        "goal",
        "goal_object_type",
    ]:
        if key in row:
            return str(row[key])

    raise KeyError(
        "Cannot find target. keys={}".format(
            sorted(row.keys())
        )
    )


def load_needed_pairs(root):
    needed_regions = defaultdict(set)
    needed_pairs = defaultdict(set)

    total_eps = 0

    for path in sorted(
        glob(
            os.path.join(
                root,
                "episodes_*.jsonl",
            )
        )
    ):
        with open(
            path,
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                ep = json.loads(line)

                scene = get_scene(ep)
                target = get_target(ep)

                if target not in SEEN_TARGETS:
                    continue

                total_eps += 1

                for item in ep.get(
                    "context_trace",
                    []
                ):
                    state = item.get("state")

                    if state is None:
                        continue

                    x, z = parse_state(state)

                    r = region_key(
                        x,
                        z,
                    )

                    needed_regions[
                        scene
                    ].add(r)

                    needed_pairs[
                        (scene, target)
                    ].add(r)

    print(
        "episodes scanned:",
        total_eps,
    )

    print(
        "scenes found:",
        len(needed_regions),
    )

    return (
        needed_regions,
        needed_pairs,
    )


def min_center_distance(
    region,
    positions,
):
    cx = region[0] + 0.5
    cz = region[1] + 0.5

    if not positions:
        return None

    return min(
        math.sqrt(
            (cx - p["x"]) ** 2
            +
            (cz - p["z"]) ** 2
        )
        for p in positions
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--episodes-root",
    default="results/lcr_full_base500",
)

parser.add_argument(
    "--output",
    required=True,
)

parser.add_argument(
    "--scenes",
    default="",
)

args = parser.parse_args()


needed_regions, needed_pairs = (
    load_needed_pairs(
        args.episodes_root
    )
)


if args.scenes.strip():
    selected_scenes = [
        x.strip()
        for x in args.scenes.split(",")
        if x.strip()
    ]
else:
    selected_scenes = sorted(
        needed_regions.keys()
    )


print(
    "selected scenes:",
    selected_scenes,
)


controller = Controller()
controller.start()

atlas = {
    "cell_size": 1.0,
    "yaw_values": [
        0,
        90,
        180,
        270,
    ],
    "horizon_values": [
        -30,
        0,
        30,
    ],
    "scenes": {},
}


for scene_i, scene in enumerate(
    selected_scenes,
    1,
):

    print()
    print(
        "[{}/{}] {}".format(
            scene_i,
            len(selected_scenes),
            scene,
        )
    )

    controller.reset(scene)

    ev = controller.step(
        dict(
            action="GetReachablePositions"
        )
    )

    if not ev.metadata.get(
        "lastActionSuccess",
        False,
    ):
        raise RuntimeError(
            "{} reachable failed: {}".format(
                scene,
                ev.metadata.get(
                    "errorMessage"
                ),
            )
        )

    reachable = ev.metadata.get(
        "actionReturn",
        [],
    )

    objects = ev.metadata.get(
        "objects",
        [],
    )

    target_positions = defaultdict(list)

    for obj in objects:
        t = obj.get("objectType")

        if t not in SEEN_TARGETS:
            continue

        p = obj.get("position")

        if p is None:
            continue

        target_positions[t].append(
            {
                "x": float(p["x"]),
                "z": float(p["z"]),
            }
        )

    positions_by_region = defaultdict(list)

    wanted = needed_regions.get(
        scene,
        set(),
    )

    for p in reachable:

        r = region_key(
            p["x"],
            p["z"],
        )

        if r in wanted:
            positions_by_region[
                r
            ].append(p)

    scene_record = {
        "num_reachable":
            len(reachable),

        "target_positions":
            dict(target_positions),

        "regions": {},
    }

    wanted_sorted = sorted(wanted)

    for region_i, r in enumerate(
        wanted_sorted,
        1,
    ):

        if (
            region_i == 1
            or
            region_i % 5 == 0
            or
            region_i == len(wanted_sorted)
        ):
            print(
                "  region {}/{}".format(
                    region_i,
                    len(wanted_sorted),
                ),
                flush=True,
            )

        positions = positions_by_region.get(
            r,
            [],
        )

        # Fast label-audit mode:
        # use at most three reachable positions
        # closest to the center of this 1m region.
        cx = r[0] + 0.5
        cz = r[1] + 0.5

        positions = sorted(
            positions,
            key=lambda p: (
                (float(p["x"]) - cx) ** 2
                +
                (float(p["z"]) - cz) ** 2
            ),
        )

        positions = positions[:1]

        visible_targets = set()

        required_targets = {
            target
            for (
                s,
                target
            ), regions in
            needed_pairs.items()
            if (
                s == scene
                and
                r in regions
            )
        }

        if not positions:
            print(
                "WARNING no reachable "
                "positions in region",
                scene,
                r,
            )

        done = False

        for p in positions:

            for yaw in [
                0,
                90,
                180,
                270,
            ]:

                for horizon in [
                    -30,
                    0,
                    30,
                ]:

                    ev = controller.step(
                        dict(
                            action="TeleportFull",
                            x=float(p["x"]),
                            y=float(p["y"]),
                            z=float(p["z"]),
                            rotation=float(yaw),
                            horizon=float(
                                horizon
                            ),
                        )
                    )

                    if not ev.metadata.get(
                        "lastActionSuccess",
                        False,
                    ):
                        continue

                    for obj in ev.metadata.get(
                        "objects",
                        [],
                    ):
                        if (
                            obj.get(
                                "visible",
                                False,
                            )
                            and
                            obj.get(
                                "objectType"
                            )
                            in required_targets
                        ):
                            visible_targets.add(
                                obj[
                                    "objectType"
                                ]
                            )

                    # Only need existence.
                    if (
                        required_targets
                        and
                        required_targets
                        <= visible_targets
                    ):
                        done = True
                        break

                if done:
                    break

            if done:
                break

        scene_record[
            "regions"
        ][region_str(r)] = {
            "num_reachable_positions":
                len(positions),

            "visible_targets":
                sorted(
                    visible_targets
                ),
        }

    atlas[
        "scenes"
    ][scene] = scene_record


controller.stop()


os.makedirs(
    os.path.dirname(args.output),
    exist_ok=True,
)

with open(
    args.output,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        atlas,
        f,
        indent=2,
    )


labels_path = (
    os.path.splitext(
        args.output
    )[0]
    + "_labels.jsonl"
)


total = 0
visibility_pos = 0
visibility_neg = 0

prox_pos = 0
prox_neg = 0
prox_ignore = 0

comparable = 0
disagree = 0

missing_region = 0
missing_target = 0


with open(
    labels_path,
    "w",
    encoding="utf-8",
) as out:

    for (
        scene,
        target
    ), regions in sorted(
        needed_pairs.items()
    ):

        if scene not in atlas["scenes"]:
            continue

        scene_data = (
            atlas[
                "scenes"
            ][scene]
        )

        target_pos = (
            scene_data[
                "target_positions"
            ].get(
                target,
                [],
            )
        )

        if not target_pos:
            missing_target += len(
                regions
            )
            continue

        for r in sorted(regions):

            rkey = region_str(r)

            region_data = (
                scene_data[
                    "regions"
                ].get(rkey)
            )

            if region_data is None:
                missing_region += 1
                continue

            visible = int(
                target
                in region_data[
                    "visible_targets"
                ]
            )

            distance = (
                min_center_distance(
                    r,
                    target_pos,
                )
            )

            if distance <= 1.5:
                prox = 1
                prox_pos += 1

            elif distance >= 2.5:
                prox = 0
                prox_neg += 1

            else:
                prox = None
                prox_ignore += 1

            total += 1

            if visible:
                visibility_pos += 1
            else:
                visibility_neg += 1

            if prox is not None:
                comparable += 1

                if visible != prox:
                    disagree += 1

            row = {
                "scene":
                    scene,

                "target":
                    target,

                "region":
                    [
                        int(r[0]),
                        int(r[1]),
                    ],

                "visibility_label":
                    visible,

                "proximity_label":
                    prox,

                "min_center_distance":
                    distance,

                "num_reachable_positions":
                    region_data[
                        "num_reachable_positions"
                    ],
            }

            out.write(
                json.dumps(row)
                + "\n"
            )


print()
print("=" * 80)
print("VISIBILITY LABEL AUDIT")
print("=" * 80)

print(
    "region-target pairs :",
    total,
)

print(
    "visibility positive :",
    visibility_pos,
)

print(
    "visibility negative :",
    visibility_neg,
)

if total:
    print(
        "visibility posrate :",
        visibility_pos / total,
    )

print()
print(
    "proximity positive  :",
    prox_pos,
)

print(
    "proximity negative  :",
    prox_neg,
)

print(
    "proximity ignored   :",
    prox_ignore,
)

print()
print(
    "comparable labels   :",
    comparable,
)

print(
    "disagreements       :",
    disagree,
)

if comparable:
    print(
        "disagreement rate  :",
        disagree / comparable,
    )

print()
print(
    "missing region      :",
    missing_region,
)

print(
    "missing target      :",
    missing_target,
)

print()
print(
    "atlas saved:",
    args.output,
)

print(
    "labels saved:",
    labels_path,
)
