import json
import pickle
import random
from pathlib import Path
from collections import Counter, defaultdict

from datasets.environment import Environment
from datasets.constants import AI2THOR_TARGET_CLASSES


SEED = 20260819
EPISODES_PER_SCENE = 50

OUT_DIR = Path("test_val_split")
OUT_DIR.mkdir(parents=True, exist_ok=True)


UNSEEN = {
    "Bowl",
    "DeskLamp",
    "Laptop",
    "LightSwitch",
    "Plate",
    "StoveBurner",
}

ALL_TARGETS = list(
    AI2THOR_TARGET_CLASSES[22]
)

SEEN = [
    x for x in ALL_TARGETS
    if x not in UNSEEN
]


ROOM_SCENES = {
    "kitchen": [
        f"FloorPlan{i}"
        for i in range(1, 21)
    ],

    "living_room": [
        f"FloorPlan{i}"
        for i in range(201, 221)
    ],

    "bedroom": [
        f"FloorPlan{i}"
        for i in range(301, 321)
    ],

    "bathroom": [
        f"FloorPlan{i}"
        for i in range(401, 421)
    ],
}


rng = random.Random(SEED)


env = Environment(
    use_offline_controller=True,
    grid_size=0.25,
    fov=100.0,
    offline_data_dir="datasets/Scene_Data",
    detection_feature_file_name=
        "det_feature_22_cates.hdf5",
    images_file_name=
        "clip_featuremap.hdf5",

    # Important:
    # raw state -> object metadata.
    visible_object_map_file_name=
        "metadata.json",

    optimal_action_file_name=
        "optimal_action.json",
)


summary = {}

first_scene = True


for room, scenes in ROOM_SCENES.items():

    episodes = []

    room_targets = Counter()
    scene_counts = Counter()

    print()
    print("=" * 100)
    print("ROOM:", room)
    print("=" * 100)

    for scene in scenes:

        if first_scene:
            env.start(scene)
            first_scene = False
        else:
            env.reset(scene)

        ctrl = env.controller

        states = sorted(
            list(ctrl.images.keys())
        )

        if len(states) < EPISODES_PER_SCENE:
            raise RuntimeError(
                f"{scene}: only {len(states)} states"
            )

        # Raw metadata is:
        # state -> {"objects": [...]}
        #
        # Object list normally contains every object
        # in the scene. Build a scene-wide union anyway
        # so the generator is robust.

        target_to_ids = defaultdict(set)

        # One state is normally sufficient, but use
        # a small deterministic sample to guard against
        # unusual metadata serialization.
        probe_states = states[
            :min(10, len(states))
        ]

        for s in probe_states:

            meta = ctrl.metadata[s]

            for obj in meta.get(
                "objects",
                []
            ):

                obj_type = obj.get(
                    "objectType"
                )

                obj_id = obj.get(
                    "objectId"
                )

                if (
                    obj_type in SEEN
                    and obj_id is not None
                ):
                    target_to_ids[
                        obj_type
                    ].add(obj_id)

        valid_targets = sorted(
            target_to_ids.keys()
        )

        if not valid_targets:
            raise RuntimeError(
                f"{scene}: no seen targets"
            )

        # ------------------------------------------------
        # 50 unique initial states.
        # ------------------------------------------------

        selected_states = rng.sample(
            states,
            EPISODES_PER_SCENE,
        )

        # ------------------------------------------------
        # Approximately target-balanced schedule within
        # this scene.
        # ------------------------------------------------

        target_schedule = []

        while len(
            target_schedule
        ) < EPISODES_PER_SCENE:

            block = valid_targets.copy()
            rng.shuffle(block)

            target_schedule.extend(
                block
            )

        target_schedule = (
            target_schedule[
                :EPISODES_PER_SCENE
            ]
        )

        # Shuffle once more to avoid periodic ordering.
        rng.shuffle(
            target_schedule
        )

        for local_idx, (
            state,
            target,
        ) in enumerate(
            zip(
                selected_states,
                target_schedule,
            )
        ):

            obj_ids = sorted(
                target_to_ids[target]
            )

            if len(obj_ids) == 0:
                raise RuntimeError(
                    f"{scene}/{target}: "
                    "empty task_data"
                )

            episode_id = (
                f"{room}|"
                f"{scene}|"
                f"{local_idx:03d}"
            )

            ep = {
                "scene":
                    scene,

                "state":
                    str(state),

                "goal_object_type":
                    target,

                "task_data":
                    obj_ids,

                "aca_episode_id":
                    episode_id,
            }

            episodes.append(ep)

            room_targets[target] += 1
            scene_counts[scene] += 1

        print(
            f"{scene:<14} "
            f"states={len(states):>5} "
            f"valid_targets={len(valid_targets):>2} "
            f"episodes={scene_counts[scene]:>2}"
        )

    expected = (
        len(scenes)
        *
        EPISODES_PER_SCENE
    )

    assert len(episodes) == expected

    out = (
        OUT_DIR /
        f"{room}_aca_train_22.pkl"
    )

    with open(
        out,
        "wb",
    ) as f:
        pickle.dump(
            episodes,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    summary[room] = {
        "episodes":
            len(episodes),

        "target_counts":
            dict(
                sorted(
                    room_targets.items()
                )
            ),
    }

    print()
    print("Saved:", out)
    print("Episodes:", len(episodes))
    print(
        "Targets:",
        dict(
            sorted(
                room_targets.items()
            )
        )
    )


total = sum(
    x["episodes"]
    for x in summary.values()
)

assert total == 4000


summary_out = Path(
    "results/aca_train500/"
    "train_split_summary.json"
)

with open(
    summary_out,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "seed": SEED,
            "seen_targets": SEEN,
            "unseen_targets":
                sorted(UNSEEN),
            "rooms": summary,
            "total": total,
        },
        f,
        indent=2,
    )


print()
print("=" * 100)
print("ACA SELECTOR TRAIN SPLIT")
print("=" * 100)

print("Seen targets :", len(SEEN))
print("Total specs  :", total)

for room in ROOM_SCENES:
    print(
        f"{room:<14}:",
        summary[room]["episodes"],
    )

print()
print("Summary:", summary_out)
print()
print("ACA TRAIN SPLIT: PASS")
