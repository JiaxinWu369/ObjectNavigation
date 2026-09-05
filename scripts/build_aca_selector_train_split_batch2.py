import json
import pickle
import random
from pathlib import Path
from collections import Counter, defaultdict

from datasets.environment import Environment
from datasets.constants import AI2THOR_TARGET_CLASSES


SEED = 20260820
EPISODES_PER_SCENE = 100

OUT_DIR = Path("test_val_split")

RESULT_DIR = Path(
    "results/aca_train500/batch2"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
    x
    for x in ALL_TARGETS
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


# --------------------------------------------------
# Existing Batch-1 start states.
# --------------------------------------------------

used_states = defaultdict(set)

for room in ROOM_SCENES:

    p = (
        OUT_DIR /
        f"{room}_aca_train_22.pkl"
    )

    with open(p, "rb") as f:
        old = pickle.load(f)

    for ep in old:

        used_states[
            ep["scene"]
        ].add(
            str(ep["state"])
        )


rng = random.Random(SEED)


env = Environment(
    use_offline_controller=True,
    grid_size=0.25,
    fov=100.0,
    offline_data_dir=
        "datasets/Scene_Data",
    detection_feature_file_name=
        "det_feature_22_cates.hdf5",
    images_file_name=
        "clip_featuremap.hdf5",
    visible_object_map_file_name=
        "metadata.json",
    optimal_action_file_name=
        "optimal_action.json",
)


summary = {}

first_scene = True


for room, scenes in ROOM_SCENES.items():

    episodes = []

    target_counts = Counter()
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

        all_states = sorted(
            list(ctrl.images.keys())
        )

        candidate_states = [
            s
            for s in all_states
            if str(s)
            not in used_states[scene]
        ]

        if (
            len(candidate_states)
            <
            EPISODES_PER_SCENE
        ):
            raise RuntimeError(
                f"{scene}: only "
                f"{len(candidate_states)} "
                "unused states available"
            )

        # ------------------------------------------
        # Find available seen targets.
        # ------------------------------------------

        target_to_ids = defaultdict(set)

        probe_states = all_states[
            :min(10, len(all_states))
        ]

        for s in probe_states:

            meta = ctrl.metadata[s]

            for obj in meta.get(
                "objects",
                [],
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
            target_to_ids
        )

        if not valid_targets:
            raise RuntimeError(
                f"{scene}: no seen targets"
            )

        # ------------------------------------------
        # 100 new, unique start states.
        # ------------------------------------------

        selected_states = rng.sample(
            candidate_states,
            EPISODES_PER_SCENE,
        )

        # ------------------------------------------
        # Same predeclared approximately-balanced
        # target schedule as Batch 1.
        # ------------------------------------------

        schedule = []

        while len(
            schedule
        ) < EPISODES_PER_SCENE:

            block = valid_targets.copy()
            rng.shuffle(block)

            schedule.extend(block)

        schedule = schedule[
            :EPISODES_PER_SCENE
        ]

        rng.shuffle(schedule)

        for local_idx, (
            state,
            target,
        ) in enumerate(
            zip(
                selected_states,
                schedule,
            )
        ):

            episode_id = (
                f"aca2|"
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
                    sorted(
                        target_to_ids[target]
                    ),

                "aca_episode_id":
                    episode_id,

                "aca_batch":
                    2,
            }

            episodes.append(ep)

            scene_counts[scene] += 1
            target_counts[target] += 1

        print(
            f"{scene:<14} "
            f"all={len(all_states):>5} "
            f"unused={len(candidate_states):>5} "
            f"targets={len(valid_targets):>2} "
            f"episodes={scene_counts[scene]:>3}"
        )

    assert len(episodes) == 2000

    out = (
        OUT_DIR /
        f"{room}_aca_train2_22.pkl"
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
                    target_counts.items()
                )
            ),
    }

    print()
    print("Saved:", out)
    print("Episodes:", len(episodes))


total = sum(
    x["episodes"]
    for x in summary.values()
)

assert total == 8000


out_summary = (
    RESULT_DIR /
    "split_summary.json"
)

with open(
    out_summary,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "seed": SEED,
            "episodes_per_scene":
                EPISODES_PER_SCENE,
            "total": total,
            "rooms": summary,
        },
        f,
        indent=2,
    )


print()
print("=" * 100)
print("ACA BATCH-2 SPLIT")
print("=" * 100)

print("Total specs:", total)

for room in ROOM_SCENES:
    print(
        f"{room:<14}:",
        summary[room]["episodes"],
    )

print()
print("ACA BATCH-2 SPLIT: PASS")
