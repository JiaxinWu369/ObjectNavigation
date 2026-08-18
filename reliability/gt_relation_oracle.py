import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def object_type(object_id):
    return object_id.split("|")[0]


def object_xz(object_id):
    """
    Extract ground-plane (x, z) from an AI2-THOR
    physical object id.

    Common forms include:

        Type|x|y|z
        Type|x|y|z|ChildName

    Do not assume that the final token is always z.
    Instead, locate the first three consecutive numeric
    fields after the object type.
    """
    parts = str(object_id).split("|")

    if len(parts) < 4:
        raise ValueError(
            "Invalid AI2-THOR object id: "
            + str(object_id)
        )

    for i in range(1, len(parts) - 2):
        try:
            x = float(parts[i])
            _y = float(parts[i + 1])
            z = float(parts[i + 2])

            return x, z

        except ValueError:
            continue

    raise ValueError(
        "Could not extract x/y/z from object id: "
        + str(object_id)
    )


def ground_distance(a, b):
    ax, az = object_xz(a)
    bx, bz = object_xz(b)

    return math.sqrt(
        (ax - bx) ** 2
        +
        (az - bz) ** 2
    )


class GTRelationOracle:
    """
    Privileged diagnostic oracle.

    It does NOT represent the final inference-time method.

    For a currently detected non-goal category:

      - obtain GT-visible physical instances;
      - compare those instances against the episode's
        true target instance(s);
      - suppress the semantic category only when all
        currently visible instances are spatially far
        from every target instance.

    Detector-only categories without a GT-visible
    instance are left unchanged, so detector false
    positives are not conflated with semantic-relation
    reliability.
    """

    def __init__(
        self,
        data_dir,
        classes,
        radius=1.5,
    ):
        self.data_dir = Path(data_dir)
        self.classes = list(classes)
        self.radius = float(radius)

        self._scene_cache = {}

    def _load_scene(self, scene):
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        path = (
            self.data_dir
            / scene
            / "visible_object_map_1.5.json"
        )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            object_to_states = json.load(f)

        state_to_objects = defaultdict(list)

        for object_id, states in object_to_states.items():
            for state in states:
                state_to_objects[state].append(
                    object_id
                )

        state_to_objects = dict(state_to_objects)

        self._scene_cache[scene] = (
            state_to_objects
        )

        return state_to_objects

    def build_weight(
        self,
        scene,
        state,
        target_ids,
        target_name,
        det_scores,
        rho,
        device,
        dtype,
    ):
        state_to_objects = self._load_scene(
            scene
        )

        visible_ids = state_to_objects.get(
            state,
            [],
        )

        visible_by_class = defaultdict(list)

        for object_id in visible_ids:
            visible_by_class[
                object_type(object_id)
            ].append(object_id)

        target_ids = [
            x
            for x in target_ids
            if isinstance(x, str)
        ]

        weights = torch.ones(
            (
                len(self.classes),
                1,
            ),
            device=device,
            dtype=dtype,
        )

        suppressed = []
        reliable = []
        no_gt_visible = []

        for idx, category in enumerate(
            self.classes
        ):
            if category == target_name:
                continue

            score = float(
                det_scores[idx]
            )

            # Not currently detected:
            # semantic row is irrelevant to this
            # diagnostic decision.
            if score <= 0:
                continue

            candidate_ids = (
                visible_by_class.get(
                    category,
                    [],
                )
            )

            # Important:
            # do NOT suppress this case.
            #
            # Otherwise we would also be testing
            # detector false-positive correction.
            if not candidate_ids:
                no_gt_visible.append(
                    category
                )
                continue

            min_dist = min(
                ground_distance(
                    candidate_id,
                    target_id,
                )
                for candidate_id
                in candidate_ids
                for target_id
                in target_ids
            )

            if min_dist > self.radius:
                weights[idx, 0] = float(rho)

                suppressed.append(
                    (
                        idx,
                        category,
                        min_dist,
                    )
                )
            else:
                reliable.append(
                    (
                        idx,
                        category,
                        min_dist,
                    )
                )

        stats = {
            "state": state,
            "visible_count": len(
                visible_ids
            ),
            "suppressed": suppressed,
            "reliable": reliable,
            "no_gt_visible": no_gt_visible,
        }

        return weights, stats
