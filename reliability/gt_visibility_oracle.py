import json
from pathlib import Path

import torch


def canonical_state(s):
    parts = str(s).split("|")

    if len(parts) != 4:
        raise ValueError(
            "Invalid state: {}".format(s)
        )

    x = float(parts[0])
    z = float(parts[1])
    rot = float(parts[2])
    hor = float(parts[3])

    return (
        f"{x:.2f}|{z:.2f}|"
        f"{int(round(rot))}|"
        f"{int(round(hor))}"
    )


class GTVisibilityOracle:
    """
    Privileged diagnostic oracle.

    This oracle is ONLY for feasibility analysis.
    Simulator GT must never be used by the final method.

    Rule:
      detected non-goal category + GT not visible
          -> rho
      otherwise
          -> 1.0
    """

    def __init__(
        self,
        classes,
        cache_dir="reliability/gt_visibility_cache",
    ):
        self.classes = list(classes)
        self.cache_dir = Path(cache_dir)

        self._scene_cache = {}

    def _load_scene(self, scene):
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        p = (
            self.cache_dir
            / f"{scene}.json"
        )

        if not p.exists():
            raise FileNotFoundError(p)

        with open(
            p,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        self._scene_cache[scene] = data

        return data

    def build_weight(
        self,
        scene,
        state,
        target_name,
        det_scores,
        rho,
        device,
        dtype,
    ):
        cache = self._load_scene(scene)

        state_key = canonical_state(state)

        if state_key not in cache:
            raise KeyError(
                "State not in GT cache: "
                f"{scene} {state_key}"
            )

        gt_info = cache[state_key]

        visible = set(
            gt_info.get(
                "visible",
                []
            )
        )

        weights = torch.ones(
            (
                len(self.classes),
                1,
            ),
            device=device,
            dtype=dtype,
        )

        suppressed = []
        confirmed_visible = []

        for idx, category in enumerate(
            self.classes
        ):
            # Never suppress the goal row.
            if category == target_name:
                continue

            score = float(
                det_scores[idx]
            )

            # No detector evidence.
            if score <= 0:
                continue

            if category not in visible:
                weights[
                    idx, 0
                ] = float(rho)

                suppressed.append(
                    (
                        idx,
                        category,
                        score,
                    )
                )

            else:
                confirmed_visible.append(
                    (
                        idx,
                        category,
                        score,
                    )
                )

        stats = {
            "scene": scene,
            "state": state_key,
            "visible": sorted(visible),
            "suppressed": suppressed,
            "confirmed_visible": confirmed_visible,
        }

        return weights, stats
