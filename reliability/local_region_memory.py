import math
import numpy as np


class LocalRegionMemory:
    """
    Lightweight local semantic reliability memory.

    No simulator GT is used here.
    """

    def __init__(
        self,
        num_classes=22,
        cell_size=1.0,
        position_bin=0.5,
        yaw_bin=90.0,
        horizon_bin=30.0,
        min_views=3,
        r_min=0.25,
        beta=1.0,
    ):
        self.num_classes = int(num_classes)
        self.cell_size = float(cell_size)
        self.position_bin = float(position_bin)
        self.yaw_bin = float(yaw_bin)
        self.horizon_bin = float(horizon_bin)
        self.min_views = int(min_views)
        self.r_min = float(r_min)
        self.beta = float(beta)

        self.reset()

    def reset(self):
        self.regions = {}

    @staticmethod
    def parse_state(state):
        """
        AKGVP / AI2-THOR state format used by the current code:
            x|z|rotation|horizon
        Example:
            1.25|-0.50|45|30
        """
        text = str(state)
        parts = text.split("|")

        if len(parts) < 4:
            raise ValueError(
                "Unexpected state format: {}".format(text)
            )

        x = float(parts[0])
        z = float(parts[1])
        yaw = float(parts[2])
        horizon = float(parts[3])

        return x, z, yaw, horizon

    def region_key(self, state):
        x, z, _, _ = self.parse_state(state)

        return (
            int(math.floor(x / self.cell_size)),
            int(math.floor(z / self.cell_size)),
        )

    def viewpoint_key(self, state):
        x, z, yaw, horizon = self.parse_state(state)

        px = int(
            math.floor(x / self.position_bin)
        )

        pz = int(
            math.floor(z / self.position_bin)
        )

        yaw_idx = int(
            math.floor(
                (yaw % 360.0) / self.yaw_bin
            )
        )

        horizon_idx = int(
            round(
                horizon / self.horizon_bin
            )
        )

        return (
            px,
            pz,
            yaw_idx,
            horizon_idx,
        )

    def _new_region(self):
        return {
            "viewpoints": set(),
            "negative_evidence": 0.0,
            "detector_max": np.zeros(
                self.num_classes,
                dtype=np.float32,
            ),
            "reliability": 1.0,
        }

    def update(
        self,
        state,
        detector_scores,
        goal_idx,
        context_profile,
    ):
        detector_scores = np.asarray(
            detector_scores,
            dtype=np.float32,
        ).reshape(-1)

        context_profile = np.asarray(
            context_profile,
            dtype=np.float32,
        ).reshape(-1)

        if detector_scores.shape != (
            self.num_classes,
        ):
            raise ValueError(
                "Bad detector shape: {}".format(
                    detector_scores.shape
                )
            )

        if context_profile.shape != (
            self.num_classes,
        ):
            raise ValueError(
                "Bad profile shape: {}".format(
                    context_profile.shape
                )
            )

        detector_scores = np.clip(
            detector_scores,
            0.0,
            1.0,
        )

        region_key = self.region_key(
            state
        )

        viewpoint_key = self.viewpoint_key(
            state
        )

        if region_key not in self.regions:
            self.regions[
                region_key
            ] = self._new_region()

        region = self.regions[
            region_key
        ]

        # Keep strongest observed category evidence
        # even if the current viewpoint was already seen.
        region["detector_max"] = np.maximum(
            region["detector_max"],
            detector_scores,
        )

        is_new_viewpoint = (
            viewpoint_key
            not in region["viewpoints"]
        )

        semantic_support = float(
            np.dot(
                context_profile,
                detector_scores,
            )
        )

        goal_score = float(
            detector_scores[
                int(goal_idx)
            ]
        )

        step_negative = 0.0

        if is_new_viewpoint:
            region["viewpoints"].add(
                viewpoint_key
            )

            step_negative = (
                semantic_support
                * (1.0 - goal_score)
            )

            region[
                "negative_evidence"
            ] += float(
                step_negative
            )

        num_views = len(
            region["viewpoints"]
        )

        if num_views < self.min_views:
            reliability = 1.0
        else:
            reliability = (
                self.r_min
                +
                (1.0 - self.r_min)
                * math.exp(
                    -self.beta
                    * region[
                        "negative_evidence"
                    ]
                )
            )

        reliability = float(
            np.clip(
                reliability,
                self.r_min,
                1.0,
            )
        )

        region[
            "reliability"
        ] = reliability

        # -----------------------------------------
        # Relation attribution.
        #
        # Which semantic categories actually
        # supported this local hypothesis?
        # -----------------------------------------
        contribution = (
            context_profile
            * region["detector_max"]
        )

        contribution[
            int(goal_idx)
        ] = 0.0

        gate = np.ones(
            self.num_classes,
            dtype=np.float32,
        )

        if num_views >= self.min_views:
            max_contribution = float(
                np.max(
                    contribution
                )
            )

            if max_contribution > 1e-8:
                attribution = (
                    contribution
                    / max_contribution
                )

                gate = (
                    1.0
                    -
                    (
                        1.0
                        - reliability
                    )
                    * attribution
                ).astype(
                    np.float32
                )

        # Goal category must never be suppressed.
        gate[
            int(goal_idx)
        ] = 1.0

        stats = {
            "region_key":
                region_key,

            "viewpoint_key":
                viewpoint_key,

            "new_viewpoint":
                bool(
                    is_new_viewpoint
                ),

            "num_views":
                int(
                    num_views
                ),

            "semantic_support":
                float(
                    semantic_support
                ),

            "goal_score":
                float(
                    goal_score
                ),

            "step_negative":
                float(
                    step_negative
                ),

            "negative_evidence":
                float(
                    region[
                        "negative_evidence"
                    ]
                ),

            "region_reliability":
                float(
                    reliability
                ),

            "gate_min":
                float(
                    gate.min()
                ),

            "gate_mean":
                float(
                    gate.mean()
                ),

            "goal_gate":
                float(
                    gate[
                        int(goal_idx)
                    ]
                ),
        }

        return gate, stats
