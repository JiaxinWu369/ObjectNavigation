import numpy as np

from .local_region_memory import (
    LocalRegionMemory,
)


class RuleLocalContextReliability:

    def __init__(
        self,
        object_distribution,
        num_classes=22,
    ):
        self.object_distribution = (
            object_distribution
        )

        self.num_classes = int(
            num_classes
        )

        self.memory = LocalRegionMemory(
            num_classes=num_classes,
            cell_size=1.0,
            position_bin=0.5,
            yaw_bin=90.0,
            horizon_bin=30.0,
            min_views=3,
            r_min=0.25,
            beta=1.0,
        )

        self._profile_cache = {}

    def reset(self):
        self.memory.reset()

    def static_goal_profile(
        self,
        scene,
        goal_idx,
    ):
        scene_id = int(
            self.object_distribution
            .get_scene_id(
                scene
            )
        )

        key = (
            scene_id,
            int(goal_idx),
        )

        if key in self._profile_cache:
            return self._profile_cache[
                key
            ]

        alpha = (
            self.object_distribution
            .init_prior_alpha[
                scene_id
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        row_sum = alpha.sum(
            axis=1,
            keepdims=True,
        )

        row_sum[
            row_sum <= 0
        ] = 1.0

        prob = (
            alpha
            / row_sum
        )

        # AKGVP uses:
        #
        #     prob_matrix @ target_one_hot
        #
        # therefore the target-conditioned
        # relation vector is P[:, goal].
        profile = prob[
            :,
            int(goal_idx)
        ].copy()

        # Goal self-relation is not contextual
        # evidence.
        profile[
            int(goal_idx)
        ] = 0.0

        total = float(
            profile.sum()
        )

        if total <= 0:
            raise RuntimeError(
                "Empty static context profile: "
                "scene={} goal_idx={}".format(
                    scene,
                    goal_idx,
                )
            )

        profile = (
            profile
            / total
        ).astype(
            np.float32
        )

        self._profile_cache[
            key
        ] = profile

        return profile

    def build_weight(
        self,
        scene,
        state,
        goal_idx,
        detector_scores,
    ):
        profile = self.static_goal_profile(
            scene=scene,
            goal_idx=goal_idx,
        )

        return self.memory.update(
            state=state,
            detector_scores=detector_scores,
            goal_idx=goal_idx,
            context_profile=profile,
        )
