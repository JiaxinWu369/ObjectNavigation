import math

import numpy as np
import torch
import torch.nn as nn

from reliability.rule_lcr import (
    RuleLocalContextReliability,
)


FEATURE_DIM = 66


class _StaticMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                FEATURE_DIM,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1,
            ),
        )


def build_feature_sequence(
    observations,
    goal_idx,
):
    """
    Reproduce EXACTLY the feature definition used by
    reliability/lcr_causal_features.py.

    observations contain only DISTINCT local viewpoints.
    """

    if len(observations) == 0:
        raise ValueError(
            "Empty Learned-LCR observation sequence"
        )

    goal_idx = int(goal_idx)

    first_attention = np.asarray(
        observations[0][
            "base_attention"
        ],
        dtype=np.float32,
    ).reshape(-1)

    if first_attention.shape != (22,):
        raise ValueError(
            "Expected 22-D first attention, got {}".format(
                first_attention.shape
            )
        )

    context_indices = [
        i
        for i in range(22)
        if i != goal_idx
    ]

    context_indices = sorted(
        context_indices,
        key=lambda i: (
            -float(
                first_attention[i]
            ),
            i,
        ),
    )

    features = []

    for t, obs in enumerate(
        observations
    ):

        attention = np.asarray(
            obs[
                "base_attention"
            ],
            dtype=np.float32,
        ).reshape(-1)

        detector = np.asarray(
            obs[
                "detector_scores"
            ],
            dtype=np.float32,
        ).reshape(-1)

        if attention.shape != (22,):
            raise ValueError(
                "Bad attention shape: {}".format(
                    attention.shape
                )
            )

        if detector.shape != (22,):
            raise ValueError(
                "Bad detector shape: {}".format(
                    detector.shape
                )
            )

        a_ctx = attention[
            context_indices
        ]

        d_ctx = detector[
            context_indices
        ]

        interaction = (
            a_ctx
            *
            d_ctx
        )

        goal_score = float(
            detector[
                goal_idx
            ]
        )

        semantic_kl = float(
            obs[
                "semantic_kl"
            ]
        )

        # Causal observed-view count.
        # No future sequence length is used.
        view_progress = min(
            float(t + 1)
            / 10.0,
            1.0,
        )

        x = np.concatenate(
            [
                a_ctx,
                d_ctx,
                interaction,
                np.asarray(
                    [
                        goal_score,
                        semantic_kl,
                        view_progress,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(
            np.float32
        )

        if x.shape != (
            FEATURE_DIM,
        ):
            raise RuntimeError(
                "Feature dimension {} != {}".format(
                    x.shape,
                    FEATURE_DIM,
                )
            )

        if not np.all(
            np.isfinite(x)
        ):
            raise RuntimeError(
                "Non-finite Learned-LCR feature"
            )

        features.append(
            x
        )

    return np.stack(
        features,
        axis=0,
    )


class LearnedLocalContextReliability:

    def __init__(
        self,
        object_distribution,
        num_classes,
        checkpoint_path,
        min_views=3,
        cell_size=1.0,
        position_bin=0.5,
        yaw_bin=90.0,
        horizon_bin=30.0,
        r_min=0.25,
        constant_q=None,
    ):
        self.object_distribution = (
            object_distribution
        )

        self.num_classes = int(
            num_classes
        )

        if self.num_classes != 22:
            raise ValueError(
                "Learned-LCR expects 22 classes, got {}".format(
                    self.num_classes
                )
            )

        self.min_views = int(
            min_views
        )

        self.cell_size = float(
            cell_size
        )

        self.position_bin = float(
            position_bin
        )

        self.yaw_bin = float(
            yaw_bin
        )

        self.horizon_bin = float(
            horizon_bin
        )

        self.r_min = float(
            r_min
        )

        self.constant_q = (
            None
            if constant_q is None
            else float(constant_q)
        )

        if (
            self.constant_q is not None
            and not (
                0.0
                <= self.constant_q
                <= 1.0
            )
        ):
            raise ValueError(
                "constant_q must be in [0,1], got {}".format(
                    self.constant_q
                )
            )

        # Reuse EXACT Rule-LCR static semantic prior.
        # Its local memory is not used here.
        self.profile_helper = (
            RuleLocalContextReliability(
                object_distribution=(
                    object_distribution
                ),
                num_classes=(
                    num_classes
                ),
            )
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        model_name = checkpoint.get(
            "model",
            None,
        )

        if (
            model_name is not None
            and model_name != "mlp"
        ):
            raise ValueError(
                "Expected MLP checkpoint, got {}".format(
                    model_name
                )
            )

        feature_dim = int(
            checkpoint.get(
                "feature_dim",
                FEATURE_DIM,
            )
        )

        if feature_dim != FEATURE_DIM:
            raise ValueError(
                "Checkpoint feature_dim {} != {}".format(
                    feature_dim,
                    FEATURE_DIM,
                )
            )

        self.model = (
            _StaticMLP()
        )

        self.model.load_state_dict(
            checkpoint[
                "state_dict"
            ],
            strict=True,
        )

        self.model.eval()

        mean = checkpoint[
            "feature_mean"
        ]

        std = checkpoint[
            "feature_std"
        ]

        if torch.is_tensor(mean):
            mean = (
                mean
                .detach()
                .cpu()
                .numpy()
            )

        if torch.is_tensor(std):
            std = (
                std
                .detach()
                .cpu()
                .numpy()
            )

        self.feature_mean = np.asarray(
            mean,
            dtype=np.float32,
        ).reshape(-1)

        self.feature_std = np.asarray(
            std,
            dtype=np.float32,
        ).reshape(-1)

        if (
            self.feature_mean.shape
            != (FEATURE_DIM,)
        ):
            raise ValueError(
                "Bad feature mean shape: {}".format(
                    self.feature_mean.shape
                )
            )

        if (
            self.feature_std.shape
            != (FEATURE_DIM,)
        ):
            raise ValueError(
                "Bad feature std shape: {}".format(
                    self.feature_std.shape
                )
            )

        self.feature_std[
            self.feature_std
            < 1e-6
        ] = 1.0

        self.regions = {}


    def reset(self):
        self.regions = {}


    @staticmethod
    def parse_state(
        state,
    ):
        if isinstance(
            state,
            str,
        ):
            parts = (
                state.split("|")
            )

            if len(parts) < 4:
                raise ValueError(
                    "Bad state: {}".format(
                        state
                    )
                )

            return (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )

        if isinstance(
            state,
            dict,
        ):
            rotation = state.get(
                "rotation",
                {}
            )

            return (
                float(
                    state["x"]
                ),
                float(
                    state["z"]
                ),
                float(
                    rotation.get(
                        "y",
                        0.0,
                    )
                ),
                float(
                    state.get(
                        "horizon",
                        0.0,
                    )
                ),
            )

        raise TypeError(
            "Unsupported state type: {}".format(
                type(state)
            )
        )


    def region_key(
        self,
        state,
    ):
        x, z, _, _ = (
            self.parse_state(
                state
            )
        )

        return (
            int(
                math.floor(
                    x
                    / self.cell_size
                )
            ),
            int(
                math.floor(
                    z
                    / self.cell_size
                )
            ),
        )


    def viewpoint_key(
        self,
        state,
    ):
        x, z, yaw, horizon = (
            self.parse_state(
                state
            )
        )

        px = int(
            math.floor(
                x
                / self.position_bin
            )
        )

        pz = int(
            math.floor(
                z
                / self.position_bin
            )
        )

        yaw_idx = int(
            math.floor(
                (
                    yaw % 360.0
                )
                / self.yaw_bin
            )
        )

        horizon_idx = int(
            round(
                horizon
                / self.horizon_bin
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
            "viewpoints":
                set(),

            "observations":
                [],

            "detector_max":
                np.zeros(
                    self.num_classes,
                    dtype=np.float32,
                ),

            "q":
                None,

            "reliability":
                1.0,
        }


    def _get_region(
        self,
        state,
    ):
        key = self.region_key(
            state
        )

        if key not in self.regions:
            self.regions[
                key
            ] = (
                self._new_region()
            )

        return (
            key,
            self.regions[key],
        )


    def _predict_q(
        self,
        observations,
        goal_idx,
    ):
        x = build_feature_sequence(
            observations=(
                observations
            ),
            goal_idx=(
                goal_idx
            ),
        )

        x = (
            x
            - self.feature_mean[
                None,
                :
            ]
        ) / (
            self.feature_std[
                None,
                :
            ]
        )

        # Static MLP = mean pooled multi-view evidence.
        pooled = (
            x.mean(
                axis=0
            )
            .astype(
                np.float32
            )
        )

        xt = (
            torch.from_numpy(
                pooled
            )
            .reshape(
                1,
                FEATURE_DIM,
            )
        )

        with torch.no_grad():

            logit = (
                self.model.net(
                    xt
                )
                .reshape(-1)[0]
            )

            q = float(
                torch.sigmoid(
                    logit
                ).item()
            )

        return float(
            np.clip(
                q,
                0.0,
                1.0,
            )
        )


    def build_weight(
        self,
        scene,
        state,
        goal_idx,
        detector_scores,
    ):
        """
        PRE-forward.

        Uses only reliability estimated from previously
        completed viewpoints. Current detector scores may
        update relation attribution because they are already
        available before AKGVP forward.

        The current viewpoint itself is NOT added to the
        predictor history here.
        """

        goal_idx = int(
            goal_idx
        )

        detector_scores = np.asarray(
            detector_scores,
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

        detector_scores = np.clip(
            detector_scores,
            0.0,
            1.0,
        )

        region_key, region = (
            self._get_region(
                state
            )
        )

        viewpoint_key = (
            self.viewpoint_key(
                state
            )
        )

        # Detector is available pre-forward, therefore it is
        # legal to use it for relation attribution.
        region[
            "detector_max"
        ] = np.maximum(
            region[
                "detector_max"
            ],
            detector_scores,
        )

        num_views = len(
            region[
                "viewpoints"
            ]
        )

        reliability = float(
            region[
                "reliability"
            ]
        )

        # Before three completed distinct viewpoints,
        # Learned-LCR is exactly Base.
        if num_views < (
            self.min_views
        ):
            reliability = 1.0

        profile = (
            self.profile_helper
            .static_goal_profile(
                scene=scene,
                goal_idx=goal_idx,
            )
        )

        contribution = (
            np.asarray(
                profile,
                dtype=np.float32,
            )
            *
            region[
                "detector_max"
            ]
        )

        contribution[
            goal_idx
        ] = 0.0

        # -------------------------------------------------
        # Selective-LCR activation.
        #
        # High conflict means:
        #   (1) the target remains weakly observed locally,
        #   (2) target-related contextual evidence is strong.
        #
        # conflict_tau=None reproduces the original
        # Always-LCR activation rule.
        # -------------------------------------------------
        conflict_tau = getattr(
            self,
            "conflict_tau",
            None,
        )

        target_evidence = float(
            region[
                "detector_max"
            ][goal_idx]
        )

        context_prior = np.asarray(
            profile,
            dtype=np.float32,
        ).copy()

        context_prior[
            goal_idx
        ] = 0.0

        prior_peak = float(
            context_prior.max()
        )

        if prior_peak > 1e-8:
            context_support = float(
                contribution.max()
                / prior_peak
            )
        else:
            context_support = 0.0

        context_support = float(
            np.clip(
                context_support,
                0.0,
                1.0,
            )
        )

        conflict_score = float(
            (
                1.0
                - target_evidence
            )
            * context_support
        )

        activated = (
            num_views
            >= self.min_views
        )

        if (
            activated
            and conflict_tau is not None
        ):
            activated = (
                conflict_score
                >= float(
                    conflict_tau
                )
            )

        gate = np.ones(
            self.num_classes,
            dtype=np.float32,
        )

        if activated:

            max_contribution = float(
                contribution.max()
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

        # Never suppress target itself.
        gate[
            goal_idx
        ] = 1.0

        stats = {
            "activated":
                bool(
                    activated
                ),

            "conflict_score":
                float(
                    conflict_score
                ),

            "conflict_tau":
                (
                    None
                    if conflict_tau is None
                    else float(
                        conflict_tau
                    )
                ),

            "target_evidence":
                float(
                    target_evidence
                ),

            "context_support":
                float(
                    context_support
                ),

            "region_key":
                tuple(
                    int(v)
                    for v in
                    region_key
                ),

            "viewpoint_key":
                tuple(
                    int(v)
                    for v in
                    viewpoint_key
                ),

            "new_viewpoint":
                bool(
                    viewpoint_key
                    not in
                    region[
                        "viewpoints"
                    ]
                ),

            "num_views_before":
                int(
                    num_views
                ),

            "num_views":
                int(
                    num_views
                ),

            "q":
                (
                    None
                    if region["q"] is None
                    else float(
                        region["q"]
                    )
                ),

            "R_before":
                float(
                    reliability
                ),

            "R_after":
                float(
                    reliability
                ),

            "region_reliability":
                float(
                    reliability
                ),

            "gate_min":
                float(
                    gate.min()
                ),

            "goal_gate":
                float(
                    gate[
                        goal_idx
                    ]
                ),
        }

        return (
            gate,
            stats,
        )


    def observe_after_forward(
        self,
        state,
        goal_idx,
        detector_scores,
        base_attention,
        semantic_kl,
    ):
        """
        POST-forward.

        Adds current viewpoint only after the current AKGVP
        action distribution has already been computed.

        Any updated q/R therefore becomes active from the
        NEXT navigation step.
        """

        goal_idx = int(
            goal_idx
        )

        detector_scores = np.asarray(
            detector_scores,
            dtype=np.float32,
        ).reshape(-1)

        base_attention = np.asarray(
            base_attention,
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

        if base_attention.shape != (
            self.num_classes,
        ):
            raise ValueError(
                "Bad base attention shape: {}".format(
                    base_attention.shape
                )
            )

        detector_scores = np.clip(
            detector_scores,
            0.0,
            1.0,
        )

        region_key, region = (
            self._get_region(
                state
            )
        )

        viewpoint_key = (
            self.viewpoint_key(
                state
            )
        )

        region[
            "detector_max"
        ] = np.maximum(
            region[
                "detector_max"
            ],
            detector_scores,
        )

        num_views_before = len(
            region[
                "viewpoints"
            ]
        )

        reliability_before = float(
            region[
                "reliability"
            ]
        )

        is_new = (
            viewpoint_key
            not in
            region[
                "viewpoints"
            ]
        )

        if is_new:

            region[
                "viewpoints"
            ].add(
                viewpoint_key
            )

            region[
                "observations"
            ].append(
                {
                    "detector_scores":
                        detector_scores.copy(),

                    "base_attention":
                        base_attention.copy(),

                    "semantic_kl":
                        float(
                            semantic_kl
                        ),
                }
            )

            num_views = len(
                region[
                    "viewpoints"
                ]
            )

            if num_views >= (
                self.min_views
            ):

                if self.constant_q is None:

                    q = (
                        self._predict_q(
                            observations=(
                                region[
                                    "observations"
                                ]
                            ),
                            goal_idx=(
                                goal_idx
                            ),
                        )
                    )

                else:

                    q = float(
                        self.constant_q
                    )

                reliability = (
                    self.r_min
                    +
                    (
                        1.0
                        - self.r_min
                    )
                    * q
                )

                reliability = float(
                    np.clip(
                        reliability,
                        self.r_min,
                        1.0,
                    )
                )

                region["q"] = float(
                    q
                )

                region[
                    "reliability"
                ] = reliability

        num_views = len(
            region[
                "viewpoints"
            ]
        )

        reliability_after = float(
            region[
                "reliability"
            ]
        )

        if num_views < (
            self.min_views
        ):
            reliability_after = 1.0
            region[
                "reliability"
            ] = 1.0

        stats = {
            "region_key":
                tuple(
                    int(v)
                    for v in
                    region_key
                ),

            "viewpoint_key":
                tuple(
                    int(v)
                    for v in
                    viewpoint_key
                ),

            "new_viewpoint":
                bool(
                    is_new
                ),

            "num_views_before":
                int(
                    num_views_before
                ),

            "num_views":
                int(
                    num_views
                ),

            "q":
                (
                    None
                    if region["q"] is None
                    else float(
                        region["q"]
                    )
                ),

            "R_before":
                float(
                    reliability_before
                ),

            "R_after":
                float(
                    reliability_after
                ),

            "region_reliability":
                float(
                    reliability_after
                ),
        }

        return stats
