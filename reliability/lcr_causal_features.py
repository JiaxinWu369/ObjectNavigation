import os
import numpy as np


FEATURE_DIM = 66


def drop_goal_score():
    return (
        os.environ.get(
            "AKGVP_LCR_DROP_GOAL_SCORE",
            "0",
        )
        == "1"
    )


def region_sequence_features(
    row,
):
    """
    All observations in row['sequence'] are already observed
    historical viewpoints.

    The predicted reliability is applied to the NEXT
    navigation step, so base_attention and semantic_kl here
    are causal historical information.
    """

    seq = row["sequence"]

    if len(seq) < 1:
        raise ValueError(
            "Empty LCR sequence"
        )

    goal_idx = int(
        row["goal_index"]
    )

    # --------------------------------------------------
    # Fixed local category ordering.
    #
    # The first observed viewpoint establishes a
    # target-conditioned semantic ordering.
    # This is known before predictions are applied
    # from later navigation steps.
    # --------------------------------------------------
    first_attention = np.asarray(
        seq[0]["base_attention"],
        dtype=np.float32,
    )

    if first_attention.shape != (22,):
        raise ValueError(
            "Expected 22-D attention"
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

    remove_goal = (
        drop_goal_score()
    )

    for t, obs in enumerate(seq):

        attention = np.asarray(
            obs["base_attention"],
            dtype=np.float32,
        )

        detector = np.asarray(
            obs["detector_scores"],
            dtype=np.float32,
        )

        if attention.shape != (22,):
            raise ValueError(
                "Bad attention shape"
            )

        if detector.shape != (22,):
            raise ValueError(
                "Bad detector shape"
            )

        a_ctx = attention[
            context_indices
        ]

        d_ctx = detector[
            context_indices
        ]

        interaction = (
            a_ctx * d_ctx
        )

        goal_score = float(
            detector[
                goal_idx
            ]
        )

        # No-goal ablation:
        # preserve dimension but remove information.
        if remove_goal:
            goal_score = 0.0

        semantic_kl = float(
            obs["semantic_kl"]
        )

        # IMPORTANT:
        # no future total-length T.
        #
        # Number of already observed distinct viewpoints.
        view_progress = min(
            float(t + 1) / 10.0,
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
                "Feature dimension "
                "{} != {}".format(
                    x.shape,
                    FEATURE_DIM,
                )
            )

        if not np.all(
            np.isfinite(x)
        ):
            raise RuntimeError(
                "Non-finite LCR feature"
            )

        features.append(x)

    return np.stack(
        features,
        axis=0,
    )
