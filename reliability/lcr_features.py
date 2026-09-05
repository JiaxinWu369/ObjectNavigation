import numpy as np


FEATURE_DIM = 66
FEATURE_DIM_NO_GOAL = 65


def region_sequence_features(row, drop_goal_score=False):
    seq = row["sequence"]

    if len(seq) < 1:
        raise ValueError("Empty LCR sequence")

    goal_idx = int(row["goal_index"])

    first_attention = np.asarray(
        seq[0]["base_attention"],
        dtype=np.float32,
    )

    if first_attention.shape != (22,):
        raise ValueError(
            "Expected 22-D base_attention"
        )

    # --------------------------------------------------
    # Category-agnostic context ordering.
    #
    # Exclude the goal category and rank the remaining
    # 21 semantic relations by the first observation's
    # target-conditioned AKGVP attention.
    #
    # The same category order is retained for the whole
    # local sequence.
    # --------------------------------------------------
    context_indices = [
        i
        for i in range(22)
        if i != goal_idx
    ]

    context_indices = sorted(
        context_indices,
        key=lambda i: (
            -float(first_attention[i]),
            i,
        ),
    )

    features = []

    T = len(seq)

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
            a_ctx * d_ctx
        )

        goal_score = float(
            detector[goal_idx]
        )

        semantic_kl = float(
            obs["semantic_kl"]
        )

        progress = float(
            t + 1
        ) / float(T)

        tail = (
            [
                semantic_kl,
                progress,
            ]
            if drop_goal_score
            else [
                goal_score,
                semantic_kl,
                progress,
            ]
        )

        x = np.concatenate(
            [
                a_ctx,
                d_ctx,
                interaction,
                np.asarray(
                    tail,
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(
            np.float32
        )

        expected_dim = (
            FEATURE_DIM_NO_GOAL
            if drop_goal_score
            else FEATURE_DIM
        )

        if x.shape != (
            expected_dim,
        ):
            raise RuntimeError(
                "Feature shape {} != {}".format(
                    x.shape,
                    expected_dim,
                )
            )

        if not np.all(
            np.isfinite(x)
        ):
            raise RuntimeError(
                "Non-finite LCR features"
            )

        features.append(x)

    return np.stack(
        features,
        axis=0,
    )
