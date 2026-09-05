import torch
import torch.nn as nn


FEATURE_NAMES = [
    "prior_entropy",
    "prior_top3_mass",

    "prior_support_score_0_10",
    "prior_support_mass_0_10",

    "goal_score_max_0_10",
    "goal_seen_count_0_10",

    "kl_mean_recent5",
    "ctx_att_l1_recent5",

    "prior_entropy_x_support_score",
    "prior_top3_x_support_mass",
]


class AdaptiveContextSelector(nn.Module):
    """
    ACA-v1

    Binary decision at the matched state s_10:

        1 -> Restore context now (Early10)
        0 -> Delay restoration until t=20 (Early20)

    This module deliberately contains only a linear
    decision layer. Non-linearity is introduced through
    the two predeclared interaction features.
    """

    def __init__(self):
        super().__init__()

        self.n_features = len(
            FEATURE_NAMES
        )

        self.linear = nn.Linear(
            self.n_features,
            1,
        )

        self.register_buffer(
            "feature_mean",
            torch.zeros(
                self.n_features
            ),
        )

        self.register_buffer(
            "feature_std",
            torch.ones(
                self.n_features
            ),
        )

        self.register_buffer(
            "restore_threshold",
            torch.tensor(0.5),
        )


    @torch.no_grad()
    def set_normalizer(
        self,
        mean,
        std,
    ):
        mean = torch.as_tensor(
            mean,
            dtype=torch.float32,
        )

        std = torch.as_tensor(
            std,
            dtype=torch.float32,
        )

        if mean.numel() != self.n_features:
            raise ValueError(
                "mean dimension mismatch"
            )

        if std.numel() != self.n_features:
            raise ValueError(
                "std dimension mismatch"
            )

        std = torch.clamp(
            std,
            min=1e-6,
        )

        self.feature_mean.copy_(
            mean
        )

        self.feature_std.copy_(
            std
        )


    @torch.no_grad()
    def set_threshold(
        self,
        threshold,
    ):
        self.restore_threshold.fill_(
            float(threshold)
        )


    def normalize(self, x):
        return (
            x - self.feature_mean
        ) / self.feature_std


    def forward(self, x):
        x = self.normalize(x)

        return self.linear(
            x
        ).squeeze(-1)


    def probability(self, x):
        return torch.sigmoid(
            self.forward(x)
        )


    def decision(self, x):
        """
        True  -> Restore at t=10.
        False -> Delay until t=20.
        """
        return (
            self.probability(x)
            >=
            self.restore_threshold
        )
