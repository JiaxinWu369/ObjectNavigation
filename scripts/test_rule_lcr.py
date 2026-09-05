import numpy as np
import torch

from reliability.rule_lcr import (
    RuleLocalContextReliability,
)


class FakeObjectDistribution:

    def __init__(self):

        alpha = np.ones(
            (4, 22, 22),
            dtype=np.float32,
        )

        # Make category 0 a stronger contextual
        # supporter for goal category 3.
        alpha[
            :,
            0,
            3,
        ] = 20.0

        self.init_prior_alpha = (
            torch.tensor(alpha)
        )

    def get_scene_id(
        self,
        scene,
    ):
        return 0


od = FakeObjectDistribution()

lcr = RuleLocalContextReliability(
    object_distribution=od,
    num_classes=22,
)

goal = 3

det = np.zeros(
    22,
    dtype=np.float32,
)

det[0] = 1.0
det[goal] = 0.0


# -------------------------------------------------
# 1. First view: not enough inspection yet.
# -------------------------------------------------
w1, s1 = lcr.build_weight(
    scene="FloorPlan1",
    state="0.10|0.10|0|0",
    goal_idx=goal,
    detector_scores=det,
)

assert s1["num_views"] == 1
assert s1["region_reliability"] == 1.0
assert np.allclose(
    w1,
    np.ones(22),
)


# -------------------------------------------------
# 2. Repeated same viewpoint:
# negative evidence must NOT increase.
# -------------------------------------------------
w1b, s1b = lcr.build_weight(
    scene="FloorPlan1",
    state="0.10|0.10|0|0",
    goal_idx=goal,
    detector_scores=det,
)

assert s1b["new_viewpoint"] is False

assert np.isclose(
    s1[
        "negative_evidence"
    ],
    s1b[
        "negative_evidence"
    ],
)


# -------------------------------------------------
# 3. Three distinct viewpoints:
# calibration becomes active.
# -------------------------------------------------
_, s2 = lcr.build_weight(
    scene="FloorPlan1",
    state="0.10|0.10|90|0",
    goal_idx=goal,
    detector_scores=det,
)

w3, s3 = lcr.build_weight(
    scene="FloorPlan1",
    state="0.10|0.10|180|0",
    goal_idx=goal,
    detector_scores=det,
)

assert s3["num_views"] == 3

assert (
    s3["region_reliability"]
    < 1.0
)

assert (
    s3["negative_evidence"]
    >
    s2["negative_evidence"]
)

assert w3[0] < 1.0

assert np.isclose(
    w3[goal],
    1.0,
)


# -------------------------------------------------
# 4. New region starts fresh.
# -------------------------------------------------
w_new, s_new = lcr.build_weight(
    scene="FloorPlan1",
    state="1.10|0.10|0|0",
    goal_idx=goal,
    detector_scores=det,
)

assert s_new["num_views"] == 1

assert np.isclose(
    s_new[
        "region_reliability"
    ],
    1.0,
)

assert np.allclose(
    w_new,
    np.ones(22),
)


# -------------------------------------------------
# 5. Return to original region:
# old reliability memory must still exist.
# -------------------------------------------------
_, s_return = lcr.build_weight(
    scene="FloorPlan1",
    state="0.10|0.10|270|0",
    goal_idx=goal,
    detector_scores=det,
)

assert s_return["num_views"] == 4

assert (
    s_return[
        "region_reliability"
    ]
    <
    s_new[
        "region_reliability"
    ]
)


# -------------------------------------------------
# 6. Reset clears episode memory.
# -------------------------------------------------
lcr.reset()

_, s_reset = lcr.build_weight(
    scene="FloorPlan1",
    state="0.10|0.10|0|0",
    goal_idx=goal,
    detector_scores=det,
)

assert s_reset["num_views"] == 1
assert np.isclose(
    s_reset[
        "region_reliability"
    ],
    1.0,
)


print(
    "Repeated-view evidence : PASS"
)

print(
    "Multi-view decay       : PASS"
)

print(
    "Region independence    : PASS"
)

print(
    "Goal protection        : PASS"
)

print(
    "Episode reset          : PASS"
)

print()
print(
    "RULE-LCR UNIT TESTS: PASS"
)
