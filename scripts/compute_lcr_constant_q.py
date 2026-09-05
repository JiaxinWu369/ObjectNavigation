import json
from pathlib import Path

import numpy as np
import torch

from reliability.learned_lcr import (
    FEATURE_DIM,
    _StaticMLP,
    build_feature_sequence,
)

CKPT = Path(
    "results/lcr_frozen/"
    "learned_lcr_mlp.pt"
)

DATA = Path(
    "results/lcr_causal_dataset/"
    "prefixes_labeled.jsonl"
)

OUT = Path(
    "results/final_protocol/"
    "constant_q_stats.json"
)


def to_numpy(x):
    if torch.is_tensor(x):
        return (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


ckpt = torch.load(
    CKPT,
    map_location="cpu",
)

model = _StaticMLP()

model.load_state_dict(
    ckpt["state_dict"],
    strict=True,
)

model.eval()

mean = to_numpy(
    ckpt["feature_mean"]
).astype(
    np.float32
).reshape(-1)

std = to_numpy(
    ckpt["feature_std"]
).astype(
    np.float32
).reshape(-1)

assert mean.shape == (
    FEATURE_DIM,
)

assert std.shape == (
    FEATURE_DIM,
)

std[
    std < 1e-6
] = 1.0

qs = []

with DATA.open() as f:

    for line in f:

        row = json.loads(
            line
        )

        if (
            row.get("data_split")
            != "train"
        ):
            continue

        sequence = row[
            "sequence"
        ]

        goal_idx = int(
            row["goal_index"]
        )

        x = (
            build_feature_sequence(
                observations=sequence,
                goal_idx=goal_idx,
            )
        )

        x = (
            x
            - mean[None, :]
        ) / std[None, :]

        pooled = (
            x.mean(axis=0)
            .astype(np.float32)
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
                model.net(xt)
                .reshape(-1)[0]
            )

            q = float(
                torch.sigmoid(
                    logit
                ).item()
            )

        qs.append(q)


if not qs:
    raise RuntimeError(
        "No training prefixes found"
    )

arr = np.asarray(
    qs,
    dtype=np.float64,
)

mean_q = float(
    arr.mean()
)

r_min = 0.25

mean_r = (
    r_min
    +
    (1.0 - r_min)
    * mean_q
)

result = {
    "num_train_prefixes":
        int(len(arr)),
    "mean_q":
        mean_q,
    "mean_R":
        float(mean_r),
    "std_q":
        float(arr.std()),
    "median_q":
        float(
            np.median(arr)
        ),
    "min_q":
        float(arr.min()),
    "max_q":
        float(arr.max()),
}

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUT.write_text(
    json.dumps(
        result,
        indent=2,
    )
)

print(
    json.dumps(
        result,
        indent=2,
    )
)

print(
    "MEAN_Q={:.10f}".format(
        mean_q
    )
)
