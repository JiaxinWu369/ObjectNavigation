import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

DATA = Path(
    "results/learned_guard_protocol/"
    "guard_train_samples.jsonl"
)

OUT = Path(
    "results/learned_guard_protocol/"
    "linear_guard_final.pt"
)


def scalar(x, key):
    v = x.get(key, None)

    if v is None:
        return 0.0

    return float(v)


def feature(x):

    pb = np.asarray(
        x["prob_base"],
        dtype=np.float32,
    )

    pl = np.asarray(
        x["prob_lcr"],
        dtype=np.float32,
    )

    dp = pl - pb

    extra = np.asarray(
        [
            scalar(x, "margin_base"),
            scalar(x, "margin_lcr"),

            scalar(x, "entropy_base"),
            scalar(x, "entropy_lcr"),

            scalar(x, "target_evidence"),
            scalar(x, "context_support"),
            scalar(x, "conflict_score"),
            scalar(x, "num_views"),

            scalar(x, "R_before"),
            scalar(x, "gate_min"),
        ],
        dtype=np.float32,
    )

    f = np.concatenate(
        [
            pb,
            pl,
            dp,
            extra,
        ]
    )

    assert f.shape == (28,)

    return f


rows = []

for line in DATA.read_text(
    encoding="utf-8"
).splitlines():

    if not line.strip():
        continue

    x = json.loads(line)

    if x["label"] is not None:
        rows.append(x)


X = np.stack(
    [feature(x) for x in rows]
)

y = np.asarray(
    [x["label"] for x in rows],
    dtype=np.float32,
)


mean = X.mean(
    axis=0,
    keepdims=True,
)

std = X.std(
    axis=0,
    keepdims=True,
)

std[
    std < 1e-6
] = 1.0

Xn = (
    X - mean
) / std


Xn = torch.tensor(
    Xn,
    dtype=torch.float32,
)

yt = torch.tensor(
    y,
    dtype=torch.float32,
).reshape(-1, 1)


model = nn.Linear(
    28,
    1,
)


npos = float(
    yt.sum().item()
)

nneg = float(
    len(yt) - npos
)

pos_weight = torch.tensor(
    [
        nneg
        / max(
            npos,
            1.0,
        )
    ],
    dtype=torch.float32,
)


loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

opt = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)


for epoch in range(
    1,
    301,
):

    model.train()

    logits = model(Xn)

    loss = loss_fn(
        logits,
        yt,
    )

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (
        epoch == 1
        or epoch % 50 == 0
    ):
        with torch.no_grad():

            prob = torch.sigmoid(
                model(Xn)
            )

            pred = (
                prob >= 0.5
            ).float()

            acc = float(
                (
                    pred == yt
                )
                .float()
                .mean()
                .item()
            )

        print(
            "epoch",
            epoch,
            "loss",
            round(
                float(loss.item()),
                6,
            ),
            "train_acc",
            round(
                acc,
                4,
            ),
        )


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    {
        "model": "LinearGuard",
        "state_dict":
            model.state_dict(),

        "feature_dim": 28,

        "feature_mean":
            mean.reshape(-1),

        "feature_std":
            std.reshape(-1),

        "num_samples":
            len(rows),

        "label_0":
            int((y == 0).sum()),

        "label_1":
            int((y == 1).sum()),

        "epochs": 300,

        "seed": SEED,

        "feature_order": [
            "prob_base[0:6]",
            "prob_lcr[0:6]",
            "prob_lcr-prob_base[0:6]",
            "margin_base",
            "margin_lcr",
            "entropy_base",
            "entropy_lcr",
            "target_evidence",
            "context_support",
            "conflict_score",
            "num_views",
            "R_before",
            "gate_min",
        ],
    },
    OUT,
)

print()
print("=" * 64)
print("FINAL LINEAR GUARD TRAINED")
print("=" * 64)
print("samples :", len(rows))
print("label 0 :", int((y == 0).sum()))
print("label 1 :", int((y == 1).sum()))
print(
    "params  :",
    sum(
        p.numel()
        for p in model.parameters()
    ),
)
print("saved   :", OUT)
