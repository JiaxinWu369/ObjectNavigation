import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


L2 = 1e-2

INPUT = Path(
    "results/aca_train500/selector/"
    "aca_v1_features_all_6655.jsonl"
)

OUT = Path(
    "results/aca_train500/selector/"
    "aca_v11_effect_oof"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


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


def load_rows():
    rows = []

    with open(
        INPUT,
        encoding="utf-8",
    ) as f:

        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def scene_fold(scene):

    n = int(
        scene.replace(
            "FloorPlan",
            "",
        )
    )

    if 1 <= n <= 20:
        local = n - 1

    elif 201 <= n <= 220:
        local = n - 201

    elif 301 <= n <= 320:
        local = n - 301

    elif 401 <= n <= 420:
        local = n - 401

    else:
        raise ValueError(scene)

    return local % 4


rows = load_rows()

assert len(rows) == 6655


X = np.asarray(
    [
        [
            r[k]
            for k in FEATURE_NAMES
        ]
        for r in rows
    ],
    dtype=np.float64,
)

y10 = np.asarray(
    [
        int(r["early10_success"])
        for r in rows
    ],
    dtype=np.int64,
)

y20 = np.asarray(
    [
        int(r["early20_success"])
        for r in rows
    ],
    dtype=np.int64,
)


# +1 RestoreNow helpful
# -1 Delay helpful
#  0 treatment makes no success difference
effect = (
    y10 - y20
).astype(
    np.float64
)


folds = np.asarray(
    [
        scene_fold(
            r["scene"]
        )
        for r in rows
    ],
    dtype=np.int64,
)


oof_effect = np.full(
    len(rows),
    np.nan,
)

oof_pred = np.full(
    len(rows),
    -1,
    dtype=np.int64,
)


print("=" * 96)
print("ACA-v1.1 PAIRED EFFECT OOF")
print("=" * 96)

print("Eligible :", len(rows))
print(
    "Effect +1:",
    int(np.sum(effect == 1)),
)
print(
    "Effect -1:",
    int(np.sum(effect == -1)),
)
print(
    "Effect  0:",
    int(np.sum(effect == 0)),
)
print("L2       :", L2)
print()


fold_results = []


for fold in range(4):

    train = folds != fold
    val = folds == fold

    mean = X[train].mean(
        axis=0
    )

    std = X[train].std(
        axis=0
    )

    std[
        std < 1e-8
    ] = 1.0

    Xtr = (
        X[train] - mean
    ) / std

    Xva = (
        X[val] - mean
    ) / std

    xt = torch.tensor(
        Xtr,
        dtype=torch.float64,
    )

    yt = torch.tensor(
        effect[train],
        dtype=torch.float64,
    ).view(-1, 1)

    model = nn.Linear(
        len(FEATURE_NAMES),
        1,
    ).double()

    with torch.no_grad():
        model.weight.zero_()

        # Mean paired treatment effect is the
        # fixed-policy prior.
        model.bias.fill_(
            float(
                effect[train].mean()
            )
        )

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=250,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():

        optimizer.zero_grad()

        pred = model(xt)

        mse = torch.mean(
            (
                pred - yt
            ) ** 2
        )

        reg = (
            0.5
            * L2
            * torch.sum(
                model.weight ** 2
            )
        )

        loss = mse + reg

        loss.backward()

        return loss

    optimizer.step(
        closure
    )

    xv = torch.tensor(
        Xva,
        dtype=torch.float64,
    )

    with torch.no_grad():

        pred_effect = (
            model(xv)
            .view(-1)
            .cpu()
            .numpy()
        )

    # Positive or zero => RestoreNow.
    # This preserves the better fixed Early10
    # as the tie/default behavior.
    arm = (
        pred_effect >= 0
    ).astype(
        np.int64
    )

    idx = np.where(
        val
    )[0]

    oof_effect[
        idx
    ] = pred_effect

    oof_pred[
        idx
    ] = arm

    selected = np.where(
        arm == 1,
        y10[idx],
        y20[idx],
    )

    sr10 = y10[
        idx
    ].mean()

    sr20 = y20[
        idx
    ].mean()

    sraca = selected.mean()

    oracle = np.maximum(
        y10[idx],
        y20[idx],
    ).mean()

    best = max(
        sr10,
        sr20,
    )

    gain = 100 * (
        sraca - best
    )

    fold_results.append(
        gain
    )

    discord = (
        y10[idx]
        !=
        y20[idx]
    )

    if np.any(discord):

        truth = (
            y10[idx][discord]
            >
            y20[idx][discord]
        ).astype(
            np.int64
        )

        acc = np.mean(
            arm[discord]
            ==
            truth
        )

    else:
        acc = float("nan")

    print(
        f"Fold {fold}: "
        f"N={len(idx):4d} "
        f"SR10={100*sr10:6.3f}% "
        f"SR20={100*sr20:6.3f}% "
        f"ACA={100*sraca:6.3f}% "
        f"Oracle={100*oracle:6.3f}% "
        f"Gain={gain:+.3f} pp "
        f"PrefAcc={100*acc:5.1f}% "
        f"Restore={100*arm.mean():5.1f}%"
    )


assert np.all(
    np.isfinite(
        oof_effect
    )
)


aca = np.where(
    oof_pred == 1,
    y10,
    y20,
)


sr10 = y10.mean()
sr20 = y20.mean()
sraca = aca.mean()

oracle = np.maximum(
    y10,
    y20,
).mean()

best = max(
    sr10,
    sr20,
)

gain = 100 * (
    sraca - best
)

headroom = 100 * (
    oracle - best
)


discord = (
    y10 != y20
)

truth = (
    y10[discord]
    >
    y20[discord]
).astype(
    np.int64
)

pref_acc = np.mean(
    oof_pred[discord]
    ==
    truth
)


positive_folds = sum(
    x > 0
    for x in fold_results
)


print()
print("=" * 96)
print("PRIMARY ACA-v1.1 RESULT")
print("=" * 96)

print(
    "Early10 SR          :",
    f"{100*sr10:.3f}%"
)

print(
    "Early20 SR          :",
    f"{100*sr20:.3f}%"
)

print(
    "Effect-OOF ACA SR   :",
    f"{100*sraca:.3f}%"
)

print(
    "Oracle SR           :",
    f"{100*oracle:.3f}%"
)

print(
    "Gain / best fixed   :",
    f"{gain:+.3f} pp"
)

print(
    "Oracle headroom     :",
    f"{headroom:.3f} pp"
)

print(
    "Headroom capture    :",
    f"{100*gain/headroom:.1f}%"
)

print(
    "Preference accuracy :",
    f"{100*pref_acc:.2f}%"
)

print(
    "Positive folds      :",
    f"{positive_folds}/4"
)


go = (
    gain >= 0.5
    and positive_folds >= 3
)


print()
print(
    "ACA-v1.1 DECISION   :",
    "GO" if go else "NO-GO"
)

print()
print(
    "ACA-v1.1 EFFECT OOF: PASS"
)
