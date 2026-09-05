import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata


L2 = 1e-2
THRESHOLD = 0.5

INPUT = Path(
    "results/aca_train500/selector/"
    "aca_v2_structured_all_6655.jsonl"
)

OUT = Path(
    "results/aca_train500/selector/"
    "aca_v2_target_loto"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


rows = [
    json.loads(x)
    for x in open(
        INPUT,
        encoding="utf-8",
    )
    if x.strip()
]

assert len(rows) == 6655


X = np.asarray(
    [
        r["features"]
        for r in rows
    ],
    dtype=np.float64,
)

assert X.shape == (6655, 87)


target = np.asarray(
    [r["target"] for r in rows]
)

pair = np.asarray(
    [r["pair_state"] for r in rows]
)

y10 = np.asarray(
    [
        int(r["early10_success"])
        for r in rows
    ]
)

y20 = np.asarray(
    [
        int(r["early20_success"])
        for r in rows
    ]
)


disc = np.isin(
    pair,
    ["N10", "N01"],
)

label = np.full(
    len(rows),
    -1,
    dtype=np.int64,
)

label[pair == "N10"] = 1
label[pair == "N01"] = 0


assert disc.sum() == 240
assert np.sum(label == 1) == 139
assert np.sum(label == 0) == 101


def fit_model(X, y):

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std[std < 1e-8] = 1.0

    Xz = (X - mean) / std

    xt = torch.tensor(
        Xz,
        dtype=torch.float64,
    )

    yt = torch.tensor(
        y,
        dtype=torch.float64,
    ).view(-1, 1)

    model = nn.Linear(
        X.shape[1],
        1,
    ).double()

    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=250,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():

        opt.zero_grad()

        logits = model(xt)

        loss = (
            F.binary_cross_entropy_with_logits(
                logits,
                yt,
            )
            +
            0.5
            * L2
            * torch.sum(
                model.weight ** 2
            )
        )

        loss.backward()

        return loss

    opt.step(closure)

    return model, mean, std


def predict(
    model,
    X,
    mean,
    std,
):

    Xz = (X - mean) / std

    xt = torch.tensor(
        Xz,
        dtype=torch.float64,
    )

    with torch.no_grad():

        p = torch.sigmoid(
            model(xt)
        ).view(-1).numpy()

    return p


def auc(y, score):

    y = np.asarray(y)
    score = np.asarray(score)

    n1 = np.sum(y == 1)
    n0 = np.sum(y == 0)

    if n1 == 0 or n0 == 0:
        return float("nan")

    ranks = rankdata(
        score,
        method="average",
    )

    return float(
        (
            ranks[y == 1].sum()
            - n1 * (n1 + 1) / 2
        )
        /
        (n1 * n0)
    )


targets = sorted(
    set(target.tolist())
)


oof_prob = np.full(
    len(rows),
    np.nan,
)

oof_pred = np.full(
    len(rows),
    -1,
)


print("=" * 108)
print("ACA-v2 LEAVE-ONE-TARGET-OUT DIAGNOSTIC")
print("=" * 108)

print("Targets      :", len(targets))
print("Eligible     :", len(rows))
print("Discordant   :", int(disc.sum()))
print("L2           :", L2)
print("Threshold    :", THRESHOLD)
print()


for held in targets:

    val = target == held

    train = (
        (target != held)
        & disc
    )

    ytr = label[train]

    if (
        np.sum(ytr == 1) == 0
        or np.sum(ytr == 0) == 0
    ):
        raise RuntimeError(
            f"one-class training set: {held}"
        )

    model, mean, std = fit_model(
        X[train],
        ytr,
    )

    p = predict(
        model,
        X[val],
        mean,
        std,
    )

    pred = (
        p >= THRESHOLD
    ).astype(np.int64)

    idx = np.where(val)[0]

    oof_prob[idx] = p
    oof_pred[idx] = pred

    selected = np.where(
        pred == 1,
        y10[idx],
        y20[idx],
    )

    sr10 = y10[idx].mean()
    sr20 = y20[idx].mean()
    sraca = selected.mean()

    best = max(sr10, sr20)

    gain = 100 * (
        sraca - best
    )

    vd = idx[
        disc[idx]
    ]

    if len(vd):

        pref_acc = np.mean(
            oof_pred[vd]
            ==
            label[vd]
        )

    else:

        pref_acc = float("nan")

    print(
        f"{held:<16} "
        f"N={len(idx):4d} "
        f"pref={len(vd):3d} "
        f"K10={100*sr10:6.2f}% "
        f"K20={100*sr20:6.2f}% "
        f"ACA={100*sraca:6.2f}% "
        f"gain={gain:+6.3f}pp "
        f"prefAcc={100*pref_acc:5.1f}%"
    )


assert np.all(
    np.isfinite(oof_prob)
)

assert np.all(
    oof_pred >= 0
)


aca_success = np.where(
    oof_pred == 1,
    y10,
    y20,
)


sr10 = y10.mean()
sr20 = y20.mean()
sraca = aca_success.mean()

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


di = np.where(disc)[0]

pref_acc = np.mean(
    oof_pred[di]
    ==
    label[di]
)


# Balanced accuracy
pos = label[di] == 1
neg = label[di] == 0

tpr = np.mean(
    oof_pred[di][pos] == 1
)

tnr = np.mean(
    oof_pred[di][neg] == 0
)

bacc = 0.5 * (
    tpr + tnr
)

auc_value = auc(
    label[di],
    oof_prob[di],
)


print()
print("=" * 108)
print("POOLED TARGET-HELD-OUT RESULT")
print("=" * 108)

print(
    "Early10 SR          :",
    f"{100*sr10:.3f}%"
)

print(
    "Early20 SR          :",
    f"{100*sr20:.3f}%"
)

print(
    "ACA-v2 LOTO SR      :",
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

print()
print(
    "Preference accuracy :",
    f"{100*pref_acc:.2f}%"
)

print(
    "Preference BAcc     :",
    f"{100*bacc:.2f}%"
)

print(
    "Preference AUC      :",
    f"{auc_value:.3f}"
)

print()
print(
    "Majority Restore Acc:",
    f"{100*139/240:.2f}%"
)

print()
print(
    "ACA-v2 TARGET LOTO DIAGNOSTIC: PASS"
)
