import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import rankdata


SEED = 20260820
L2 = 1e-2
THRESHOLD = 0.5

INPUT = Path(
    "results/aca_train500/selector/"
    "aca_v2_structured_all_6655.jsonl"
)

OUT_DIR = Path(
    "results/aca_train500/selector/"
    "aca_v2_scene_oof"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_rows():

    rows = []

    with open(
        INPUT,
        "r",
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
        raise ValueError(
            f"Unexpected scene: {scene}"
        )

    return local % 4


def fit_logistic(
    X,
    y,
    mean,
    std,
):

    Xz = (
        X - mean
    ) / std

    xt = torch.tensor(
        Xz,
        dtype=torch.float64,
    )

    yt = torch.tensor(
        y,
        dtype=torch.float64,
    ).view(
        -1,
        1,
    )

    model = nn.Linear(
        X.shape[1],
        1,
        bias=True,
    ).double()

    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

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

        logits = model(
            xt
        )

        bce = (
            F.binary_cross_entropy_with_logits(
                logits,
                yt,
            )
        )

        reg = (
            0.5
            * L2
            * torch.sum(
                model.weight ** 2
            )
        )

        loss = (
            bce + reg
        )

        loss.backward()

        return loss

    optimizer.step(
        closure
    )

    return model


def predict_prob(
    model,
    X,
    mean,
    std,
):

    Xz = (
        X - mean
    ) / std

    xt = torch.tensor(
        Xz,
        dtype=torch.float64,
    )

    with torch.no_grad():

        p = torch.sigmoid(
            model(xt)
        ).view(-1)

    return (
        p.cpu().numpy()
    )


def auc_score(
    y,
    score,
):

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    n1 = int(
        np.sum(y == 1)
    )

    n0 = int(
        np.sum(y == 0)
    )

    if n1 == 0 or n0 == 0:
        return float("nan")

    ranks = rankdata(
        score,
        method="average",
    )

    auc = (
        ranks[y == 1].sum()
        -
        n1 * (n1 + 1) / 2.0
    ) / (
        n1 * n0
    )

    return float(auc)


def balanced_accuracy(
    y,
    pred,
):

    y = np.asarray(y)
    pred = np.asarray(pred)

    pos = (
        y == 1
    )

    neg = (
        y == 0
    )

    if (
        not pos.any()
        or not neg.any()
    ):
        return float("nan")

    tpr = np.mean(
        pred[pos] == 1
    )

    tnr = np.mean(
        pred[neg] == 0
    )

    return float(
        0.5 * (
            tpr + tnr
        )
    )


rows = load_rows()

assert len(rows) == 6655


X = np.asarray(
    [
        r["features"]
        for r in rows
    ],
    dtype=np.float64,
)

assert X.shape == (
    6655,
    87,
)

assert np.all(
    np.isfinite(X)
)


y10 = np.asarray(
    [
        int(
            r["early10_success"]
        )
        for r in rows
    ],
    dtype=np.int64,
)

y20 = np.asarray(
    [
        int(
            r["early20_success"]
        )
        for r in rows
    ],
    dtype=np.int64,
)


pair = np.asarray(
    [
        r["pair_state"]
        for r in rows
    ]
)


is_disc = np.isin(
    pair,
    [
        "N10",
        "N01",
    ],
)


label = np.full(
    len(rows),
    -1,
    dtype=np.int64,
)

label[
    pair == "N10"
] = 1

label[
    pair == "N01"
] = 0


assert int(
    is_disc.sum()
) == 240

assert int(
    np.sum(label == 1)
) == 139

assert int(
    np.sum(label == 0)
) == 101


folds = np.asarray(
    [
        scene_fold(
            r["scene"]
        )
        for r in rows
    ],
    dtype=np.int64,
)


oof_prob = np.full(
    len(rows),
    np.nan,
    dtype=np.float64,
)

oof_pred = np.full(
    len(rows),
    -1,
    dtype=np.int64,
)


fold_results = []


print("=" * 108)
print("ACA-v2 STRUCTURED 4-FOLD SCENE-GROUPED OOF")
print("=" * 108)

print(
    "L2             :",
    L2,
)

print(
    "Threshold      :",
    THRESHOLD,
)

print(
    "Feature dim    :",
    X.shape[1],
)

print(
    "Eligible       :",
    len(rows),
)

print(
    "Preference N   :",
    int(
        is_disc.sum()
    ),
)

print()


for fold in range(4):

    val_mask = (
        folds == fold
    )

    train_disc = (
        (folds != fold)
        & is_disc
    )

    val_disc = (
        val_mask
        & is_disc
    )


    Xtr = X[
        train_disc
    ]

    ytr = label[
        train_disc
    ]


    mean = Xtr.mean(
        axis=0
    )

    std = Xtr.std(
        axis=0
    )

    std[
        std < 1e-8
    ] = 1.0


    model = fit_logistic(
        Xtr,
        ytr,
        mean,
        std,
    )


    p = predict_prob(
        model,
        X[val_mask],
        mean,
        std,
    )

    pred = (
        p >= THRESHOLD
    ).astype(
        np.int64
    )


    oof_prob[
        val_mask
    ] = p

    oof_pred[
        val_mask
    ] = pred


    idx = np.where(
        val_mask
    )[0]


    selected = np.where(
        pred == 1,
        y10[idx],
        y20[idx],
    )


    sr10 = float(
        y10[idx].mean()
    )

    sr20 = float(
        y20[idx].mean()
    )

    sr_aca = float(
        selected.mean()
    )

    sr_oracle = float(
        np.maximum(
            y10[idx],
            y20[idx],
        ).mean()
    )


    best_fixed = max(
        sr10,
        sr20,
    )

    gain_pp = 100.0 * (
        sr_aca
        - best_fixed
    )


    vd = np.where(
        val_disc
    )[0]

    disc_y = label[
        vd
    ]

    disc_prob = oof_prob[
        vd
    ]

    disc_pred = oof_pred[
        vd
    ]


    acc = float(
        np.mean(
            disc_pred
            ==
            disc_y
        )
    )

    bacc = balanced_accuracy(
        disc_y,
        disc_pred,
    )

    auc = auc_score(
        disc_y,
        disc_prob,
    )


    restore_rate = float(
        pred.mean()
    )


    result = {
        "fold":
            fold,

        "val_n":
            int(
                val_mask.sum()
            ),

        "train_preference_n":
            int(
                train_disc.sum()
            ),

        "val_preference_n":
            int(
                val_disc.sum()
            ),

        "early10_sr":
            sr10,

        "early20_sr":
            sr20,

        "aca_sr":
            sr_aca,

        "oracle_sr":
            sr_oracle,

        "gain_over_best_fixed_pp":
            gain_pp,

        "preference_accuracy":
            acc,

        "preference_balanced_accuracy":
            bacc,

        "preference_auc":
            auc,

        "restore_prediction_rate":
            restore_rate,
    }


    fold_results.append(
        result
    )


    print(
        f"Fold {fold}: "
        f"N={result['val_n']:4d}  "
        f"pref={result['val_preference_n']:3d}  "
        f"train_pref={result['train_preference_n']:3d}"
    )

    print(
        "  "
        f"SR K10={100*sr10:6.3f}%  "
        f"K20={100*sr20:6.3f}%  "
        f"ACA={100*sr_aca:6.3f}%  "
        f"Oracle={100*sr_oracle:6.3f}%  "
        f"Gain={gain_pp:+.3f} pp"
    )

    print(
        "  "
        f"Pref Acc={100*acc:6.2f}%  "
        f"BAcc={100*bacc:6.2f}%  "
        f"AUC={auc:.3f}  "
        f"RestoreRate={100*restore_rate:6.2f}%"
    )

    print()


assert np.all(
    np.isfinite(
        oof_prob
    )
)

assert np.all(
    oof_pred >= 0
)


aca_success = np.where(
    oof_pred == 1,
    y10,
    y20,
)


sr10_all = float(
    y10.mean()
)

sr20_all = float(
    y20.mean()
)

sr_aca_all = float(
    aca_success.mean()
)

sr_oracle_all = float(
    np.maximum(
        y10,
        y20,
    ).mean()
)


best_fixed_all = max(
    sr10_all,
    sr20_all,
)


gain_all_pp = 100.0 * (
    sr_aca_all
    - best_fixed_all
)

oracle_headroom_pp = 100.0 * (
    sr_oracle_all
    - best_fixed_all
)


if oracle_headroom_pp > 0:

    oracle_capture = (
        gain_all_pp
        / oracle_headroom_pp
    )

else:

    oracle_capture = 0.0


disc_idx = np.where(
    is_disc
)[0]


overall_pref_acc = float(
    np.mean(
        oof_pred[
            disc_idx
        ]
        ==
        label[
            disc_idx
        ]
    )
)


overall_pref_bacc = (
    balanced_accuracy(
        label[
            disc_idx
        ],
        oof_pred[
            disc_idx
        ],
    )
)


overall_pref_auc = auc_score(
    label[
        disc_idx
    ],
    oof_prob[
        disc_idx
    ],
)


positive_folds = sum(
    r[
        "gain_over_best_fixed_pp"
    ] > 0
    for r in fold_results
)


go = (
    gain_all_pp >= 0.5
    and positive_folds >= 3
)


print("=" * 108)
print("PRIMARY ACA-v2 OOF NAVIGATION RESULT")
print("=" * 108)

print(
    "Fixed Early10 SR       :",
    f"{100*sr10_all:.3f}%"
)

print(
    "Fixed Early20 SR       :",
    f"{100*sr20_all:.3f}%"
)

print(
    "ACA-v2 OOF SR          :",
    f"{100*sr_aca_all:.3f}%"
)

print(
    "Oracle SR              :",
    f"{100*sr_oracle_all:.3f}%"
)

print(
    "ACA gain / best fixed  :",
    f"{gain_all_pp:+.3f} pp"
)

print(
    "Oracle headroom        :",
    f"{oracle_headroom_pp:.3f} pp"
)

print(
    "Oracle headroom capture:",
    f"{100*oracle_capture:.1f}%"
)

print(
    "Positive folds         :",
    f"{positive_folds}/4"
)

print()

print(
    "Preference OOF Acc     :",
    f"{100*overall_pref_acc:.2f}%"
)

print(
    "Preference OOF BAcc    :",
    f"{100*overall_pref_bacc:.2f}%"
)

print(
    "Preference OOF AUC     :",
    f"{overall_pref_auc:.3f}"
)

print()

print(
    "GO criterion:"
)

print(
    "  gain >= +0.500 pp    :",
    gain_all_pp >= 0.5
)

print(
    "  positive folds >= 3  :",
    positive_folds >= 3
)

print()

print(
    "ACA-v2 DECISION        :",
    "GO" if go else "NO-GO"
)


oof_path = (
    OUT_DIR
    / "oof_predictions_6655.jsonl"
)


with open(
    oof_path,
    "w",
    encoding="utf-8",
) as f:

    for i, r in enumerate(rows):

        out = {
            "batch":
                r["batch"],

            "room":
                r["room"],

            "scene":
                r["scene"],

            "target":
                r["target"],

            "start_state":
                r["start_state"],

            "pair_state":
                r["pair_state"],

            "fold":
                int(
                    folds[i]
                ),

            "restore_probability":
                float(
                    oof_prob[i]
                ),

            "restore_prediction":
                int(
                    oof_pred[i]
                ),

            "early10_success":
                int(
                    y10[i]
                ),

            "early20_success":
                int(
                    y20[i]
                ),

            "aca_success":
                int(
                    aca_success[i]
                ),
        }


        f.write(
            json.dumps(
                out,
                ensure_ascii=False,
            )
            + "\n"
        )


summary = {
    "seed":
        SEED,

    "representation":
        "21 prior-ranked context slots x4 + 3 global",

    "feature_dim":
        87,

    "l2":
        L2,

    "threshold":
        THRESHOLD,

    "eligible_n":
        len(rows),

    "preference_n":
        int(
            is_disc.sum()
        ),

    "restore_n":
        int(
            np.sum(
                label == 1
            )
        ),

    "delay_n":
        int(
            np.sum(
                label == 0
            )
        ),

    "fixed_early10_sr":
        sr10_all,

    "fixed_early20_sr":
        sr20_all,

    "aca_oof_sr":
        sr_aca_all,

    "oracle_sr":
        sr_oracle_all,

    "gain_over_best_fixed_pp":
        gain_all_pp,

    "oracle_headroom_pp":
        oracle_headroom_pp,

    "oracle_capture":
        oracle_capture,

    "positive_folds":
        positive_folds,

    "preference_oof_accuracy":
        overall_pref_acc,

    "preference_oof_balanced_accuracy":
        overall_pref_bacc,

    "preference_oof_auc":
        overall_pref_auc,

    "go":
        go,

    "fold_results":
        fold_results,
}


with open(
    OUT_DIR / "summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print(
    "OOF predictions:",
    oof_path,
)

print(
    "Summary        :",
    OUT_DIR / "summary.json",
)

print()

print(
    "ACA-v2 SCENE OOF: PASS"
)
