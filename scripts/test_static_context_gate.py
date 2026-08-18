import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu


ROOT = Path("results/context_reversal_100k")

df = pd.read_csv(
    ROOT / "initial_attention_features.csv"
)

FEATURE = "ctx_attention_max"

df["label"] = (
    df["group"] == "suppress_only"
).astype(int)


# ---------------------------------------------------------
# Deterministic scene-grouped 5-fold split
# ---------------------------------------------------------

scenes = sorted(df["scene"].unique())

scene_to_fold = {
    scene: i % 5
    for i, scene in enumerate(scenes)
}

df["fold"] = df["scene"].map(scene_to_fold)


def balanced_accuracy(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    pos = y == 1
    neg = y == 0

    tpr = (
        (pred[pos] == 1).mean()
        if pos.any()
        else np.nan
    )

    tnr = (
        (pred[neg] == 0).mean()
        if neg.any()
        else np.nan
    )

    return np.nanmean([tpr, tnr])


def find_threshold(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    values = np.unique(x)

    candidates = [
        values[0] - 1e-8
    ]

    if len(values) > 1:
        candidates.extend(
            (
                values[:-1]
                + values[1:]
            ) / 2.0
        )

    candidates.append(
        values[-1] + 1e-8
    )

    best_thr = None
    best_bacc = -1.0

    for thr in candidates:
        # Hypothesis:
        # larger peak context attention
        # -> suppress context
        pred = (x >= thr).astype(int)

        bacc = balanced_accuracy(
            y,
            pred,
        )

        if bacc > best_bacc:
            best_bacc = bacc
            best_thr = float(thr)

    return best_thr, best_bacc


all_predictions = np.zeros(
    len(df),
    dtype=int,
)

fold_results = []


print("=" * 100)
print("SCENE-GROUPED STATIC CONTEXT GATE")
print("=" * 100)

for fold in range(5):

    train_mask = (
        df["fold"] != fold
    )

    test_mask = (
        df["fold"] == fold
    )

    train = df[train_mask]
    test = df[test_mask]

    threshold, train_bacc = (
        find_threshold(
            train[FEATURE],
            train["label"],
        )
    )

    pred = (
        test[FEATURE].to_numpy()
        >= threshold
    ).astype(int)

    all_predictions[
        np.where(test_mask)[0]
    ] = pred

    test_bacc = balanced_accuracy(
        test["label"],
        pred,
    )

    pos = test[
        test["label"] == 1
    ][FEATURE].to_numpy()

    neg = test[
        test["label"] == 0
    ][FEATURE].to_numpy()

    if len(pos) > 0 and len(neg) > 0:
        u, _ = mannwhitneyu(
            pos,
            neg,
            alternative="two-sided",
        )

        auc = (
            float(u)
            /
            (len(pos) * len(neg))
        )
    else:
        auc = np.nan

    fold_results.append(
        {
            "fold": fold,
            "threshold": threshold,
            "train_bacc": train_bacc,
            "test_bacc": test_bacc,
            "test_auc": auc,
            "N": len(test),
            "N_sup": int(
                test["label"].sum()
            ),
            "N_base": int(
                (test["label"] == 0).sum()
            ),
        }
    )

    print()
    print(
        f"Fold {fold}: "
        f"N={len(test):3d} "
        f"Sup={int(test['label'].sum()):2d} "
        f"Base={int((test['label']==0).sum()):2d}"
    )

    print(
        f"  threshold  = {threshold:.6f}"
    )

    print(
        f"  train BAcc = {train_bacc:.3f}"
    )

    print(
        f"  test BAcc  = {test_bacc:.3f}"
    )

    print(
        f"  test AUC   = {auc:.3f}"
    )


# ---------------------------------------------------------
# Overall held-out predictions
# ---------------------------------------------------------

y = df["label"].to_numpy()

overall_bacc = balanced_accuracy(
    y,
    all_predictions,
)

tp = int(
    (
        (y == 1)
        & (all_predictions == 1)
    ).sum()
)

fn = int(
    (
        (y == 1)
        & (all_predictions == 0)
    ).sum()
)

tn = int(
    (
        (y == 0)
        & (all_predictions == 0)
    ).sum()
)

fp = int(
    (
        (y == 0)
        & (all_predictions == 1)
    ).sum()
)


print()
print("=" * 100)
print("OUT-OF-SCENE PREDICTION SUMMARY")
print("=" * 100)

print("TP suppress correctly :", tp)
print("FN suppress missed    :", fn)
print("TN preserve correctly :", tn)
print("FP preserve lost      :", fp)

print()
print(
    "Balanced accuracy     : "
    f"{overall_bacc:.3f}"
)

print(
    "Suppress recall       : "
    f"{tp / (tp + fn):.3f}"
)

print(
    "Preserve recall       : "
    f"{tn / (tn + fp):.3f}"
)


res = pd.DataFrame(
    fold_results
)

print()
print("=" * 100)
print("FOLD SUMMARY")
print("=" * 100)

print(
    res.to_string(
        index=False
    )
)

print()
print(
    "Mean fold AUC :",
    round(
        res["test_auc"].mean(),
        3,
    ),
)

print(
    "Mean threshold:",
    round(
        res["threshold"].mean(),
        6,
    ),
)

print(
    "Threshold SD  :",
    round(
        res["threshold"].std(),
        6,
    ),
)


out = df.copy()

out["cv_prediction"] = (
    all_predictions
)

out.to_csv(
    ROOT
    / "static_gate_cv_predictions.csv",
    index=False,
)

res.to_csv(
    ROOT
    / "static_gate_fold_results.csv",
    index=False,
)
