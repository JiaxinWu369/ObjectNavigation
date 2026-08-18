import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path("results/context_reversal_100k")

df = pd.read_csv(
    ROOT / "initial_attention_features.csv"
)

summary = json.loads(
    (ROOT / "summary.json").read_text()
)

FEATURE = "ctx_attention_max"

df["label"] = (
    df["group"] == "suppress_only"
).astype(int)

SCENES = sorted(
    df["scene"].unique()
)

K = 5
N_REPEATS = 200
MAX_ATTEMPTS = 20000
SEED = 20260818

rng = np.random.default_rng(SEED)


def balanced_accuracy(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    pos = y == 1
    neg = y == 0

    if not pos.any() or not neg.any():
        return np.nan

    tpr = (
        pred[pos] == 1
    ).mean()

    tnr = (
        pred[neg] == 0
    ).mean()

    return 0.5 * (tpr + tnr)


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
        pred = (
            x >= thr
        ).astype(int)

        bacc = balanced_accuracy(
            y,
            pred,
        )

        if bacc > best_bacc:
            best_bacc = bacc
            best_thr = float(thr)

    return best_thr


results = []

valid_repeat = 0
attempt = 0

while (
    valid_repeat < N_REPEATS
    and attempt < MAX_ATTEMPTS
):
    attempt += 1

    shuffled = np.array(
        SCENES,
        dtype=object,
    )

    rng.shuffle(shuffled)

    fold_scenes = np.array_split(
        shuffled,
        K,
    )

    # Require both reversal labels
    # in every held-out fold.
    valid = True

    for fold in range(K):
        test = df[
            df["scene"].isin(
                fold_scenes[fold]
            )
        ]

        if (
            test["label"].nunique()
            < 2
        ):
            valid = False
            break

    if not valid:
        continue

    predictions = np.zeros(
        len(df),
        dtype=int,
    )

    fold_bacc = []
    fold_auc = []
    thresholds = []

    for fold in range(K):

        test_scene_set = set(
            fold_scenes[fold]
        )

        test_mask = df["scene"].isin(
            test_scene_set
        )

        train_mask = ~test_mask

        train = df[train_mask]
        test = df[test_mask]

        thr = find_threshold(
            train[FEATURE],
            train["label"],
        )

        thresholds.append(thr)

        pred = (
            test[FEATURE].to_numpy()
            >= thr
        ).astype(int)

        predictions[
            np.where(test_mask)[0]
        ] = pred

        y_test = (
            test["label"]
            .to_numpy()
        )

        fold_bacc.append(
            balanced_accuracy(
                y_test,
                pred,
            )
        )

        pos = (
            test[
                test["label"] == 1
            ][FEATURE]
            .to_numpy()
        )

        neg = (
            test[
                test["label"] == 0
            ][FEATURE]
            .to_numpy()
        )

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

        fold_auc.append(auc)

    y = df["label"].to_numpy()

    pooled_bacc = balanced_accuracy(
        y,
        predictions,
    )

    tp = int(
        (
            (y == 1)
            & (predictions == 1)
        ).sum()
    )

    tn = int(
        (
            (y == 0)
            & (predictions == 0)
        ).sum()
    )

    # N00 never succeeds.
    # N11 succeeds whichever fixed
    # policy is selected.
    selector_success = (
        summary["N11_both_success"]
        + tp
        + tn
    )

    selector_sr = (
        100.0
        * selector_success
        / summary["N"]
    )

    results.append(
        {
            "repeat":
                valid_repeat,

            "pooled_bacc":
                pooled_bacc,

            "macro_fold_bacc":
                float(
                    np.mean(
                        fold_bacc
                    )
                ),

            "macro_fold_auc":
                float(
                    np.mean(
                        fold_auc
                    )
                ),

            "min_fold_auc":
                float(
                    np.min(
                        fold_auc
                    )
                ),

            "max_fold_auc":
                float(
                    np.max(
                        fold_auc
                    )
                ),

            "threshold_mean":
                float(
                    np.mean(
                        thresholds
                    )
                ),

            "threshold_sd":
                float(
                    np.std(
                        thresholds
                    )
                ),

            "tp":
                tp,

            "tn":
                tn,

            "selector_sr":
                selector_sr,
        }
    )

    valid_repeat += 1


if len(results) < N_REPEATS:
    raise RuntimeError(
        f"Only obtained "
        f"{len(results)} valid splits "
        f"after {attempt} attempts."
    )


res = pd.DataFrame(results)

res.to_csv(
    ROOT
    / "static_gate_repeated_cv.csv",
    index=False,
)


def report(name):
    x = res[name].to_numpy()

    print(
        f"{name:<22} "
        f"mean={x.mean():.4f}  "
        f"median={np.median(x):.4f}  "
        f"P10={np.percentile(x,10):.4f}  "
        f"P90={np.percentile(x,90):.4f}"
    )


print("=" * 100)
print("REPEATED SCENE-GROUPED STATIC GATE")
print("=" * 100)

print(
    "Valid repetitions :",
    len(res),
)

print(
    "Attempts          :",
    attempt,
)

print()

report("pooled_bacc")
report("macro_fold_bacc")
report("macro_fold_auc")
report("min_fold_auc")
report("threshold_mean")
report("threshold_sd")
report("selector_sr")

print()
print("=" * 100)
print("REFERENCE")
print("=" * 100)

print(
    "Base SR       : 29.92%"
)

print(
    "NoContext SR  : 31.59%"
)

print(
    "Uniform-0.1   : 31.67%"
)

print(
    "Oracle SR     : 36.11%"
)
