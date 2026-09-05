import json
import numpy as np
from pathlib import Path


for model in [
    "mlp",
    "gru",
]:

    rows = []

    for fold in range(4):

        path = Path(
            "results/"
            f"lcr_targetcv_{model}_"
            f"fold{fold}/"
            "best_metrics.json"
        )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            m = json.load(f)

        rows.append(m)

    print("=" * 72)
    print(
        "TARGET-HELD-OUT:",
        model.upper(),
    )
    print("=" * 72)

    for i, m in enumerate(rows):
        print(
            "fold={} "
            "N={} "
            "pos={} "
            "neg={} "
            "AUC={:.4f} "
            "AUPRC={:.4f} "
            "BAcc={:.4f} "
            "Acc={:.4f} "
            "Brier={:.4f}".format(
                i,
                m["n"],
                m["positive"],
                m["negative"],
                m["auc"],
                m["auprc"],
                m["bacc"],
                m["accuracy"],
                m["brier"],
            )
        )

    print()

    for metric in [
        "auc",
        "auprc",
        "bacc",
        "brier",
    ]:

        values = np.asarray(
            [
                x[metric]
                for x in rows
            ],
            dtype=float,
        )

        print(
            f"{metric:<8} "
            f"mean={values.mean():.4f} "
            f"std={values.std(ddof=1):.4f} "
            f"min={values.min():.4f} "
            f"max={values.max():.4f}"
        )

    print()
