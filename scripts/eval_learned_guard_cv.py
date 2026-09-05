import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DATA = Path(
    "results/learned_guard_protocol/"
    "guard_train_samples.jsonl"
)

SEEDS = [0, 1, 2]
N_FOLDS = 5


def scalar(x, key):
    v = x.get(key, None)
    return 0.0 if v is None else float(v)


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

    out = np.concatenate(
        [pb, pl, dp, extra]
    )

    assert out.shape == (28,)
    return out


def auc_score(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)

    pos = y == 1
    neg = y == 0

    n1 = int(pos.sum())
    n0 = int(neg.sum())

    if n1 == 0 or n0 == 0:
        return float("nan")

    order = np.argsort(
        score,
        kind="mergesort",
    )

    ranks = np.empty(
        len(score),
        dtype=np.float64,
    )

    ranks[order] = np.arange(
        1,
        len(score) + 1,
        dtype=np.float64,
    )

    # average tied ranks
    unique = {}
    for i, s in enumerate(score):
        unique.setdefault(
            float(s), []
        ).append(i)

    for ids in unique.values():
        if len(ids) > 1:
            r = ranks[ids].mean()
            ranks[ids] = r

    return float(
        (
            ranks[pos].sum()
            - n1 * (n1 + 1) / 2
        )
        / (n1 * n0)
    )


class LinearGuard(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(28, 1)

    def forward(self, x):
        return self.net(x)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(28, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


rows = []

for line in DATA.read_text(
    encoding="utf-8"
).splitlines():

    if not line.strip():
        continue

    x = json.loads(line)

    if x["label"] is not None:
        rows.append(x)


print("=" * 72)
print("DATA")
print("=" * 72)
print("clean samples:", len(rows))
print(
    "labels:",
    Counter(x["label"] for x in rows)
)
print(
    "scenes:",
    len(set(x["scene"] for x in rows))
)
print(
    "targets:",
    len(set(x["target"] for x in rows))
)


def make_folds(group_key, seed):
    groups = {}

    for x in rows:
        g = x[group_key]
        groups.setdefault(g, []).append(x)

    names = list(groups)

    rng = random.Random(seed)
    rng.shuffle(names)

    # greedy balance by sample count
    names.sort(
        key=lambda g: len(groups[g]),
        reverse=True,
    )

    folds = [[] for _ in range(N_FOLDS)]
    sizes = [0] * N_FOLDS

    for g in names:
        k = int(np.argmin(sizes))
        folds[k].append(g)
        sizes[k] += len(groups[g])

    return folds


def train_eval(
    model_name,
    train_rows,
    val_rows,
    seed,
):
    Xtr = np.stack(
        [feature(x) for x in train_rows]
    )

    ytr = np.asarray(
        [x["label"] for x in train_rows],
        dtype=np.float32,
    )

    Xva = np.stack(
        [feature(x) for x in val_rows]
    )

    yva = np.asarray(
        [x["label"] for x in val_rows],
        dtype=np.float32,
    )

    mean = Xtr.mean(
        axis=0,
        keepdims=True,
    )

    std = Xtr.std(
        axis=0,
        keepdims=True,
    )

    std[std < 1e-6] = 1.0

    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    Xtr = torch.tensor(
        Xtr,
        dtype=torch.float32,
    )

    Xva = torch.tensor(
        Xva,
        dtype=torch.float32,
    )

    ytr_t = torch.tensor(
        ytr,
        dtype=torch.float32,
    ).reshape(-1, 1)

    torch.manual_seed(seed)

    if model_name == "linear":
        model = LinearGuard()
        epochs = 300
    else:
        model = TinyMLP()
        epochs = 150

    npos = float(ytr_t.sum())
    nneg = float(len(ytr_t) - npos)

    pos_weight = torch.tensor(
        [nneg / max(npos, 1.0)]
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    for _ in range(epochs):
        model.train()

        logits = model(Xtr)
        loss = loss_fn(logits, ytr_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()

    with torch.no_grad():
        prob = torch.sigmoid(
            model(Xva)
        ).reshape(-1).numpy()

    return (
        auc_score(yva, prob),
        yva,
        prob,
    )


def run_cv(
    title,
    group_key,
):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    for model_name in [
        "linear",
        "mlp",
    ]:
        aucs = []
        pooled_y = []
        pooled_p = []

        for seed in SEEDS:
            folds = make_folds(
                group_key,
                seed,
            )

            for fold_id, val_groups in enumerate(
                folds
            ):
                val_groups = set(val_groups)

                train_rows = [
                    x for x in rows
                    if x[group_key]
                    not in val_groups
                ]

                val_rows = [
                    x for x in rows
                    if x[group_key]
                    in val_groups
                ]

                auc, y, p = train_eval(
                    model_name,
                    train_rows,
                    val_rows,
                    seed * 100 + fold_id,
                )

                aucs.append(auc)
                pooled_y.extend(y.tolist())
                pooled_p.extend(p.tolist())

                print(
                    model_name,
                    "seed",
                    seed,
                    "fold",
                    fold_id,
                    "train",
                    len(train_rows),
                    "val",
                    len(val_rows),
                    "auc",
                    round(auc, 4),
                    "groups",
                    sorted(val_groups),
                )

        a = np.asarray(
            aucs,
            dtype=np.float64,
        )

        pooled_auc = auc_score(
            pooled_y,
            pooled_p,
        )

        print()
        print(
            model_name,
            "MEAN AUC =",
            round(float(np.nanmean(a)), 4),
        )

        print(
            model_name,
            "STD AUC  =",
            round(float(np.nanstd(a)), 4),
        )

        print(
            model_name,
            "POOLED AUC =",
            round(float(pooled_auc), 4),
        )

        print()


run_cv(
    "SCENE-HELD-OUT CV",
    "scene",
)

run_cv(
    "TARGET-HELD-OUT CV",
    "target",
)
