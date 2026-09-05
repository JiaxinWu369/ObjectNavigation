import argparse
import os
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scipy.stats import rankdata

from reliability.lcr_causal_features import (
    FEATURE_DIM,
    region_sequence_features,
)


SEED = 20260822


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path):
    rows = [
        json.loads(x)
        for x in open(
            path,
            encoding="utf-8",
        )
        if x.strip()
    ]

    rows = [
        r
        for r in rows
        if r["label"] is not None
    ]

    return rows


def binary_metrics(
    y_true,
    y_prob,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=np.float64,
    )

    n_pos = int(
        np.sum(y_true == 1)
    )

    n_neg = int(
        np.sum(y_true == 0)
    )

    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
    else:
        ranks = rankdata(
            y_prob,
            method="average",
        )

        pos_rank_sum = float(
            ranks[
                y_true == 1
            ].sum()
        )

        auc = (
            pos_rank_sum
            - n_pos
            * (n_pos + 1)
            / 2.0
        ) / (
            n_pos * n_neg
        )

    # Average precision / AUPRC.
    order = np.argsort(
        -y_prob,
        kind="mergesort",
    )

    ys = y_true[order]

    tp = np.cumsum(
        ys == 1
    )

    fp = np.cumsum(
        ys == 0
    )

    precision = tp / np.maximum(
        tp + fp,
        1,
    )

    if n_pos > 0:
        ap = float(
            precision[
                ys == 1
            ].sum()
            / n_pos
        )
    else:
        ap = float("nan")

    pred = (
        y_prob >= 0.5
    ).astype(
        np.int64
    )

    tp_n = int(
        np.sum(
            (pred == 1)
            &
            (y_true == 1)
        )
    )

    tn_n = int(
        np.sum(
            (pred == 0)
            &
            (y_true == 0)
        )
    )

    fp_n = int(
        np.sum(
            (pred == 1)
            &
            (y_true == 0)
        )
    )

    fn_n = int(
        np.sum(
            (pred == 0)
            &
            (y_true == 1)
        )
    )

    tpr = (
        tp_n / (tp_n + fn_n)
        if tp_n + fn_n > 0
        else 0.0
    )

    tnr = (
        tn_n / (tn_n + fp_n)
        if tn_n + fp_n > 0
        else 0.0
    )

    bacc = (
        tpr + tnr
    ) / 2.0

    acc = float(
        np.mean(
            pred == y_true
        )
    )

    brier = float(
        np.mean(
            (
                y_prob
                - y_true
            ) ** 2
        )
    )

    return {
        "auc": float(auc),
        "auprc": float(ap),
        "bacc": float(bacc),
        "accuracy": float(acc),
        "brier": float(brier),
        "positive_rate": float(
            y_true.mean()
        ),
        "n": int(len(y_true)),
        "positive": int(n_pos),
        "negative": int(n_neg),
    }


class LCRDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        rows,
        mean,
        std,
    ):
        self.rows = rows
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.rows)

    def __getitem__(
        self,
        index,
    ):
        row = self.rows[index]

        x = (
            region_sequence_features(
                row
            )
        )

        x = (
            x - self.mean
        ) / self.std

        y = float(
            row["label"]
        )

        return (
            torch.from_numpy(
                x.astype(
                    np.float32
                )
            ),
            torch.tensor(
                y,
                dtype=torch.float32,
            ),
        )


def collate_batch(batch):
    xs, ys = zip(*batch)

    lengths = torch.tensor(
        [
            x.shape[0]
            for x in xs
        ],
        dtype=torch.long,
    )

    max_len = int(
        lengths.max()
    )

    batch_x = torch.zeros(
        len(xs),
        max_len,
        FEATURE_DIM,
        dtype=torch.float32,
    )

    mask = torch.zeros(
        len(xs),
        max_len,
        dtype=torch.float32,
    )

    for i, x in enumerate(xs):
        n = x.shape[0]

        batch_x[
            i,
            :n,
        ] = x

        mask[
            i,
            :n,
        ] = 1.0

    y = torch.stack(
        ys
    )

    return (
        batch_x,
        lengths,
        mask,
        y,
    )


class StaticMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                FEATURE_DIM,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x,
        lengths,
        mask,
    ):
        del lengths

        weighted = (
            x
            * mask.unsqueeze(-1)
        )

        pooled = (
            weighted.sum(dim=1)
            /
            mask.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
        )

        return (
            self.net(
                pooled
            )
            .squeeze(-1)
        )


class TemporalGRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                FEATURE_DIM,
                32,
            ),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=32,
            hidden_size=32,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(
                32,
                16,
            ),
            nn.ReLU(),
            nn.Linear(
                16,
                1,
            ),
        )

    def forward(
        self,
        x,
        lengths,
        mask,
    ):
        del mask

        z = self.encoder(x)

        packed = (
            nn.utils.rnn
            .pack_padded_sequence(
                z,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
        )

        _, h = self.gru(
            packed
        )

        h = h[-1]

        return (
            self.head(h)
            .squeeze(-1)
        )


def compute_train_stats(
    rows,
):
    all_steps = []

    for row in rows:
        all_steps.append(
            region_sequence_features(
                row
            )
        )

    all_steps = np.concatenate(
        all_steps,
        axis=0,
    )

    mean = all_steps.mean(
        axis=0
    ).astype(
        np.float32
    )

    std = all_steps.std(
        axis=0
    ).astype(
        np.float32
    )

    std[
        std < 1e-6
    ] = 1.0

    return mean, std


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    probs = []
    labels = []

    for (
        x,
        lengths,
        mask,
        y,
    ) in loader:

        x = x.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)

        logits = model(
            x,
            lengths,
            mask,
        )

        p = torch.sigmoid(
            logits
        )

        probs.extend(
            p.cpu().numpy().tolist()
        )

        labels.extend(
            y.numpy().tolist()
        )

    return binary_metrics(
        labels,
        probs,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=(
            "results/lcr_causal_dataset/"
            "prefixes_labeled.jsonl"
        ),
    )

    parser.add_argument(
        "--model",
        choices=[
            "mlp",
            "gru",
        ],
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--split-mode",
        choices=[
            "scene",
            "target",
        ],
        default="scene",
    )

    parser.add_argument(
        "--target-fold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--target-fold-file",
        default=(
            "results/"
            "lcr_target_folds.json"
        ),
    )

    args = parser.parse_args()

    set_seed(SEED)

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows(
        args.dataset
    )

    heldout_targets = None

    if args.split_mode == "scene":

        train_rows = [
            r
            for r in rows
            if r["data_split"]
            == "train"
        ]

        # One final prefix per validation region.
        val_rows = [
            r
            for r in rows
            if (
                r["data_split"]
                == "val"
                and r.get(
                    "is_final_prefix",
                    False,
                )
            )
        ]

    else:

        if args.target_fold is None:
            raise ValueError(
                "--target-fold is required "
                "for --split-mode target"
            )

        with open(
            args.target_fold_file,
            "r",
            encoding="utf-8",
        ) as f:
            fold_data = json.load(f)

        fold_lookup = {
            int(x["fold"]): x
            for x in fold_data[
                "folds"
            ]
        }

        if (
            args.target_fold
            not in fold_lookup
        ):
            raise ValueError(
                "Unknown target fold: "
                "{}".format(
                    args.target_fold
                )
            )

        heldout_targets = set(
            fold_lookup[
                args.target_fold
            ][
                "heldout_targets"
            ]
        )

        # IMPORTANT:
        # target diagnostic uses only the 64
        # train scenes. This isolates target
        # generalization from scene generalization.
        train_rows = [
            r
            for r in rows
            if (
                r["data_split"]
                == "train"
                and r["target"]
                not in heldout_targets
            )
        ]

        # Again, one final prefix per region.
        val_rows = [
            r
            for r in rows
            if (
                r["data_split"]
                == "train"
                and r["target"]
                in heldout_targets
                and r.get(
                    "is_final_prefix",
                    False,
                )
            )
        ]

    train_scenes = {
        r["scene"]
        for r in train_rows
    }

    val_scenes = {
        r["scene"]
        for r in val_rows
    }

    if args.split_mode == "scene":

        assert not (
            train_scenes
            &
            val_scenes
        )

    else:

        train_targets = {
            r["target"]
            for r in train_rows
        }

        val_targets = {
            r["target"]
            for r in val_rows
        }

        assert not (
            train_targets
            &
            val_targets
        )

        assert val_targets == (
            heldout_targets
        )

    mean, std = (
        compute_train_stats(
            train_rows
        )
    )

    np.savez(
        output
        / "feature_stats.npz",
        mean=mean,
        std=std,
    )

    train_ds = LCRDataset(
        train_rows,
        mean,
        std,
    )

    val_ds = LCRDataset(
        val_rows,
        mean,
        std,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = (
        torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_batch,
            generator=generator,
        )
    )

    val_loader = (
        torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )
    )

    if args.model == "mlp":
        model = StaticMLP()
    else:
        model = TemporalGRU()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    majority = max(
        np.mean(
            [
                r["label"]
                for r in val_rows
            ]
        ),
        1.0
        - np.mean(
            [
                r["label"]
                for r in val_rows
            ]
        ),
    )

    print("=" * 80)
    print("LCR RELIABILITY PREDICTOR")
    print("=" * 80)
    print("model          :", args.model)
    print("split mode     :", args.split_mode)

    if args.split_mode == "target":
        print(
            "target fold    :",
            args.target_fold,
        )
        print(
            "heldout targets:",
            sorted(
                heldout_targets
            ),
        )

    print("train regions  :", len(train_rows))
    print("val regions    :", len(val_rows))
    print("train scenes   :", len(train_scenes))
    print("val scenes     :", len(val_scenes))
    print(
        "val majority acc:",
        round(
            float(majority),
            4,
        ),
    )
    print("device         :", device)
    print(
        "drop goal score:",
        os.environ.get(
            "AKGVP_LCR_DROP_GOAL_SCORE",
            "0",
        ),
    )

    best_auc = -math.inf
    bad_epochs = 0
    best_metrics = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        losses = []

        for (
            x,
            lengths,
            mask,
            y,
        ) in train_loader:

            x = x.to(device)
            lengths = (
                lengths.to(device)
            )
            mask = mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(
                x,
                lengths,
                mask,
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            losses.append(
                float(loss.item())
            )

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            "epoch={:02d} "
            "loss={:.4f} "
            "AUC={:.4f} "
            "AUPRC={:.4f} "
            "BAcc={:.4f} "
            "Acc={:.4f} "
            "Brier={:.4f}".format(
                epoch,
                sum(losses)
                / len(losses),
                metrics["auc"],
                metrics["auprc"],
                metrics["bacc"],
                metrics["accuracy"],
                metrics["brier"],
            )
        )

        if (
            metrics["auc"]
            >
            best_auc + 1e-6
        ):
            best_auc = (
                metrics["auc"]
            )

            best_metrics = dict(
                metrics
            )

            bad_epochs = 0

            torch.save(
                {
                    "model":
                        args.model,

                    "state_dict":
                        model.state_dict(),

                    "feature_mean":
                        mean,

                    "feature_std":
                        std,

                    "feature_dim":
                        FEATURE_DIM,

                    "seed":
                        SEED,

                    "metrics":
                        best_metrics,
                },
                output
                / "best.pt",
            )

            with open(
                output
                / "best_metrics.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    best_metrics,
                    f,
                    indent=2,
                )

        else:
            bad_epochs += 1

            if (
                bad_epochs
                >= args.patience
            ):
                print(
                    "Early stopping."
                )
                break

    print()
    print("=" * 80)
    print("BEST VALIDATION")
    print("=" * 80)

    for k, v in (
        best_metrics.items()
    ):
        print(
            f"{k:14s}: {v}"
        )

    print()
    print(
        "checkpoint:",
        output / "best.pt"
    )


if __name__ == "__main__":
    main()
