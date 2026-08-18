import argparse
import csv
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from models.AKGVP import AKGVPModel
from models.model_io import ModelInput
from datasets.constants import AI2THOR_TARGET_CLASSES


RHO_VALUES = [1.0, 0.5, 0.1, 0.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="datasets/Scene_Data")
    p.add_argument("--split-dir", default="test_val_split")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--output",
        default="results/gate_sensitivity_50k/details.csv",
    )
    return p.parse_args()


def make_model_args():
    return SimpleNamespace(
        action_space=6,
        num_category=22,
        hidden_state_sz=512,
        dropout_rate=0.25,
        feature_memory=False,
        model_phase="test",
        TDE_threshold=2.0,
        scale_min=-0.001,
        scale_max=0.8,
        TDE_mode="random",
    )


def load_model(checkpoint):
    torch.manual_seed(1234)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AKGVPModel(make_model_args())

    state = torch.load(
        checkpoint,
        map_location="cpu",
    )

    result = model.load_state_dict(
        state,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print("Checkpoint :", checkpoint)
    print("Missing    :", result.missing_keys)
    print("Unexpected :", result.unexpected_keys)
    print("Device     :", device)

    return model, device


def reset_model_state(model):
    # Match the episode reset logic used by NavigationAgent.
    if hasattr(model, "reset"):
        model.reset()

    if hasattr(model, "object_distribution"):
        model.object_distribution.reset_memory()


def make_input(
    frame,
    det,
    scene,
    target_name,
    reliability,
    classes,
    device,
):
    indicator = np.zeros(
        (len(classes), 1),
        dtype=np.float32,
    )

    goal_idx = classes.index(target_name)
    indicator[goal_idx, 0] = 1.0

    target = {
        "appear": torch.tensor(
            det[:, :512],
            dtype=torch.float32,
            device=device,
        ),
        "info": torch.tensor(
            det[:, 512:],
            dtype=torch.float32,
            device=device,
        ),
        "indicator": torch.tensor(
            indicator,
            dtype=torch.float32,
            device=device,
        ),
    }

    mi = ModelInput()

    mi.state = torch.tensor(
        frame,
        dtype=torch.float32,
        device=device,
    )

    mi.hidden = (
        torch.zeros(
            2, 1, 512,
            device=device,
        ),
        torch.zeros(
            2, 1, 512,
            device=device,
        ),
    )

    mi.target_class_embedding = target

    # First-step diagnostic:
    # no previous action yet.
    mi.action_probs = torch.zeros(
        1, 6,
        device=device,
    )

    mi.scene = scene
    mi.target_object = target_name

    if reliability is not None:
        reliability = reliability.to(device)

    mi.reliability_weight = reliability

    return mi


def run_once(
    model,
    frame,
    det,
    scene,
    target_name,
    reliability,
    classes,
    device,
):
    reset_model_state(model)
    model.eval()

    model_input = make_input(
        frame,
        det,
        scene,
        target_name,
        reliability,
        classes,
        device,
    )

    with torch.no_grad():
        out = model(
            model_input,
            None,
        )

    logits = out.logit.detach().cpu().clone()

    probs = torch.softmax(
        logits,
        dim=1,
    )

    return {
        "logits": logits,
        "probs": probs,
        "base_attention":
            model.last_attention_weight_base
            .detach()
            .cpu()
            .clone(),
        "attention":
            model.last_attention_weight
            .detach()
            .cpu()
            .clone(),
    }


def load_validation_episodes(
    split_dir,
    limit,
    seed,
):
    root = Path(split_dir)

    episodes = []

    for pkl_path in sorted(
        root.glob("*_zs_val_22.pkl")
    ):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        for idx, ep in enumerate(data):
            item = dict(ep)
            item["_split_file"] = pkl_path.name
            item["_split_index"] = idx
            episodes.append(item)

    print(
        "Available ZS-val episodes:",
        len(episodes),
    )

    rng = random.Random(seed)
    rng.shuffle(episodes)

    if limit > 0:
        episodes = episodes[:limit]

    print(
        "Selected episodes:",
        len(episodes),
    )

    return episodes


def load_features(
    data_dir,
    scene,
    state,
):
    scene_dir = (
        Path(data_dir) / scene
    )

    with h5py.File(
        scene_dir / "clip_featuremap.hdf5",
        "r",
    ) as f:
        frame = f[state][()]

    with h5py.File(
        scene_dir /
        "det_feature_22_cates.hdf5",
        "r",
    ) as f:
        det = f[state][()]

    if frame.shape != (1, 2048, 7, 7):
        raise RuntimeError(
            f"Unexpected frame shape "
            f"{frame.shape}"
        )

    if det.shape != (22, 517):
        raise RuntimeError(
            f"Unexpected DET shape "
            f"{det.shape}"
        )

    return frame, det


def probability_metrics(
    base_probs,
    gated_probs,
):
    eps = 1e-12

    p = base_probs.clamp_min(eps)
    q = gated_probs.clamp_min(eps)

    l1 = torch.sum(
        torch.abs(p - q)
    ).item()

    kl = torch.sum(
        p * (
            torch.log(p) -
            torch.log(q)
        )
    ).item()

    return l1, kl


def main():
    args = parse_args()

    classes = list(
        AI2THOR_TARGET_CLASSES[22]
    )

    model, device = load_model(
        args.checkpoint
    )

    episodes = load_validation_episodes(
        args.split_dir,
        args.limit,
        args.seed,
    )

    rows = []
    skipped = 0

    equivalence_max = 0.0

    for n, ep in enumerate(
        episodes,
        start=1,
    ):
        scene = ep["scene"]
        state = ep["state"]
        target = ep["goal_object_type"]

        frame, det = load_features(
            args.data_dir,
            scene,
            state,
        )

        # -----------------------------
        # Original trained AKGVP
        # -----------------------------
        base = run_once(
            model,
            frame,
            det,
            scene,
            target,
            reliability=None,
            classes=classes,
            device=device,
        )

        goal_idx = classes.index(target)

        scores = torch.tensor(
            det[:, -1],
            dtype=torch.float32,
        ).reshape(-1)

        attention = (
            base["attention"]
            .reshape(-1)
        )

        # Select the strongest currently
        # observed NON-GOAL semantic category.
        #
        # This is not yet the final
        # "wrong-instance oracle".
        # It is a controlled coupling test.
        valid = scores > 0
        valid[goal_idx] = False

        if not torch.any(valid):
            skipped += 1
            continue

        strength = (
            attention * scores
        ).clone()

        strength[~valid] = -float("inf")

        gate_idx = int(
            torch.argmax(
                strength
            ).item()
        )

        gate_category = classes[gate_idx]

        base_action = int(
            torch.argmax(
                base["probs"],
                dim=1,
            ).item()
        )

        # -----------------------------
        # rho sweep
        # -----------------------------
        for rho_value in RHO_VALUES:
            rho = torch.ones(
                22,
                1,
                dtype=torch.float32,
            )

            rho[gate_idx, 0] = rho_value

            gated = run_once(
                model,
                frame,
                det,
                scene,
                target,
                reliability=rho,
                classes=classes,
                device=device,
            )

            gated_action = int(
                torch.argmax(
                    gated["probs"],
                    dim=1,
                ).item()
            )

            delta_logits = (
                gated["logits"] -
                base["logits"]
            )

            logit_l2 = float(
                torch.norm(
                    delta_logits
                ).item()
            )

            l1, kl = probability_metrics(
                base["probs"],
                gated["probs"],
            )

            max_logit_diff = float(
                torch.max(
                    torch.abs(
                        delta_logits
                    )
                ).item()
            )

            if rho_value == 1.0:
                equivalence_max = max(
                    equivalence_max,
                    max_logit_diff,
                )

            rows.append({
                "episode_no": n,
                "split_file":
                    ep["_split_file"],
                "split_index":
                    ep["_split_index"],
                "scene": scene,
                "state": state,
                "target": target,

                "gate_category":
                    gate_category,
                "gate_index":
                    gate_idx,

                "det_score":
                    float(
                        scores[gate_idx]
                    ),

                "base_attention":
                    float(
                        base["attention"][
                            gate_idx
                        ]
                    ),

                "rho":
                    rho_value,

                "gated_attention":
                    float(
                        gated["attention"][
                            gate_idx
                        ]
                    ),

                "base_action":
                    base_action,
                "gated_action":
                    gated_action,

                "action_changed":
                    int(
                        base_action !=
                        gated_action
                    ),

                "logit_l2":
                    logit_l2,

                "prob_l1":
                    l1,

                "kl_base_to_gate":
                    kl,
            })

        if n % 20 == 0:
            print(
                f"Processed "
                f"{n}/{len(episodes)}"
            )

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "No valid episodes."
        )

    with open(
        out,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 90)
    print("TRAINED GATE SENSITIVITY")
    print("=" * 90)

    print(
        "Episodes requested :",
        len(episodes),
    )
    print(
        "Episodes skipped   :",
        skipped,
    )
    print(
        "Valid episodes     :",
        len(rows) // len(RHO_VALUES),
    )

    print(
        "rho=1 equivalence "
        "max logit diff:",
        equivalence_max,
    )

    print()
    print(
        f"{'rho':>6}"
        f"{'N':>8}"
        f"{'mean_L1':>14}"
        f"{'mean_KL':>14}"
        f"{'mean_L2':>14}"
        f"{'action_change':>18}"
    )

    print("-" * 74)

    for rho in RHO_VALUES:
        rr = [
            x for x in rows
            if x["rho"] == rho
        ]

        mean_l1 = np.mean([
            x["prob_l1"]
            for x in rr
        ])

        mean_kl = np.mean([
            x["kl_base_to_gate"]
            for x in rr
        ])

        mean_l2 = np.mean([
            x["logit_l2"]
            for x in rr
        ])

        change = np.mean([
            x["action_changed"]
            for x in rr
        ])

        print(
            f"{rho:>6.1f}"
            f"{len(rr):>8d}"
            f"{mean_l1:>14.6f}"
            f"{mean_kl:>14.6f}"
            f"{mean_l2:>14.6f}"
            f"{100*change:>17.2f}%"
        )

    print()
    print(
        "Details saved to:",
        out,
    )

    # all-ones gate must reproduce Base.
    if equivalence_max > 1e-5:
        raise RuntimeError(
            "rho=1.0 does not reproduce Base."
        )

    print(
        "\nBASE EQUIVALENCE: PASS"
    )


if __name__ == "__main__":
    main()
