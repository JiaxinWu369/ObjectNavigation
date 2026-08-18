import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from datasets.constants import AI2THOR_TARGET_CLASSES
from models.AKGVP import AKGVPModel
from models.model_io import ModelInput


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument(
    "--root",
    default="results/context_reversal_100k",
)
args_cli = parser.parse_args()


ROOT = Path(args_cli.root)
DATA_ROOT = Path("datasets/Scene_Data")

CLASSES = list(
    AI2THOR_TARGET_CLASSES[22]
)

GROUP_FILES = {
    "suppress_only":
        ROOT / "suppress_only.csv",
    "base_only":
        ROOT / "base_only.csv",
}


# --------------------------------------------------
# Model
# --------------------------------------------------

model_args = SimpleNamespace(
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

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

model = AKGVPModel(model_args)

state_dict = torch.load(
    args_cli.checkpoint,
    map_location="cpu",
)

result = model.load_state_dict(
    state_dict,
    strict=True,
)

model = model.to(device)
model.eval()

print("Checkpoint :", args_cli.checkpoint)
print("Missing    :", result.missing_keys)
print("Unexpected :", result.unexpected_keys)
print("Device     :", device)


def state_key(row):
    x = float(row["start_x"])
    z = float(row["start_z"])

    if abs(x) < 0.005:
        x = 0.0

    if abs(z) < 0.005:
        z = 0.0

    rot = int(
        round(
            float(row["start_rotation"])
        )
    )

    hor = int(
        round(
            float(row["start_horizon"])
        )
    )

    return (
        f"{x:.2f}|{z:.2f}|"
        f"{rot}|{hor}"
    )


def entropy_positive(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x) & (x > 0)
    ]

    if len(x) <= 1:
        return 0.0

    p = x / x.sum()

    return float(
        -(p * np.log(p + 1e-12)).sum()
    )


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / denom
    )


def load_features(
    scene,
    state,
):
    scene_dir = (
        DATA_ROOT / scene
    )

    with h5py.File(
        scene_dir / "clip_featuremap.hdf5",
        "r",
    ) as f:
        if state not in f:
            raise KeyError(
                f"Missing frame: "
                f"{scene} {state}"
            )

        frame = f[state][()]

    with h5py.File(
        scene_dir
        / "det_feature_22_cates.hdf5",
        "r",
    ) as f:
        if state not in f:
            raise KeyError(
                f"Missing detection: "
                f"{scene} {state}"
            )

        det = f[state][()]

    if det.shape != (22, 517):
        raise RuntimeError(
            f"Unexpected det shape "
            f"{det.shape}: "
            f"{scene} {state}"
        )

    return frame, det


def make_input(
    frame,
    det,
    target,
    scene,
):
    goal_idx = CLASSES.index(target)

    indicator = np.zeros(
        (22, 1),
        dtype=np.float32,
    )

    indicator[
        goal_idx, 0
    ] = 1.0

    target_embedding = {
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

    # Initial episode action embedding.
    mi.action_probs = torch.zeros(
        1,
        6,
        device=device,
    )

    mi.target_class_embedding = (
        target_embedding
    )

    mi.scene = scene
    mi.target_object = target

    # Exact Base path.
    mi.reliability_weight = None

    return mi


def prior_attention(
    scene,
    goal_idx,
):
    od = model.object_distribution

    scene_id = od.get_scene_id(
        scene
    )

    alpha = (
        od.init_prior_alpha[
            scene_id
        ]
        .detach()
        .float()
        .to(device)
    )

    denom = (
        alpha.sum(
            dim=1,
            keepdim=True,
        )
        .clamp_min(1e-12)
    )

    prob = alpha / denom

    prob = (
        prob
        + torch.eye(
            prob.shape[0],
            device=device,
        )
    )

    return (
        prob[:, goal_idx]
        .detach()
        .cpu()
        .numpy()
    )


def extract_one(row):
    scene = row["scene"]
    target = row["target"]
    state = state_key(row)

    goal_idx = CLASSES.index(
        target
    )

    frame, det = load_features(
        scene,
        state,
    )

    det_scores = np.asarray(
        det[:, -1],
        dtype=np.float64,
    )

    # Each row must behave like a fresh
    # episode. Object_Distribution keeps
    # mutable episode memory.
    model.object_distribution.reset_memory()

    prior_att = prior_attention(
        scene,
        goal_idx,
    )

    mi = make_input(
        frame,
        det,
        target,
        scene,
    )

    with torch.no_grad():
        _ = model(
            mi,
            None,
        )

    att = (
        model.last_attention_weight_base
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float64)
    )

    if not np.all(
        np.isfinite(att)
    ):
        raise RuntimeError(
            f"Non-finite attention: "
            f"{scene} {state}"
        )

    if att.shape != (22,):
        raise RuntimeError(
            f"Attention shape="
            f"{att.shape}"
        )

    ctx_mask = np.ones(
        22,
        dtype=bool,
    )
    ctx_mask[goal_idx] = False

    ctx_att = att[ctx_mask]
    ctx_prior = prior_att[
        ctx_mask
    ]

    ctx_scores = det_scores[
        ctx_mask
    ]

    detected = ctx_scores > 0

    detected_att = (
        ctx_att[detected]
    )

    detected_scores = (
        ctx_scores[detected]
    )

    attended_score = (
        ctx_att * ctx_scores
    )

    goal_att = float(
        att[goal_idx]
    )

    ctx_att_max = float(
        ctx_att.max()
    )

    ctx_att_sum = float(
        ctx_att.sum()
    )

    eps = 1e-8

    if detected.any():
        detected_att_max = float(
            detected_att.max()
        )

        detected_att_mean = float(
            detected_att.mean()
        )

        detected_att_sum = float(
            detected_att.sum()
        )
    else:
        detected_att_max = 0.0
        detected_att_mean = 0.0
        detected_att_sum = 0.0

    top_ctx_local = int(
        np.argmax(ctx_att)
    )

    ctx_indices = np.arange(22)[
        ctx_mask
    ]

    top_ctx_idx = int(
        ctx_indices[
            top_ctx_local
        ]
    )

    top_attdet_local = int(
        np.argmax(
            attended_score
        )
    )

    top_attdet_idx = int(
        ctx_indices[
            top_attdet_local
        ]
    )

    delta_ctx = (
        ctx_att - ctx_prior
    )

    # KL generated by AKGVP's
    # observation-conditioned distribution
    # update at the initial state.
    kl_value = (
        model.object_distribution.kl
    )

    if torch.is_tensor(kl_value):
        semantic_kl = float(
            kl_value
            .detach()
            .reshape(-1)[0]
            .cpu()
            .item()
        )
    else:
        semantic_kl = float(
            kl_value
        )

    return {
        "scene": scene,
        "scene_type":
            row["scene_type"],
        "eval_index":
            int(row["eval_index"]),
        "target": target,
        "state": state,

        # Target-conditioned graph attention.
        "goal_attention":
            goal_att,

        "ctx_attention_max":
            ctx_att_max,

        "ctx_attention_mean":
            float(ctx_att.mean()),

        "ctx_attention_sum":
            ctx_att_sum,

        "ctx_attention_entropy":
            entropy_positive(
                ctx_att
            ),

        "ctx_att_minus_goal":
            ctx_att_max
            - goal_att,

        "log_ctxatt_goal_ratio":
            float(
                math.log(
                    (
                        ctx_att_max
                        + eps
                    )
                    /
                    (
                        goal_att
                        + eps
                    )
                )
            ),

        # Attention restricted to
        # currently detected context.
        "detected_ctx_att_max":
            detected_att_max,

        "detected_ctx_att_mean":
            detected_att_mean,

        "detected_ctx_att_sum":
            detected_att_sum,

        "detected_att_mass_ratio":
            float(
                detected_att_sum
                /
                (
                    ctx_att_sum
                    + eps
                )
            ),

        # Joint semantic relevance
        # × detector evidence.
        "att_det_max":
            float(
                attended_score.max()
            ),

        "att_det_mean":
            float(
                attended_score.mean()
            ),

        "att_det_sum":
            float(
                attended_score.sum()
            ),

        "att_det_entropy":
            entropy_positive(
                attended_score
            ),

        "attention_detector_cosine":
            cosine(
                ctx_att,
                np.maximum(
                    ctx_scores,
                    0.0,
                ),
            ),

        # Is the highest semantic-context
        # relation actually observed?
        "top_attention_detected":
            int(
                det_scores[
                    top_ctx_idx
                ] > 0
            ),

        # Observation-induced change
        # relative to static scene prior.
        "ctx_attention_shift_l1":
            float(
                np.mean(
                    np.abs(
                        delta_ctx
                    )
                )
            ),

        "ctx_attention_shift_linf":
            float(
                np.max(
                    np.abs(
                        delta_ctx
                    )
                )
            ),

        "goal_attention_shift":
            float(
                att[goal_idx]
                - prior_att[
                    goal_idx
                ]
            ),

        "semantic_update_kl":
            semantic_kl,

        # Diagnostic only.
        "top_attention_category":
            CLASSES[
                top_ctx_idx
            ],

        "top_attdet_category":
            CLASSES[
                top_attdet_idx
            ],
    }


rows = []

for group, path in (
    GROUP_FILES.items()
):
    df = pd.read_csv(path)

    print(
        f"Loading {group}: "
        f"{len(df)}"
    )

    for i, (_, row) in enumerate(
        df.iterrows(),
        1,
    ):
        feat = extract_one(row)
        feat["group"] = group
        rows.append(feat)

        if (
            i % 20 == 0
            or i == len(df)
        ):
            print(
                f"  {i}/{len(df)}"
            )


out = pd.DataFrame(rows)

out_path = (
    ROOT
    / "initial_attention_features.csv"
)

out.to_csv(
    out_path,
    index=False,
)

print()
print("Saved:", out_path)
print("N =", len(out))


FEATURES = [
    "goal_attention",
    "ctx_attention_max",
    "ctx_attention_mean",
    "ctx_attention_sum",
    "ctx_attention_entropy",
    "ctx_att_minus_goal",
    "log_ctxatt_goal_ratio",

    "detected_ctx_att_max",
    "detected_ctx_att_mean",
    "detected_ctx_att_sum",
    "detected_att_mass_ratio",

    "att_det_max",
    "att_det_mean",
    "att_det_sum",
    "att_det_entropy",

    "attention_detector_cosine",
    "top_attention_detected",

    "ctx_attention_shift_l1",
    "ctx_attention_shift_linf",
    "goal_attention_shift",
    "semantic_update_kl",
]


def compare_groups(
    df,
    title,
):
    sup = df[
        df["group"]
        == "suppress_only"
    ]

    base = df[
        df["group"]
        == "base_only"
    ]

    print()
    print("=" * 126)
    print(title)
    print("=" * 126)

    print(
        "N suppress-only="
        f"{len(sup)}  "
        "base-only="
        f"{len(base)}"
    )

    print(
        f"{'Feature':<30}"
        f"{'Sup mean':>12}"
        f"{'Base mean':>12}"
        f"{'Sup med':>12}"
        f"{'Base med':>12}"
        f"{'AUC':>9}"
        f"{'RBC':>9}"
        f"{'p':>14}"
    )

    print("-" * 124)

    for feature in FEATURES:
        x = (
            sup[feature]
            .astype(float)
            .to_numpy()
        )

        y = (
            base[feature]
            .astype(float)
            .to_numpy()
        )

        u, p = mannwhitneyu(
            x,
            y,
            alternative="two-sided",
        )

        auc = (
            float(u)
            /
            (
                len(x)
                * len(y)
            )
        )

        rbc = (
            2.0 * auc
            - 1.0
        )

        print(
            f"{feature:<30}"
            f"{x.mean():>12.5f}"
            f"{y.mean():>12.5f}"
            f"{np.median(x):>12.5f}"
            f"{np.median(y):>12.5f}"
            f"{auc:>9.3f}"
            f"{rbc:>9.3f}"
            f"{p:>14.6g}"
        )


compare_groups(
    out,
    "POOLED ATTENTION FEATURE COMPARISON",
)


# Clean within-target comparisons.
for target in [
    "Laptop",
    "LightSwitch",
]:
    sub = out[
        out["target"]
        == target
    ]

    ns = int(
        (
            sub["group"]
            == "suppress_only"
        ).sum()
    )

    nb = int(
        (
            sub["group"]
            == "base_only"
        ).sum()
    )

    if ns > 0 and nb > 0:
        compare_groups(
            sub,
            "WITHIN TARGET "
            f"ATTENTION: {target}",
        )
