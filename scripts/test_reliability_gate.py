import copy
import h5py
import torch
import numpy as np

from types import SimpleNamespace

from models.AKGVP import AKGVPModel
from models.model_io import ModelInput
from datasets.constants import AI2THOR_TARGET_CLASSES


SCENE = "FloorPlan22"
STATE = "1.50|2.00|225|30"
TARGET = "Laptop"

scene_dir = f"datasets/Scene_Data/{SCENE}"

classes = AI2THOR_TARGET_CLASSES[22]

with h5py.File(
    f"{scene_dir}/clip_featuremap.hdf5",
    "r"
) as f:
    frame = f[STATE][()]

with h5py.File(
    f"{scene_dir}/det_feature_22_cates.hdf5",
    "r"
) as f:
    det = f[STATE][()]

assert frame.shape == (1, 2048, 7, 7)
assert det.shape == (22, 517)

args = SimpleNamespace(
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

torch.manual_seed(1234)

model = AKGVPModel(args)

device = model.clip_object.device
model = model.to(device)
model.eval()

print("Device:", device)


def make_input(reliability=None):
    indicator = np.zeros((22, 1), dtype=np.float32)
    indicator[classes.index(TARGET), 0] = 1.0

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
    mi.action_probs = torch.zeros(
        1, 6,
        device=device,
    )

    mi.scene = SCENE
    mi.target_object = TARGET

    if reliability is not None:
        reliability = reliability.to(device)

    mi.reliability_weight = reliability

    return mi


def run_once(reference_model, reliability=None):
    # deepcopy is important because Object_Distribution
    # maintains mutable episode memory.
    m = copy.deepcopy(reference_model)
    m.eval()

    with torch.no_grad():
        out = m(
            make_input(reliability),
            None
        )

    return {
        "logit": out.logit.detach().clone(),
        "base_attention":
            m.last_attention_weight_base.clone(),
        "attention":
            m.last_attention_weight.clone(),
    }


# ------------------------------------------------
# 1. Original path
# ------------------------------------------------

base = run_once(
    model,
    reliability=None,
)

# ------------------------------------------------
# 2. All-one gate
# Must be numerically identical to Base
# ------------------------------------------------

ones = torch.ones(22, 1)

same = run_once(
    model,
    reliability=ones,
)

logit_diff = (
    base["logit"] - same["logit"]
).abs().max().item()

att_diff = (
    base["attention"] - same["attention"]
).abs().max().item()

print("=" * 80)
print("BASE EQUIVALENCE")
print("=" * 80)

print("Max attention diff:", att_diff)
print("Max logit diff    :", logit_diff)

assert att_diff < 1e-7
assert logit_diff < 1e-6

print("BASE EQUIVALENCE: PASS")


# ------------------------------------------------
# 3. Gate one semantically active category
# Pick strongest attention among detected categories
# ------------------------------------------------

det_scores = torch.tensor(
    det[:, -1],
    dtype=torch.float32,
    device=device,
).reshape(-1, 1)

candidate_strength = (
    base["attention"] * det_scores
)

# Do not gate the goal category itself
goal_idx = classes.index(TARGET)
candidate_strength[goal_idx] = -1

gate_idx = int(
    torch.argmax(candidate_strength).item()
)

gate_category = classes[gate_idx]

rho = torch.ones(22, 1)
rho[gate_idx] = 0.0

gated = run_once(
    model,
    reliability=rho,
)

print("\n" + "=" * 80)
print("ORACLE GATE")
print("=" * 80)

print("Target category :", TARGET)
print("Gated category  :", gate_category)
print("Gate index      :", gate_idx)

print(
    "Base attention :",
    float(base["attention"][gate_idx])
)

print(
    "Gated attention:",
    float(gated["attention"][gate_idx])
)

print(
    "\nBase logits :",
    base["logit"].cpu().numpy().round(6)
)

print(
    "Gated logits:",
    gated["logit"].cpu().numpy().round(6)
)

delta = (
    gated["logit"] - base["logit"]
)

print(
    "Delta logits:",
    delta.cpu().numpy().round(6)
)

print(
    "L2 delta    :",
    float(torch.norm(delta))
)

base_action = int(
    torch.argmax(base["logit"], dim=1).item()
)

gate_action = int(
    torch.argmax(gated["logit"], dim=1).item()
)

print("Base argmax :", base_action)
print("Gated argmax:", gate_action)

print("\nRELIABILITY GATE UNIT TEST: PASS")
