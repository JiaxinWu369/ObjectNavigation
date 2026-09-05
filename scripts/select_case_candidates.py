import json
from pathlib import Path

ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

BASE = Path("results/final_test_base500")
OURS = Path("results/guard_tau02_zstest")


def load(root):
    rows = {}
    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                x = json.loads(line)
                key = (
                    x.get("scene_type", room),
                    int(x["eval_index"]),
                )
                rows[key] = x
    return rows


A = load(BASE)
B = load(OURS)

assert set(A) == set(B)

cases = []

for key in sorted(A):
    a = A[key]
    b = B[key]

    if (
        not bool(a["success"])
        and bool(b["success"])
    ):
        cases.append({
            "key": key,
            "scene": a["scene"],
            "target": a["target"],
            "base_len": a["ep_length"],
            "ours_len": b["ep_length"],
            "gain": a["ep_length"] - b["ep_length"],
        })

print("repair cases:", len(cases))
print()

# 优先选择 Base 跑满/较长，而 Ours 较快成功的 episode
cases = sorted(
    cases,
    key=lambda x: (
        -x["base_len"],
        x["ours_len"],
    )
)

for i, x in enumerate(cases, 1):
    print(
        f"{i:02d}",
        x["key"],
        "scene=", x["scene"],
        "target=", x["target"],
        "BaseLen=", x["base_len"],
        "OursLen=", x["ours_len"],
    )
