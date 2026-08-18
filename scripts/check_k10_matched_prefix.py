import json
from pathlib import Path

ROOMS = ["kitchen", "living_room", "bedroom", "bathroom"]

def load_run(root):
    root = Path(root)
    out = {}
    for room in ROOMS:
        with open(root / f"episodes_{room}.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                out[(r["scene_type"], int(r["eval_index"]))] = r
    return out

nc = load_run("results/pair_nocontext_base100")
k10 = load_run("results/pair_early10_nocontext_base100")

assert set(nc) == set(k10)

prefix_mismatch = 0
short_episode = 0
short_mismatch = 0
discordant = 0
discordant_short = 0

for key in nc:
    a = nc[key]
    b = k10[key]

    assert a["scene"] == b["scene"]
    assert a["target"] == b["target"]
    assert a["start_state"] == b["start_state"]

    aa = a["actions"]
    bb = b["actions"]

    if min(len(aa), len(bb)) < 10:
        short_episode += 1
        if aa != bb:
            short_mismatch += 1
    else:
        if aa[:10] != bb[:10]:
            prefix_mismatch += 1

    if bool(a["success"]) != bool(b["success"]):
        discordant += 1
        if min(len(aa), len(bb)) < 10:
            discordant_short += 1

print("Episodes                :", len(nc))
print("Prefix mismatch first 10:", prefix_mismatch)
print("Short episodes (<10)    :", short_episode)
print("Short mismatch           :", short_mismatch)
print("Success discordant       :", discordant)
print("Discordant short         :", discordant_short)

assert prefix_mismatch == 0
assert short_mismatch == 0
assert discordant == 79
assert discordant_short == 0

print("\nK10 MATCHED PREFIX: PASS")
