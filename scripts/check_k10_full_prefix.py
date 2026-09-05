import json
import re
from pathlib import Path
from collections import defaultdict

ROOMS = ["kitchen", "living_room", "bedroom", "bathroom"]

NC = Path("results/pair_nocontext_base100")
K10 = Path("results/pair_early10_nocontext_base100")


def room_from_scene(scene):
    n = int(re.search(r"\d+", scene).group())

    if n < 200:
        return "kitchen"
    if n < 300:
        return "living_room"
    if n < 400:
        return "bedroom"
    return "bathroom"


def state_from_vis(s):
    x, z, rot, hor = s.split("|")

    return (
        round(float(x), 4),
        round(float(z), 4),
        int(round(float(rot))),
        int(round(float(hor))),
    )


def state_from_json(r):
    s = r["start_state"]

    return (
        round(float(s["x"]), 4),
        round(float(s["z"]), 4),
        int(round(float(s["rotation"]["y"]))),
        int(round(float(s["horizon"]))),
    )


def target_from_vis(v):
    t = v["target"]

    if isinstance(t, list):
        cats = {
            str(x).split("|")[0]
            for x in t
        }

        if len(cats) != 1:
            raise RuntimeError(
                f"multiple target categories: {t}"
            )

        return next(iter(cats))

    return str(t).split("|")[0]


def load_jsonl(root):
    out = {}

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"

        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)

                key = (
                    room,
                    int(r["eval_index"]),
                )

                out[key] = r

    return out


def load_vis(root, json_rows):
    p = (
        root
        / "visualization_files"
        / "visualization_metrics.json"
    )

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1260

    # Important:
    # global order is interleaved across processes,
    # but relative order from each room process is preserved.
    by_room = defaultdict(list)

    for v in data:
        room = room_from_scene(v["scene"])
        by_room[room].append(v)

    out = {}
    mismatch = []

    for room in ROOMS:

        expected = sum(
            1
            for r, _ in json_rows
            if r == room
        )

        print(
            f"{room:<12} "
            f"vis={len(by_room[room]):4d} "
            f"jsonl={expected:4d}"
        )

        assert len(by_room[room]) == expected

        for idx, v in enumerate(by_room[room]):

            key = (room, idx)
            j = json_rows[key]

            problems = []

            if v["scene"] != j["scene"]:
                problems.append(
                    f"scene {v['scene']} != {j['scene']}"
                )

            vt = target_from_vis(v)

            if vt != j["target"]:
                problems.append(
                    f"target {vt} != {j['target']}"
                )

            vs = state_from_vis(
                v["states"][0]
            )

            js = state_from_json(j)

            if vs != js:
                problems.append(
                    f"start {vs} != {js}"
                )

            if bool(v["success"]) != bool(j["success"]):
                problems.append(
                    f"success "
                    f"{v['success']} != {j['success']}"
                )

            if problems:
                mismatch.append(
                    (key, problems)
                )

            out[key] = v

    if mismatch:

        print("\nMAPPING MISMATCH EXAMPLES")

        for x in mismatch[:20]:
            print(x)

        raise RuntimeError(
            f"mapping failed: {len(mismatch)} mismatches"
        )

    print("metadata validation: PASS")

    return out


nc_json = load_jsonl(NC)
k10_json = load_jsonl(K10)

assert set(nc_json) == set(k10_json)


print("=" * 80)
print("NOCONTEXT MAPPING")
print("=" * 80)

nc_vis = load_vis(
    NC,
    nc_json,
)


print()
print("=" * 80)
print("EARLY10 MAPPING")
print("=" * 80)

k10_vis = load_vis(
    K10,
    k10_json,
)


assert set(nc_vis) == set(k10_vis)


prefix_mismatch = []
state10_mismatch = []
length_mismatch = []
discordant = []
discordant_before_restore = []


for key in sorted(nc_vis):

    a = nc_vis[key]
    b = k10_vis[key]

    ja = nc_json[key]
    jb = k10_json[key]

    aa = [
        int(x)
        for x in a["action_list"]
    ]

    bb = [
        int(x)
        for x in b["action_list"]
    ]

    # Full history sanity check
    if (
        len(aa) != int(ja["ep_length"])
        or len(a["states"]) != int(ja["ep_length"])
    ):
        length_mismatch.append(
            ("NC", key, len(aa),
             len(a["states"]),
             ja["ep_length"])
        )

    if (
        len(bb) != int(jb["ep_length"])
        or len(b["states"]) != int(jb["ep_length"])
    ):
        length_mismatch.append(
            ("K10", key, len(bb),
             len(b["states"]),
             jb["ep_length"])
        )

    # Actions 0..9 must be identical.
    n = min(10, len(aa), len(bb))

    if aa[:n] != bb[:n]:
        prefix_mismatch.append(
            (
                key,
                aa[:10],
                bb[:10],
            )
        )

    # If both reach state before action index 10,
    # states[10] must also be identical.
    if len(aa) >= 11 and len(bb) >= 11:

        sa = state_from_vis(
            a["states"][10]
        )

        sb = state_from_vis(
            b["states"][10]
        )

        if sa != sb:
            state10_mismatch.append(
                (key, sa, sb)
            )

    # Final outcome reversal
    suc_a = bool(ja["success"])
    suc_b = bool(jb["success"])

    if suc_a != suc_b:

        discordant.append(key)

        # Restoration begins when action index 10
        # is evaluated/executed.
        if len(aa) < 11 or len(bb) < 11:
            discordant_before_restore.append(
                (
                    key,
                    len(aa),
                    len(bb),
                    suc_a,
                    suc_b,
                )
            )


print()
print("=" * 80)
print("K10 MATCHED INTERVENTION CHECK")
print("=" * 80)

print(
    "Episodes                      :",
    len(nc_vis)
)

print(
    "Full history length mismatch  :",
    len(length_mismatch)
)

print(
    "First-10 action mismatch      :",
    len(prefix_mismatch)
)

print(
    "t=10 state mismatch           :",
    len(state10_mismatch)
)

print(
    "Success-discordant episodes   :",
    len(discordant)
)

print(
    "Discordant before restoration :",
    len(discordant_before_restore)
)


if prefix_mismatch:
    print("\nPREFIX EXAMPLES")
    for x in prefix_mismatch[:10]:
        print(x)

if state10_mismatch:
    print("\nSTATE10 EXAMPLES")
    for x in state10_mismatch[:10]:
        print(x)

if discordant_before_restore:
    print("\nEARLY DISCORDANT EXAMPLES")
    for x in discordant_before_restore[:10]:
        print(x)


assert len(length_mismatch) == 0
assert len(prefix_mismatch) == 0
assert len(state10_mismatch) == 0
assert len(discordant) == 79
assert len(discordant_before_restore) == 0

print()
print("K10 FULL MATCHED INTERVENTION: PASS")
