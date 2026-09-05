import json
from pathlib import Path
from collections import Counter, defaultdict


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

# AKGVP action order
MOVE = 0
ROT_L = 1
ROT_R = 2
LOOK_D = 3
LOOK_U = 4
DONE = 5


def load(root):
    root = Path(root)
    rows = {}

    for room in ROOMS:
        p = root / f"episodes_{room}.jsonl"
        if not p.exists():
            return None

        for line in open(p):
            if not line.strip():
                continue

            x = json.loads(line)

            key = (
                x.get("scene_type", room),
                int(x["eval_index"]),
            )

            rows[key] = x

    return rows


def discover_run(
    success_target,
    n_expected,
):
    hits = []

    for p in Path("results").rglob("metrics.json"):
        try:
            m = json.load(open(p))
        except Exception:
            continue

        if abs(
            float(m.get("success", -999))
            - success_target
        ) > 2e-6:
            continue

        root = p.parent
        rows = load(root)

        if rows is not None and len(rows) == n_expected:
            hits.append(root)

    return hits


def parse_state(s):
    a = s.split("|")

    return (
        round(float(a[0]), 3),
        round(float(a[1]), 3),
        int(float(a[2])),
        int(float(a[3])),
    )


def behavior(rec):

    actions = rec["full_actions"]
    states = rec["full_states"]

    parsed = [
        parse_state(s)
        for s in states
    ]

    room = rec["scene_type"]

    max_steps = (
        200
        if room == "living_room"
        else 100
    )

    n = max(len(actions), 1)

    positions = [
        (s[0], s[1])
        for s in parsed
    ]

    unique_states = len(set(parsed))
    unique_pos = len(set(positions))

    repeat_ratio = (
        1.0
        - unique_states / max(len(parsed), 1)
    )

    move_n = sum(
        a == MOVE
        for a in actions
    )

    rot_n = sum(
        a in (ROT_L, ROT_R)
        for a in actions
    )

    # MoveAhead but state does not change
    stalled_moves = 0

    for t, a in enumerate(actions):

        if a != MOVE:
            continue

        if t + 1 < len(states):
            if states[t + 1] == states[t]:
                stalled_moves += 1

    stall_ratio = (
        stalled_moves / move_n
        if move_n
        else 0.0
    )

    state_counts = Counter(parsed)

    max_state_visits = (
        max(state_counts.values())
        if state_counts
        else 0
    )

    dts = float(
        rec.get("dts", float("nan"))
    )

    failed = not bool(rec["success"])

    premature_done = (
        failed
        and len(actions) > 0
        and actions[-1] == DONE
        and len(actions) < max_steps
    )

    timeout = (
        failed
        and len(actions) >= max_steps
    )

    blocked = (
        failed
        and stalled_moves >= 5
        and stall_ratio >= 0.30
    )

    spin_loop = (
        failed
        and rot_n / n >= 0.60
        and unique_pos <= 5
    )

    local_cycle = (
        failed
        and (
            repeat_ratio >= 0.45
            or max_state_visits >= 5
        )
    )

    near_goal = (
        failed
        and dts <= 1.5
    )

    # mutually exclusive primary label
    if premature_done:
        primary = "premature_done"
    elif blocked:
        primary = "blocked"
    elif spin_loop:
        primary = "rotation_loop"
    elif local_cycle:
        primary = "local_cycle"
    elif timeout and near_goal:
        primary = "timeout_near_goal"
    elif timeout:
        primary = "timeout_far"
    elif near_goal:
        primary = "near_goal_other"
    else:
        primary = "other"

    return {
        "primary": primary,
        "dts": dts,
        "steps": len(actions),
        "unique_pos": unique_pos,
        "repeat_ratio": repeat_ratio,
        "move_n": move_n,
        "rot_n": rot_n,
        "stalled_moves": stalled_moves,
        "stall_ratio": stall_ratio,
        "max_state_visits": max_state_visits,
    }


def first_divergence(a, b):

    A = a["full_states"]
    B = b["full_states"]

    AA = a["full_actions"]
    BB = b["full_actions"]

    n = min(
        len(A),
        len(B),
        len(AA),
        len(BB),
    )

    for t in range(n):

        if (
            A[t] != B[t]
            or AA[t] != BB[t]
        ):
            return t

    if (
        len(A) != len(B)
        or len(AA) != len(BB)
    ):
        return n

    return None


def analyze(
    title,
    base_root,
    ours_root,
):

    base = load(base_root)
    ours = load(ours_root)

    assert base is not None
    assert ours is not None
    assert set(base) == set(ours)

    print()
    print("=" * 92)
    print(title)
    print("=" * 92)

    pair_counts = Counter()

    modes = Counter()
    modes_both_fail = Counter()
    modes_harm = Counter()

    target_fail = defaultdict(Counter)
    room_fail = defaultdict(Counter)
    scene_fail = defaultdict(Counter)

    dts_by_mode = defaultdict(list)

    harms = []
    common_fail = []

    for key in sorted(base):

        a = base[key]
        b = ours[key]

        sa = bool(a["success"])
        sb = bool(b["success"])

        if not sa and not sb:
            pair = "both_fail"
        elif not sa and sb:
            pair = "repair"
        elif sa and not sb:
            pair = "harm"
        else:
            pair = "both_success"

        pair_counts[pair] += 1

        if not sb:

            z = behavior(b)
            mode = z["primary"]

            modes[mode] += 1
            dts_by_mode[mode].append(
                z["dts"]
            )

            target_fail[
                b["target"]
            ][mode] += 1

            room_fail[
                b["scene_type"]
            ][mode] += 1

            scene_fail[
                b["scene"]
            ][mode] += 1

            item = {
                "key": key,
                "scene": b["scene"],
                "target": b["target"],
                "mode": mode,
                "dts": z["dts"],
                "steps": z["steps"],
                "unique_pos": z["unique_pos"],
                "repeat_ratio": z["repeat_ratio"],
                "stalled_moves": z["stalled_moves"],
                "diverge":
                    first_divergence(a, b),
            }

            if pair == "harm":
                modes_harm[mode] += 1
                harms.append(item)

            if pair == "both_fail":
                modes_both_fail[mode] += 1
                common_fail.append(item)

    print("Base root :", base_root)
    print("Ours root :", ours_root)
    print("episodes  :", len(base))
    print()

    print("PAIR OUTCOMES")
    for k in [
        "both_fail",
        "repair",
        "harm",
        "both_success",
    ]:
        print(
            f"{k:<15}:",
            pair_counts[k],
        )

    print()
    print("OURS FAILURE MODES")
    for mode, n in modes.most_common():
        print(
            f"{mode:<22}",
            f"{n:>5}",
        )

    print()
    print("BOTH-FAIL MODES")
    for mode, n in modes_both_fail.most_common():
        print(
            f"{mode:<22}",
            f"{n:>5}",
        )

    print()
    print("HARM MODES")
    for mode, n in modes_harm.most_common():
        print(
            f"{mode:<22}",
            f"{n:>5}",
        )

    print()
    print("TARGET FAILURE COUNTS")
    for target in sorted(target_fail):

        c = target_fail[target]

        print(
            f"{target:<15}",
            f"total={sum(c.values()):>4}",
            " | ",
            ", ".join(
                f"{k}:{v}"
                for k, v in c.most_common()
            ),
        )

    print()
    print("ROOM FAILURE COUNTS")
    for room in sorted(room_fail):

        c = room_fail[room]

        print(
            f"{room:<15}",
            f"total={sum(c.values()):>4}",
            " | ",
            ", ".join(
                f"{k}:{v}"
                for k, v in c.most_common()
            ),
        )

    print()
    print("TOP FAILURE SCENES")

    ranked = sorted(
        scene_fail.items(),
        key=lambda x:
            -sum(x[1].values())
    )

    for scene, c in ranked[:15]:
        print(
            f"{scene:<15}",
            f"total={sum(c.values()):>4}",
            " | ",
            ", ".join(
                f"{k}:{v}"
                for k, v in c.most_common()
            ),
        )

    print()
    print("HARMS")
    for x in harms:
        print(x)

    print()
    print("SAMPLE COMMON FAILURES")
    for x in common_fail[:30]:
        print(x)


# -------- locate VAL runs automatically --------

val_base = discover_run(
    0.7309523809523809,
    1260,
)

val_ours = discover_run(
    0.7428571428571429,
    1260,
)

print("VAL BASE candidates:", val_base)
print("VAL OURS candidates:", val_ours)

if not val_base:
    raise RuntimeError(
        "Could not locate complete Base Val run"
    )

if not val_ours:
    raise RuntimeError(
        "Could not locate complete Ours Val run"
    )

analyze(
    "VALIDATION FAILURE ANALYSIS",
    val_base[0],
    val_ours[0],
)

analyze(
    "TEST FAILURE ANALYSIS",
    Path("results/final_test_base500"),
    Path("results/guard_tau02_zstest"),
)
