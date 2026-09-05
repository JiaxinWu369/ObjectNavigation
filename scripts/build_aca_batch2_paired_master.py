import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict

from scipy.stats import binomtest


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]

EARLY20_ROOT = Path(
    "results/aca_train500_batch2_early20"
)

EARLY10_ROOT = Path(
    "results/aca_train500_batch2_early10"
)

OUT_ROOT = Path(
    "results/aca_train500/batch2/paired"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


def load_logs(root, room):
    p = root / f"episodes_{room}.jsonl"

    rows = {}

    with open(
        p,
        "r",
        encoding="utf-8",
    ) as f:
        for file_idx, line in enumerate(f):

            if not line.strip():
                continue

            r = json.loads(line)

            idx = int(
                r.get(
                    "eval_index",
                    file_idx,
                )
            )

            if idx in rows:
                raise RuntimeError(
                    f"duplicate index: "
                    f"{root}/{room}/{idx}"
                )

            rows[idx] = r

    return rows


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_actions(r):
    for key in [
        "full_actions",
        "action_list",
        "actions",
    ]:
        if key in r:
            return r[key]

    raise KeyError(
        "No full action trajectory"
    )


def get_states(r):
    for key in [
        "full_states",
        "states",
    ]:
        if key in r:
            return r[key]

    raise KeyError(
        "No full state trajectory"
    )


def get_success(r):
    return int(bool(r["success"]))


def get_target(r):
    return (
        r.get("target")
        or r.get("target_object")
        or r.get("goal_object_type")
    )


counts = Counter()

room_counts = defaultdict(Counter)
target_counts = defaultdict(Counter)

master = []
discordant = []

metadata_mismatch = 0
prefix_mismatch = 0
state10_mismatch = 0
missing_state10 = 0
trace_missing = 0

total_expected = 0


for room in ROOMS:

    early20 = load_logs(
        EARLY20_ROOT,
        room,
    )

    early10 = load_logs(
        EARLY10_ROOT,
        room,
    )

    source_specs = load_pickle(
        Path(
            "test_val_split"
        )
        / f"{room}_aca_train2_22.pkl"
    )

    eligible_specs = load_pickle(
        Path(
            "test_val_split"
        )
        / f"{room}_aca_train2_k10_22.pkl"
    )

    print()
    print("=" * 90)
    print(room)
    print("=" * 90)

    print(
        "Early20 source :",
        len(early20),
    )

    print(
        "Early10 elig.  :",
        len(early10),
    )

    print(
        "Eligible specs :",
        len(eligible_specs),
    )

    assert len(early20) == 2000
    assert len(source_specs) == 2000

    assert (
        len(early10)
        ==
        len(eligible_specs)
    )

    total_expected += len(
        eligible_specs
    )

    for filtered_idx in range(
        len(eligible_specs)
    ):

        spec = eligible_specs[
            filtered_idx
        ]

        source_idx = int(
            spec[
                "aca_source_eval_index"
            ]
        )

        if source_idx not in early20:
            raise RuntimeError(
                f"{room}/{filtered_idx}: "
                f"Early20 source index "
                f"{source_idx} missing"
            )

        if filtered_idx not in early10:
            raise RuntimeError(
                f"{room}/{filtered_idx}: "
                "Early10 filtered index missing"
            )

        r20 = early20[source_idx]
        r10 = early10[filtered_idx]

        source_spec = source_specs[
            source_idx
        ]

        # ------------------------------------------
        # Verify filtered spec came from exactly
        # this original source episode.
        # ------------------------------------------

        for key in [
            "scene",
            "state",
            "goal_object_type",
        ]:
            if (
                str(spec[key])
                !=
                str(source_spec[key])
            ):
                raise RuntimeError(
                    "Eligible/source spec mismatch: "
                    f"{room}/"
                    f"{filtered_idx}/"
                    f"{source_idx}/"
                    f"{key}"
                )

        expected_scene = spec["scene"]
        expected_target = spec[
            "goal_object_type"
        ]

        scene10 = r10.get("scene")
        scene20 = r20.get("scene")

        target10 = get_target(r10)
        target20 = get_target(r20)

        if (
            scene10 != expected_scene
            or scene20 != expected_scene
            or target10 != expected_target
            or target20 != expected_target
        ):
            metadata_mismatch += 1

            raise RuntimeError(
                "Episode identity mismatch: "
                f"{room}/"
                f"{filtered_idx}/"
                f"{source_idx}"
            )

        a10 = get_actions(r10)
        a20 = get_actions(r20)

        s10 = get_states(r10)
        s20 = get_states(r20)

        # ------------------------------------------
        # These were selected because Early20
        # executed action index 10.
        # ------------------------------------------

        if len(a20) <= 10:
            raise RuntimeError(
                "Eligibility violation in Early20: "
                f"{room}/{source_idx}"
            )

        # Under identical pre-treatment history,
        # Early10 must also reach the decision
        # state s_10.
        if len(a10) <= 10:
            raise RuntimeError(
                "Early10 did not reach K10 "
                "despite matched Early20 eligibility: "
                f"{room}/{filtered_idx}"
            )

        # ------------------------------------------
        # Action 0..9 must be identical.
        # ------------------------------------------

        if a10[:10] != a20[:10]:
            prefix_mismatch += 1

            raise RuntimeError(
                "First-10 action mismatch: "
                f"{room}/"
                f"{filtered_idx}/"
                f"{source_idx}"
            )

        if (
            len(s10) <= 10
            or len(s20) <= 10
        ):
            missing_state10 += 1

            raise RuntimeError(
                "state[10] missing: "
                f"{room}/"
                f"{filtered_idx}/"
                f"{source_idx}"
            )

        if s10[10] != s20[10]:
            state10_mismatch += 1

            raise RuntimeError(
                "state[10] mismatch: "
                f"{room}/"
                f"{filtered_idx}/"
                f"{source_idx}"
            )

        y10 = get_success(r10)
        y20 = get_success(r20)

        pair_state = (
            f"N{y10}{y20}"
        )

        counts[pair_state] += 1

        room_counts[
            room
        ][pair_state] += 1

        target_counts[
            expected_target
        ][pair_state] += 1

        trace = r20.get(
            "context_trace"
        )

        if trace is None:
            trace_missing += 1

        row = {
            "batch":
                2,

            "room":
                room,

            "early10_eval_index":
                filtered_idx,

            "early20_source_eval_index":
                source_idx,

            "aca_episode_id":
                spec.get(
                    "aca_episode_id"
                ),

            "scene":
                expected_scene,

            "target":
                expected_target,

            "start_state":
                str(spec["state"]),

            "reached_k10":
                True,

            "early10_success":
                y10,

            "early20_success":
                y20,

            "pair_state":
                pair_state,

            "early10_length":
                len(a10),

            "early20_length":
                len(a20),

            # Selector features must be extracted
            # only from the delayed arm.
            "context_trace":
                trace,
        }

        master.append(row)

        if y10 != y20:

            pref = dict(row)

            pref[
                "restore_now_label"
            ] = int(
                y10 == 1
                and y20 == 0
            )

            discordant.append(
                pref
            )


assert total_expected == 4431
assert len(master) == 4431
assert sum(counts.values()) == 4431


n = len(master)

n00 = counts["N00"]
n01 = counts["N01"]
n10 = counts["N10"]
n11 = counts["N11"]

nd = n01 + n10

sr10 = (
    n10 + n11
) / n

sr20 = (
    n01 + n11
) / n

oracle = (
    n01 + n10 + n11
) / n

best_fixed = max(
    sr10,
    sr20,
)

headroom = (
    oracle
    - best_fixed
)

p_exact = (
    binomtest(
        n10,
        nd,
        p=0.5,
        alternative="two-sided",
    ).pvalue
    if nd
    else 1.0
)


all_path = (
    OUT_ROOT
    / "all_4431.jsonl"
)

with open(
    all_path,
    "w",
    encoding="utf-8",
) as f:

    for r in master:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


pref_path = (
    OUT_ROOT
    / "discordant_preference.jsonl"
)

with open(
    pref_path,
    "w",
    encoding="utf-8",
) as f:

    for r in discordant:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


summary = {
    "batch": 2,
    "eligible": n,

    "N00": n00,
    "N01_delay_only": n01,
    "N10_restore_only": n10,
    "N11": n11,

    "discordant": nd,
    "discordant_rate": nd / n,

    "early10_sr": sr10,
    "early20_sr": sr20,

    "early10_minus_early20_pp":
        100 * (sr10 - sr20),

    "paired_exact_p":
        p_exact,

    "oracle_sr":
        oracle,

    "adaptive_headroom_pp":
        100 * headroom,

    "trace_missing":
        trace_missing,

    "metadata_mismatch":
        metadata_mismatch,

    "prefix_mismatch":
        prefix_mismatch,

    "state10_mismatch":
        state10_mismatch,

    "missing_state10":
        missing_state10,
}


with open(
    OUT_ROOT / "summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=" * 90)
print("ACA BATCH2 PAIRED MASTER")
print("=" * 90)

print(
    "Eligible pairs       :",
    n,
)

print()
print(
    "N00 both fail        :",
    n00,
)

print(
    "N01 Delay only       :",
    n01,
)

print(
    "N10 RestoreNow only  :",
    n10,
)

print(
    "N11 both success     :",
    n11,
)

print(
    "Discordant           :",
    nd,
)

print(
    "Discord rate         :",
    f"{100*nd/n:.3f}%"
)

print()
print(
    "Early10 SR           :",
    f"{100*sr10:.3f}%"
)

print(
    "Early20 SR           :",
    f"{100*sr20:.3f}%"
)

print(
    "Early10 - Early20    :",
    f"{100*(sr10-sr20):+.3f} pp"
)

print(
    "Exact paired p       :",
    f"{p_exact:.6g}"
)

print(
    "Oracle SR            :",
    f"{100*oracle:.3f}%"
)

print(
    "Adaptive headroom    :",
    f"{100*headroom:.3f} pp"
)

print()
print(
    "Trace missing        :",
    trace_missing,
)

print(
    "Metadata mismatch    :",
    metadata_mismatch,
)

print(
    "Prefix mismatch      :",
    prefix_mismatch,
)

print(
    "state[10] mismatch   :",
    state10_mismatch,
)

print(
    "Missing state[10]    :",
    missing_state10,
)


print()
print("=" * 90)
print("BY TARGET")
print("=" * 90)

print(
    f"{'Target':<18}"
    f"{'N':>6}"
    f"{'Restore':>10}"
    f"{'Delay':>10}"
    f"{'Net':>8}"
    f"{'Discord':>10}"
)

print("-" * 62)

for target in sorted(
    target_counts
):

    c = target_counts[target]

    nn = sum(c.values())

    restore = c["N10"]
    delay = c["N01"]

    print(
        f"{target:<18}"
        f"{nn:>6d}"
        f"{restore:>10d}"
        f"{delay:>10d}"
        f"{restore-delay:>8d}"
        f"{restore+delay:>10d}"
    )


print()
print("=" * 90)
print("BY ROOM")
print("=" * 90)

for room in ROOMS:

    c = room_counts[room]

    print(
        f"{room:<14}"
        f"N00={c['N00']:4d}  "
        f"N01={c['N01']:4d}  "
        f"N10={c['N10']:4d}  "
        f"N11={c['N11']:4d}"
    )


print()
print(
    "All dataset :",
    all_path,
)

print(
    "Preference  :",
    pref_path,
)

print()
print(
    "ACA BATCH2 PAIRED MASTER: PASS"
)
