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

EARLY10_ROOT = Path(
    "results/aca_train500_early10"
)

EARLY20_ROOT = Path(
    "results/aca_train500_early20"
)

OUT_ROOT = Path(
    "results/aca_train500/paired"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


def load_run(root, room):

    p = (
        root
        / f"episodes_{room}.jsonl"
    )

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
                    f"duplicate eval_index "
                    f"{room}/{idx}"
                )

            rows[idx] = r

    return rows


def load_specs(room):

    p = Path(
        f"test_val_split/"
        f"{room}_aca_train_22.pkl"
    )

    with open(p, "rb") as f:
        data = pickle.load(f)

    return data


def get_actions(r):

    for key in [
        "full_actions",
        "action_list",
        "actions",
    ]:
        if key in r:
            return r[key]

    raise KeyError(
        "No full action trajectory."
    )


def get_states(r):

    for key in [
        "full_states",
        "states",
    ]:
        if key in r:
            return r[key]

    raise KeyError(
        "No full state trajectory."
    )


def get_success(r):
    return int(
        bool(r["success"])
    )


counts = Counter()
target_counts = defaultdict(Counter)
room_counts = defaultdict(Counter)

master = []
discordant = []

metadata_mismatch = 0
prefix_mismatch = 0
state10_mismatch = 0
early_end_mismatch = 0

reached10 = 0
terminated_by10 = 0

trace_available = 0
trace_reached10 = 0


for room in ROOMS:

    a10 = load_run(
        EARLY10_ROOT,
        room,
    )

    a20 = load_run(
        EARLY20_ROOT,
        room,
    )

    specs = load_specs(room)

    print()
    print("=" * 90)
    print(room)
    print("=" * 90)

    print(
        "Early10:",
        len(a10),
    )

    print(
        "Early20:",
        len(a20),
    )

    print(
        "Specs  :",
        len(specs),
    )

    assert len(a10) == 1000
    assert len(a20) == 1000
    assert len(specs) == 1000

    assert set(a10) == set(a20)

    for idx in sorted(a10):

        r10 = a10[idx]
        r20 = a20[idx]
        spec = specs[idx]

        # ----------------------------------------
        # Identity check against both arms and
        # original episode specification.
        # ----------------------------------------

        scene10 = r10.get("scene")
        scene20 = r20.get("scene")

        target10 = (
            r10.get("target")
            or r10.get("target_object")
            or r10.get("goal_object_type")
        )

        target20 = (
            r20.get("target")
            or r20.get("target_object")
            or r20.get("goal_object_type")
        )

        expected_scene = spec["scene"]
        expected_target = spec[
            "goal_object_type"
        ]

        if (
            scene10 != scene20
            or scene10 != expected_scene
            or target10 != target20
            or target10 != expected_target
        ):
            metadata_mismatch += 1

            raise RuntimeError(
                "Episode identity mismatch: "
                f"{room}/{idx}\n"
                f"expected="
                f"{expected_scene}/"
                f"{expected_target}\n"
                f"early10="
                f"{scene10}/"
                f"{target10}\n"
                f"early20="
                f"{scene20}/"
                f"{target20}"
            )

        aa = get_actions(r10)
        ab = get_actions(r20)

        sa = get_states(r10)
        sb = get_states(r20)

        y10 = get_success(r10)
        y20 = get_success(r20)

        # ----------------------------------------
        # Actions 0..9 are under identical
        # treatment.
        # ----------------------------------------

        npre = min(
            10,
            len(aa),
            len(ab),
        )

        if aa[:npre] != ab[:npre]:
            prefix_mismatch += 1

            raise RuntimeError(
                f"Prefix mismatch: "
                f"{room}/{idx}"
            )

        # ----------------------------------------
        # If either arm has no action index 10,
        # intervention was never applied.
        #
        # len(actions) <= 10 means only
        # a0...a9 (or fewer) were executed.
        # ----------------------------------------

        reaches = (
            len(aa) > 10
            and len(ab) > 10
        )

        if not reaches:

            terminated_by10 += 1

            if (
                aa != ab
                or sa != sb
                or y10 != y20
            ):
                early_end_mismatch += 1

                raise RuntimeError(
                    "Pre-intervention termination "
                    f"mismatch: {room}/{idx}"
                )

        else:

            reached10 += 1

            if (
                len(sa) <= 10
                or len(sb) <= 10
            ):
                raise RuntimeError(
                    f"state[10] missing: "
                    f"{room}/{idx}"
                )

            if sa[10] != sb[10]:
                state10_mismatch += 1

                raise RuntimeError(
                    f"state[10] mismatch: "
                    f"{room}/{idx}"
                )

        # ----------------------------------------
        # Preference outcome.
        #
        # state string follows:
        # N<early10><early20>
        #
        # N10 = RestoreNow only
        # N01 = Delay only
        # ----------------------------------------

        state = f"N{y10}{y20}"

        counts[state] += 1
        target_counts[
            expected_target
        ][state] += 1

        room_counts[
            room
        ][state] += 1

        trace = r20.get(
            "context_trace"
        )

        if trace is not None:
            trace_available += 1

            if reaches:
                trace_reached10 += 1

        row = {
            "room":
                room,

            "eval_index":
                idx,

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
                bool(reaches),

            "early10_success":
                y10,

            "early20_success":
                y20,

            "pair_state":
                state,

            "early10_length":
                len(aa),

            "early20_length":
                len(ab),

            # All selector features must later
            # come from this Early20 trace.
            "context_trace":
                trace,
        }

        master.append(row)

        # Only treatment-exposed discordant
        # episodes are valid preference labels.
        if y10 != y20:

            if not reaches:
                raise RuntimeError(
                    "Discordant pair did not "
                    "reach intervention: "
                    f"{room}/{idx}"
                )

            row_pref = dict(row)

            row_pref[
                "restore_now_label"
            ] = int(
                y10 == 1
                and y20 == 0
            )

            discordant.append(
                row_pref
            )


assert len(master) == 4000
assert sum(counts.values()) == 4000


# ------------------------------------------------
# Statistics
# ------------------------------------------------

n00 = counts["N00"]
n01 = counts["N01"]
n10 = counts["N10"]
n11 = counts["N11"]

discordant_n = n01 + n10

sr10 = (
    n10 + n11
) / 4000.0

sr20 = (
    n01 + n11
) / 4000.0

union = (
    n01
    + n10
    + n11
) / 4000.0

best_fixed = max(
    sr10,
    sr20,
)

headroom = (
    union
    - best_fixed
)

if discordant_n:

    p = binomtest(
        n10,
        discordant_n,
        p=0.5,
        alternative="two-sided",
    ).pvalue

else:
    p = 1.0


# ------------------------------------------------
# Save
# ------------------------------------------------

all_path = (
    OUT_ROOT
    / "all_4000.jsonl"
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
    "episodes": 4000,
    "reached_k10": reached10,
    "terminated_by_k10": terminated_by10,

    "N00": n00,
    "N01_delay_only": n01,
    "N10_restore_only": n10,
    "N11": n11,

    "discordant": discordant_n,

    "early10_sr": sr10,
    "early20_sr": sr20,

    "early10_minus_early20_pp":
        100.0 * (sr10 - sr20),

    "paired_exact_p": p,

    "oracle_union_sr": union,

    "adaptive_headroom_pp":
        100.0 * headroom,

    "trace_available":
        trace_available,

    "trace_reached_k10":
        trace_reached10,

    "metadata_mismatch":
        metadata_mismatch,

    "prefix_mismatch":
        prefix_mismatch,

    "state10_mismatch":
        state10_mismatch,

    "early_end_mismatch":
        early_end_mismatch,
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
print("ACA PAIRED MASTER DATASET")
print("=" * 90)

print(
    "Total episodes       :",
    len(master),
)

print(
    "Reached K10          :",
    reached10,
)

print(
    "Terminated by K10    :",
    terminated_by10,
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
    discordant_n,
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
    f"{p:.6g}"
)

print(
    "Oracle union SR      :",
    f"{100*union:.3f}%"
)

print(
    "Adaptive headroom    :",
    f"{100*headroom:.3f} pp"
)

print()
print(
    "Early20 traces       :",
    trace_available,
)

print(
    "Trace + reached K10  :",
    trace_reached10,
)

print()
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
    "Early-end mismatch   :",
    early_end_mismatch,
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

    n = sum(c.values())

    restore = c["N10"]
    delay = c["N01"]

    print(
        f"{target:<18}"
        f"{n:>6d}"
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
    "ACA PAIRED MASTER: PASS"
)
