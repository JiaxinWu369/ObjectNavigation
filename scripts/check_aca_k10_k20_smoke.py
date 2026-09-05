import json
from pathlib import Path


ROOMS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
]


A_ROOT = Path(
    "results/aca_smoke_early10"
)

B_ROOT = Path(
    "results/aca_smoke_early20"
)


def load_room(root, room):

    path = (
        root /
        f"episodes_{room}.jsonl"
    )

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for i, line in enumerate(f):

            if not line.strip():
                continue

            r = json.loads(line)

            # Use logged eval_index if present.
            # Otherwise deterministic file order is
            # the per-room episode order.
            idx = r.get(
                "eval_index",
                i,
            )

            rows.append(
                (int(idx), r)
            )

    return dict(rows)


def get_actions(r):

    if "full_actions" in r:
        return r["full_actions"]

    if "action_list" in r:
        return r["action_list"]

    raise KeyError(
        "No full trajectory action field."
    )


def get_states(r):

    if "full_states" in r:
        return r["full_states"]

    if "states" in r:
        return r["states"]

    raise KeyError(
        "No full trajectory state field."
    )


total = 0
reached_k10 = 0

action_prefix_mismatch = 0
state10_mismatch = 0
early_terminal_mismatch = 0
metadata_mismatch = 0
trace_prefix_mismatch = 0


for room in ROOMS:

    a = load_room(
        A_ROOT,
        room,
    )

    b = load_room(
        B_ROOT,
        room,
    )

    print()
    print("=" * 80)
    print(room)
    print("=" * 80)

    print(
        "Early10:",
        len(a),
    )

    print(
        "Early20:",
        len(b),
    )

    assert set(a) == set(b)

    for idx in sorted(a):

        ra = a[idx]
        rb = b[idx]

        total += 1

        # --------------------------------------------
        # Episode identity.
        # --------------------------------------------

        for key in [
            "scene",
            "target",
            "target_object",
            "goal_object_type",
            "start_state",
            "state_raw",
        ]:
            if key in ra and key in rb:
                if ra[key] != rb[key]:
                    metadata_mismatch += 1
                    print(
                        "metadata mismatch:",
                        room,
                        idx,
                        key,
                    )

        aa = get_actions(ra)
        ab = get_actions(rb)

        sa = get_states(ra)
        sb = get_states(rb)

        # --------------------------------------------
        # Treatment is identical for action 0..9.
        # --------------------------------------------

        npre = min(
            10,
            len(aa),
            len(ab),
        )

        if aa[:npre] != ab[:npre]:
            action_prefix_mismatch += 1
            print(
                "action prefix mismatch:",
                room,
                idx,
            )

        # --------------------------------------------
        # If either episode terminates before K10,
        # treatment never differs. Entire trajectories
        # should therefore agree.
        # --------------------------------------------

        if min(
            len(aa),
            len(ab),
        ) <= 10:

            if (
                aa != ab
                or sa != sb
            ):
                early_terminal_mismatch += 1
                print(
                    "pre-K10 terminal mismatch:",
                    room,
                    idx,
                )

            continue

        # --------------------------------------------
        # state[10] is the pre-action state for action10,
        # i.e. the matched intervention state.
        # --------------------------------------------

        if (
            len(sa) <= 10
            or len(sb) <= 10
        ):
            raise RuntimeError(
                f"{room}/{idx}: "
                "reached K10 but state[10] missing"
            )

        reached_k10 += 1

        if sa[10] != sb[10]:
            state10_mismatch += 1
            print(
                "state10 mismatch:",
                room,
                idx,
                sa[10],
                sb[10],
            )

        # --------------------------------------------
        # Optional trace sanity check:
        # compare only t=0..9.
        # Never compare action_probs at t=10 because
        # treatment differs there.
        # --------------------------------------------

        ta = ra.get(
            "context_trace"
        )

        tb = rb.get(
            "context_trace"
        )

        if ta is not None and tb is not None:

            ta_pre = [
                x
                for x in ta
                if int(x["step"]) < 10
            ]

            tb_pre = [
                x
                for x in tb
                if int(x["step"]) < 10
            ]

            if len(ta_pre) != len(tb_pre):
                trace_prefix_mismatch += 1
                continue

            for xa, xb in zip(
                ta_pre,
                tb_pre,
            ):

                for key in [
                    "step",
                    "state",
                    "goal_index",
                    "detector_scores",
                    "base_attention",
                    "semantic_kl",
                    "action",
                ]:

                    if (
                        key in xa
                        and key in xb
                        and xa[key] != xb[key]
                    ):
                        trace_prefix_mismatch += 1
                        break


print()
print("=" * 80)
print("ACA K10 / K20 MATCHED SMOKE")
print("=" * 80)

print(
    "Episodes                 :",
    total,
)

print(
    "Reached intervention K10 :",
    reached_k10,
)

print(
    "Metadata mismatch        :",
    metadata_mismatch,
)

print(
    "First-10 action mismatch :",
    action_prefix_mismatch,
)

print(
    "state[10] mismatch       :",
    state10_mismatch,
)

print(
    "Pre-K10 terminal mismatch:",
    early_terminal_mismatch,
)

print(
    "Trace-prefix mismatch    :",
    trace_prefix_mismatch,
)


assert total == 12
assert metadata_mismatch == 0
assert action_prefix_mismatch == 0
assert state10_mismatch == 0
assert early_terminal_mismatch == 0
assert trace_prefix_mismatch == 0


print()
print(
    "ACA K10/K20 MATCHED SMOKE: PASS"
)
