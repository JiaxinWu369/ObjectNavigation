import json
from glob import glob
from collections import defaultdict

import numpy as np


ROOTS = {
    "Base": "results/base500_zsval",
    "Rule": "results/lcr_rule500_zsval",
    "Learned": "results/lcr_learned500_zsval",
}


def room_from_path(path):
    name = path.split("/")[-1]
    return name[
        len("episodes_"):
        -len(".jsonl")
    ]


def load(root):
    rows = {}

    for path in sorted(
        glob(
            root
            + "/episodes_*.jsonl"
        )
    ):
        room = room_from_path(path)

        with open(
            path,
            encoding="utf-8",
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                r = json.loads(line)

                key = (
                    room,
                    int(
                        r["eval_index"]
                    ),
                )

                rows[key] = r

    return rows


def get_target(r):

    for k in [
        "target",
        "target_object",
        "goal",
        "goal_object_type",
    ]:
        if k in r:
            return str(r[k])

    return "UNKNOWN"


data = {
    name: load(root)
    for name, root
    in ROOTS.items()
}

keys = sorted(
    data["Base"]
)

assert len(keys) == 1260

assert (
    set(keys)
    ==
    set(data["Rule"])
    ==
    set(data["Learned"])
)


def print_group(
    title,
    group_fn,
):

    groups = defaultdict(list)

    for key in keys:
        groups[
            group_fn(
                key,
                data["Base"][key]
            )
        ].append(key)

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(
        "{:<18} {:>5} "
        "{:>10} {:>10} {:>10} "
        "{:>11} {:>11}".format(
            "Group",
            "N",
            "Base SR",
            "Rule SR",
            "Learn SR",
            "L-B SPL",
            "L-R SPL",
        )
    )

    for group in sorted(groups):

        ks = groups[group]

        def sr(method):
            return np.mean(
                [
                    float(
                        bool(
                            data[
                                method
                            ][k][
                                "success"
                            ]
                        )
                    )
                    for k in ks
                ]
            )

        def spl(method):
            return np.mean(
                [
                    float(
                        data[
                            method
                        ][k]["spl"]
                    )
                    for k in ks
                ]
            )

        base_sr = sr("Base")
        rule_sr = sr("Rule")
        learned_sr = sr(
            "Learned"
        )

        base_spl = spl("Base")
        rule_spl = spl("Rule")
        learned_spl = spl(
            "Learned"
        )

        print(
            "{:<18} {:>5d} "
            "{:>9.2f}% {:>9.2f}% "
            "{:>9.2f}% "
            "{:>+10.3f} "
            "{:>+10.3f}".format(
                str(group),
                len(ks),
                base_sr * 100,
                rule_sr * 100,
                learned_sr * 100,
                (
                    learned_spl
                    - base_spl
                )
                * 100,
                (
                    learned_spl
                    - rule_spl
                )
                * 100,
            )
        )


print_group(
    "ROOM ANALYSIS",
    lambda key, row: key[0],
)


print_group(
    "TARGET ANALYSIS",
    lambda key, row: get_target(
        row
    ),
)
