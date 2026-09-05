import json


FILES = {
    "Base":
        "results/base500_zsval/"
        "metrics.json",

    "NoContext":
        "results/nocontext500_zsval/"
        "metrics.json",

    "Rule-LCR":
        "results/lcr_rule500_zsval/"
        "metrics.json",

    "Learned-LCR":
        "results/lcr_learned500_zsval/"
        "metrics.json",
}


for name, path in FILES.items():

    try:
        with open(
            path,
            encoding="utf-8",
        ) as f:
            m = json.load(f)

    except FileNotFoundError:
        print(
            name,
            "NOT FOUND:",
            path,
        )
        continue

    print(
        "{:<14} "
        "SR={:>7.3f} "
        "SPL={:>7.3f} "
        "Length={:>7.3f} "
        "Done={:>7.3f}".format(
            name,
            100.0
            * float(
                m["success"]
            ),
            100.0
            * float(
                m["spl"]
            ),
            float(
                m["ep_length"]
            ),
            100.0
            * float(
                m["done_count"]
            ),
        )
    )
