import json
from glob import glob
import numpy as np


BASE = "results/base500_zsval"
LEARNED = "results/lcr_learned500_zsval"

SEED = 2026
N_BOOT = 20000


def room(path):
    name = path.split("/")[-1]
    return name[len("episodes_"):-len(".jsonl")]


def load(root):
    data = {}

    for path in glob(root + "/episodes_*.jsonl"):
        rname = room(path)

        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                x = json.loads(line)

                key = (
                    rname,
                    int(x["eval_index"]),
                )

                data[key] = x

    return data


def get_length(x):
    for k in [
        "ep_length",
        "episode_length",
        "length",
    ]:
        if k in x:
            return float(x[k])

    raise KeyError(
        "No episode length key: "
        + str(sorted(x.keys()))
    )


base = load(BASE)
learned = load(LEARNED)

assert set(base) == set(learned)
assert len(base) == 1260

keys = sorted(base)

b_spl = np.asarray(
    [float(base[k]["spl"]) for k in keys]
)

l_spl = np.asarray(
    [float(learned[k]["spl"]) for k in keys]
)

b_len = np.asarray(
    [get_length(base[k]) for k in keys]
)

l_len = np.asarray(
    [get_length(learned[k]) for k in keys]
)

d_spl = l_spl - b_spl
d_len = l_len - b_len

rng = np.random.default_rng(SEED)

boot_spl = np.empty(N_BOOT)
boot_len = np.empty(N_BOOT)

n = len(keys)

for i in range(N_BOOT):
    idx = rng.integers(
        0,
        n,
        size=n,
    )

    boot_spl[i] = d_spl[idx].mean()
    boot_len[i] = d_len[idx].mean()


def report(name, raw, boot, scale=1.0):

    estimate = raw.mean() * scale

    lo, hi = (
        np.percentile(
            boot * scale,
            [2.5, 97.5],
        )
    )

    print(name)
    print(
        "  estimate : {:+.6f}".format(
            estimate
        )
    )
    print(
        "  95% CI   : [{:+.6f}, {:+.6f}]".format(
            lo,
            hi,
        )
    )

    if lo > 0:
        direction = "POSITIVE"
    elif hi < 0:
        direction = "NEGATIVE"
    else:
        direction = "INCLUDES ZERO"

    print(
        "  result   :",
        direction,
    )


print("=" * 72)
print("BASE vs LEARNED-LCR PAIRED BOOTSTRAP")
print("=" * 72)

report(
    "Delta SPL (percentage points)",
    d_spl,
    boot_spl,
    scale=100.0,
)

print()

report(
    "Delta episode length (steps)",
    d_len,
    boot_len,
    scale=1.0,
)
