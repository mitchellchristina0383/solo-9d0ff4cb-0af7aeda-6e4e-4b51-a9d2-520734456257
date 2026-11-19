#!/usr/bin/env python3
"""Generate the "later" sample checkpoint (defaults to ./sample/next).

Pairs with ./sample/source (see make_sample.py) to demo POST /diagnose:
same total_length, a later step, deliberately DIFFERENT shard splits for
both tensors, and a handful of values flipped from finite to NaN/Inf:

    weights only  : idx 1234 -> NaN,  idx 9876 -> +inf
    momentum only : idx 55   -> NaN,  idx 7000 -> -inf
    both          : idx 5000 -> weights NaN, momentum +inf

Base values follow the same pattern as the source sample
(weights[i] = float(i), momentum[i] = -0.5*i), so the expected diagnose
result is exactly the table above.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import sys
from pathlib import Path

DEFAULT_STEP = 7460  # source sample is 7459
WEIGHT_SPLITS = (0, 4000, 8000)        # source weights: (0, 3500, 7200)
MOMENTUM_SPLITS = (0, 1500, 6000, 9500)  # source momentum: (0, 2000, 9000)

WEIGHT_ANOMALIES = {1234: math.nan, 9876: math.inf, 5000: math.nan}
MOMENTUM_ANOMALIES = {55: math.nan, 7000: -math.inf, 5000: math.inf}


def write_parts(dest: Path, subdir: str, splits: tuple[int, ...], total: int,
                values: list[float]) -> list[dict]:
    (dest / subdir).mkdir(parents=True, exist_ok=True)
    segments = []
    bounds = list(splits) + [total]
    for idx, (start, end) in enumerate(zip(bounds, bounds[1:])):
        vals = array.array("f", values[start:end])
        if sys.byteorder == "big":
            vals.byteswap()
        path = f"{subdir}/part-{idx}.f32"
        (dest / path).write_bytes(vals.tobytes())
        segments.append({"path": path, "start": start, "end": end})
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", nargs="?", default=str(Path(__file__).parent / "next"))
    parser.add_argument("--source", default=str(Path(__file__).parent / "source"),
                        help="earlier sample checkpoint (for total_length)")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    args = parser.parse_args()

    source_manifest = json.loads((Path(args.source) / "manifest.json").read_text())
    total = int(source_manifest["total_length"])
    if max(max(WEIGHT_ANOMALIES), max(MOMENTUM_ANOMALIES)) >= total:
        raise SystemExit(f"anomaly indices do not fit total_length={total}")

    weights = [float(i) for i in range(total)]
    momentum = [-0.5 * i for i in range(total)]
    for idx, val in WEIGHT_ANOMALIES.items():
        weights[idx] = val
    for idx, val in MOMENTUM_ANOMALIES.items():
        momentum[idx] = val

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "step": args.step,
        "total_length": total,
        "weights": write_parts(dest, "weights", WEIGHT_SPLITS, total, weights),
        "momentum": write_parts(dest, "momentum", MOMENTUM_SPLITS, total, momentum),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"later sample checkpoint written to {dest} "
          f"(step={args.step}, total_length={total}, "
          f"anomalies: weights-only 2, momentum-only 2, both 1)")


if __name__ == "__main__":
    main()
