from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path
import numpy as np

# Ensure repo root on path when running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from har.transforms.stats_io import load_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_glob", type=str, required=True)
    ap.add_argument("--stats", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--mean_tol", type=float, default=0.05, help="allowed abs mean after norm")
    ap.add_argument("--std_tol", type=float, default=0.05, help="allowed |std-1| after norm")
    args = ap.parse_args()

    files = sorted(glob.glob(args.shards_glob))
    if not files:
        raise SystemExit("No shards matched.")

    stats = load_stats(args.stats)

    total = 0
    sum_per_ch = None
    sumsq_per_ch = None
    T_accum = 0

    for f in files:
        z = np.load(f, allow_pickle=False)
        X = z["X"].astype(np.float32)
        if "split" in z.files:
            m = np.array(z["split"]) == args.split
            if not m.any():
                continue
            X = X[m]
        # Apply normalization per sample
        # X: (N,C,T)
        Xn = (X - stats.mean[None, :, None]) / stats.std[None, :, None]
        N, C, T = Xn.shape
        if args.limit and total + N > args.limit:
            keep = args.limit - total
            if keep <= 0:
                break
            Xn = Xn[:keep]
            N = keep
        # accumulate moments across N and T
        if sum_per_ch is None:
            sum_per_ch = Xn.sum(axis=(0, 2))
            sumsq_per_ch = (Xn ** 2).sum(axis=(0, 2))
        else:
            sum_per_ch += Xn.sum(axis=(0, 2))
            sumsq_per_ch += (Xn ** 2).sum(axis=(0, 2))
        total += N
        T_accum += N * T
        if args.limit and total >= args.limit:
            break

    if total == 0:
        raise SystemExit("No samples after filtering.")

    mean = sum_per_ch / T_accum
    var = sumsq_per_ch / T_accum - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))

    mean_err = np.abs(mean)
    std_err = np.abs(std - 1.0)

    print(f"Checked {total} samples; channels={len(mean)}")
    print("Per-channel max |mean|:", float(mean_err.max()))
    print("Per-channel max |std-1|:", float(std_err.max()))

    ok = (mean_err.max() <= args.mean_tol) and (std_err.max() <= args.std_tol)
    if not ok:
        print("FAILED: normalization outside tolerances.")
        print("mean per channel:", mean.astype(np.float32))
        print("std per channel:", std.astype(np.float32))
        raise SystemExit(1)
    print("PASSED: normalization within tolerances.")


if __name__ == "__main__":
    main()


