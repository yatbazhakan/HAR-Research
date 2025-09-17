from __future__ import annotations
import argparse, numpy as np, json
from pathlib import Path
import glob, sys

# Ensure repository root on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Computes normalization stats and writes an extended JSON containing:
- mean/std at the root for the requested split (backward compatible)
- splits: {split_name: {mean, std, count_samples, count_time}}
- per_subject: {split_name: {subject_id: {mean, std, count_samples, count_time}}}

expects NPZ shards with keys: X (N,C,T), y, and optionally: dataset, split, subject_id
"""
from har.transforms.normalize import NormStats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_glob", type=str, required=True,
                    help="e.g., /workspace/artifacts/preprocessed/mhealth/*.npz")
    ap.add_argument("--split", type=str, default="train",
                    help="filter by split if present (else uses all)")
    ap.add_argument("--out", type=str, required=True,
                    help="where to write stats json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.shards_glob))
    if not files:
        raise SystemExit("No shards matched.")

    # Accumulators: sums over (N,T) for each channel
    # root (requested split)
    root_sum = None
    root_sumsq = None
    root_T = 0
    root_N = 0

    # per split and per subject
    split_acc = {}  # split -> {sum, sumsq, T, N}
    subject_acc = {}  # split -> {subject_id -> {sum, sumsq, T, N}}

    def ensure_split(name, C):
        if name not in split_acc:
            split_acc[name] = {"sum": np.zeros(C, np.float64),
                               "sumsq": np.zeros(C, np.float64),
                               "T": 0, "N": 0}
        if name not in subject_acc:
            subject_acc[name] = {}

    for f in files:
        z = np.load(f, allow_pickle=False)
        X = z["X"].astype(np.float32)  # (N,C,T)
        N, C, T = X.shape
        splits = np.array(z["split"]) if "split" in z.files else np.array([args.split] * N)
        subs = np.array(z["subject_id"]) if "subject_id" in z.files else np.array([0] * N)

        # Iterate per split present in this file
        for s_name in np.unique(splits):
            m = splits == s_name
            if not m.any():
                continue
            Xm = X[m]
            n_m = Xm.shape[0]
            # sums over samples and time
            sum_ch = Xm.sum(axis=(0, 2)).astype(np.float64)
            sumsq_ch = (Xm ** 2).sum(axis=(0, 2)).astype(np.float64)
            T_m = n_m * T

            ensure_split(str(s_name), C)
            split_acc[str(s_name)]["sum"] += sum_ch
            split_acc[str(s_name)]["sumsq"] += sumsq_ch
            split_acc[str(s_name)]["T"] += T_m
            split_acc[str(s_name)]["N"] += n_m

            # per subject within this split
            subs_m = subs[m]
            for subj in np.unique(subs_m):
                ms = subs_m == subj
                Xm_s = Xm[ms]
                if Xm_s.size == 0:
                    continue
                sum_ch_s = Xm_s.sum(axis=(0, 2)).astype(np.float64)
                sumsq_ch_s = (Xm_s ** 2).sum(axis=(0, 2)).astype(np.float64)
                T_s = Xm_s.shape[0] * T
                d = subject_acc[str(s_name)].setdefault(str(int(subj)),
                    {"sum": np.zeros(C, np.float64), "sumsq": np.zeros(C, np.float64), "T": 0, "N": 0})
                d["sum"] += sum_ch_s
                d["sumsq"] += sumsq_ch_s
                d["T"] += T_s
                d["N"] += Xm_s.shape[0]

            # Also accumulate root if this split matches requested split
            if str(s_name) == args.split:
                if root_sum is None:
                    root_sum = np.zeros(C, np.float64)
                    root_sumsq = np.zeros(C, np.float64)
                root_sum += sum_ch
                root_sumsq += sumsq_ch
                root_T += T_m
                root_N += n_m

    if root_T == 0:
        raise SystemExit(f"No samples matched split={args.split}.")

    # Compute root stats and prepare JSON
    mean_root = (root_sum / root_T).astype(np.float32)
    var_root = (root_sumsq / root_T) - (mean_root.astype(np.float64) ** 2)
    std_root = np.sqrt(np.maximum(var_root, 1e-12)).astype(np.float32)

    def finalize(acc):
        mean = (acc["sum"] / acc["T"]).astype(np.float32)
        var = (acc["sumsq"] / acc["T"]) - (mean.astype(np.float64) ** 2)
        std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
        return {"mean": mean.tolist(), "std": std.tolist(),
                "count_samples": int(acc["N"]), "count_time": int(acc["T"])}

    out = {
        "mean": mean_root.tolist(),
        "std": std_root.tolist(),
        "split": args.split,
        "splits": {k: finalize(v) for k, v in split_acc.items()},
        "per_subject": {
            s: {sub: finalize(d) for sub, d in subj_map.items()} for s, subj_map in subject_acc.items()
        }
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"Saved extended stats to {args.out} (C={len(mean_root)})")

if __name__ == "__main__":
    main()
