from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
import sys
import numpy as np
from collections import defaultdict

# Ensure repo root in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def stratified_subset(indices_by_class: dict[int, list[int]], max_per_class: int) -> set[int]:
    sel = set()
    for cls, idxs in indices_by_class.items():
        if not idxs:
            continue
        n = min(max_per_class, len(idxs))
        choice = np.random.choice(idxs, size=n, replace=False)
        sel.update(int(i) for i in choice)
    return sel


def build_index(shards_glob: str) -> dict:
    files = sorted(glob.glob(shards_glob))
    if not files:
        raise SystemExit("No shards matched.")
    index = []  # entries: {file, j, subject, y}
    for f in files:
        z = np.load(f, allow_pickle=False)
        y = z["y"].astype(int)
        n = y.shape[0]
        if "subject_id" in z.files:
            subs = np.array(z["subject_id"]).astype(int)
        else:
            subs = np.zeros(n, dtype=int)
        for j in range(n):
            index.append({"file": f, "j": j, "subject": int(subs[j]), "y": int(y[j])})
    return {"entries": index}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_glob", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--val_per_class", type=int, default=20,
                    help="validation samples per class per fold (cap; uses less if unavailable)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    idx = build_index(args.shards_glob)
    entries = idx["entries"]
    subjects = sorted({e["subject"] for e in entries})
    classes = sorted({e["y"] for e in entries})

    # group entries by subject and by class
    by_subject = defaultdict(list)
    for i, e in enumerate(entries):
        by_subject[e["subject"]].append(i)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # For each LOSO fold: leave one subject as test, stratify a small val from train
    for test_subj in subjects:
        test_idx = set(by_subject[test_subj])
        train_pool = [i for s, idxs in by_subject.items() if s != test_subj for i in idxs]

        # stratify val from train_pool
        cls_to_idxs = defaultdict(list)
        for i in train_pool:
            cls_to_idxs[entries[i]["y"]].append(i)
        val_idx = stratified_subset(cls_to_idxs, args.val_per_class)
        train_idx = set(train_pool) - val_idx

        fold = {
            "test_subject": int(test_subj),
            "classes": classes,
            "counts": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
            "train": sorted(int(i) for i in train_idx),
            "val": sorted(int(i) for i in val_idx),
            "test": sorted(int(i) for i in test_idx),
            "index_schema": {"entry": ["file", "j", "subject", "y"]},
        }

        # save fold and companion index once per dataset
        fold_path = outdir / f"loso_fold_subject_{test_subj}.json"
        fold_path.write_text(json.dumps(fold))

    # Save the compacted index too for reference
    (outdir / "index.json").write_text(json.dumps(idx))
    print(f"Wrote LOSO folds to {outdir} for {len(subjects)} subjects.")


if __name__ == "__main__":
    main()


