#!/usr/bin/env python
import argparse
import os
import sys
import zipfile
import shutil
from pathlib import Path
from urllib.parse import quote
import requests

# Simple helper to stream-download with progress-ish prints
def fetch(url, out_path, timeout=120):
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r  -> {downloaded/1e6:.1f} MB / {total/1e6:.1f} MB ({pct}%)", end="")
        print()

def try_urls(name, urls, dest_zip):
    for u in urls:
        try:
            fetch(u, dest_zip)
            return True
        except Exception as e:
            print(f"[warn] failed: {u} ({e})")
    return False

def unzip_to(src_zip, target_dir):
    with zipfile.ZipFile(src_zip, 'r') as z:
        z.extractall(target_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./data", help="where to put datasets")
    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # UCI repository sometimes serves from /ml/machine-learning-databases/<id>/ or /static/public/<id>/
    # Include multiple candidates for robustness.
    datasets = {
        "UCI-HAR": {
            "folder": "UCI-HAR",
            "zname": "UCI_HAR_Dataset.zip",
            "urls": [
                # classic location
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/" + quote("UCI HAR Dataset.zip"),
                # static path
                "https://archive.ics.uci.edu/static/public/240/" + quote("UCI HAR Dataset.zip"),
            ],
            "post": "UCI HAR Dataset"
        },
        "PAMAP2": {
            "folder": "PAMAP2",
            "zname": "PAMAP2_Dataset.zip",
            "urls": [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
                "https://archive.ics.uci.edu/static/public/231/" + quote("PAMAP2 Dataset.zip"),
            ],
            "post": None  # root contains .dat files and subfolders
        },
        "MHEALTH": {
            "folder": "MHEALTH",
            "zname": "MHEALTHDATASET.zip",
            "urls": [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip",
                "https://archive.ics.uci.edu/static/public/319/MHEALTHDATASET.zip",
            ],
            "post": "MHEALTHDATASET"
        },
    }

    for key, meta in datasets.items():
        print(f"\n=== {key} ===")
        droot = outdir / meta["folder"]
        if droot.exists():
            print(f"[skip] found {droot}")
            continue
        tmp_dir = outdir / f"__tmp_{meta['folder']}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        zpath = tmp_dir / meta["zname"]

        ok = try_urls(key, meta["urls"], zpath)
        if not ok or not zpath.exists() or zpath.stat().st_size == 0:
            print(f"[error] could not download {key}. Please fetch manually and place under {droot}.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        print(f"[unzip] {zpath} -> {droot}")
        droot.mkdir(parents=True, exist_ok=True)
        unzip_to(zpath, droot)

        # optional: flatten common top-level folder (UCI HAR / MHEALTH ship with a folder)
        post = meta.get("post")
        if post:
            inner = droot / post
            if inner.exists() and inner.is_dir():
                # move inner content up
                for p in inner.iterdir():
                    shutil.move(str(p), str(droot / p.name))
                shutil.rmtree(inner, ignore_errors=True)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        # small sanity ping
        nfiles = sum(1 for _ in droot.rglob("*") if _.is_file())
        print(f"[ok] {key} ready at {droot} ({nfiles} files)")

    print("\nAll done.")
    print(f"Datasets are under: {outdir}")
    print("Structure:")
    for name in datasets:
        print(f"  - {name} -> {outdir / datasets[name]['folder']}")

if __name__ == "__main__":
    sys.exit(main())
