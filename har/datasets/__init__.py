from pathlib import Path
from typing import Any, Dict

from .ucihar import load_ucihar_windows
from .pamap2 import load_pamap2_stream
from .mhealth import load_mhealth_stream

def load_dataset(name: str, root_dir: str, **kwargs: Any):
    """
    Unified loader entry point.

    - name: one of {"uci_har","ucihar","pamap2","mhealth"}
    - root_dir: dataset root path
    - kwargs: forwarded to the underlying loader
      * UCI-HAR: no extra args required
      * PAMAP2: supports fs, win_sec, overlap
      * MHEALTH: supports fs, win_sec, overlap
    """
    key = name.lower().replace("-", "_")
    if key in {"uci_har", "ucihar"}:
        return load_ucihar_windows(root_dir)
    if key == "pamap2":
        return load_pamap2_stream(root_dir, **kwargs)
    if key == "mhealth":
        return load_mhealth_stream(root_dir, **kwargs)
    raise ValueError(f"Unknown dataset name: {name}")

__all__ = [
    "load_dataset",
    "load_ucihar_windows",
    "load_pamap2_stream",
    "load_mhealth_stream",
]


