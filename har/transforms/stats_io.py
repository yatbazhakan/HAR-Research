import json
import numpy as np
from pathlib import Path
from .normalize import NormStats

def save_stats(stats: NormStats, path: str | Path):
    path = Path(path)
    obj = {"mean": stats.mean.tolist(), "std": stats.std.tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))

def load_stats(path: str | Path) -> NormStats:
    obj = json.loads(Path(path).read_text())
    return NormStats(mean=np.array(obj["mean"], dtype=np.float32),
                     std=np.array(obj["std"], dtype=np.float32))
