from __future__ import annotations
import numpy as np, glob
from torch.utils.data import Dataset
from har.transforms.normalize import NormStats

class NPZShardsDataset(Dataset):
    def __init__(self, shards_glob: str, split: str, stats: NormStats | None = None):
        self.files = sorted(glob.glob(shards_glob))
        self.index = []  # list of (file_idx, sample_idx)
        self._map = []  # cache of per-file indices
        self.stats = stats

        for i, f in enumerate(self.files):
            z = np.load(f, allow_pickle=False)
            n = z["X"].shape[0]
            if "split" in z.files:
                if split == "all":
                    idx = np.arange(n)
                else:
                    m = np.array(z["split"]) == split
                    idx = np.nonzero(m)[0]
            else:
                # if shards are not split-tagged, include all
                idx = np.arange(n)
            self._map.append((f, idx))
            for j in idx:
                self.index.append((i, j))

    def __len__(self): return len(self.index)

    def __getitem__(self, k):
        file_i, sample_j = self.index[k]
        f, idx = self._map[file_i]
        z = np.load(f, allow_pickle=False)
        x = z["X"][sample_j].astype(np.float32)   # (C,T)
        y = int(z["y"][sample_j])
        if self.stats is not None:
            x = self.stats.apply(x)
        return x, y
