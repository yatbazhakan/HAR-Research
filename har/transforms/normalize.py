from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class NormStats:
    mean: np.ndarray  # (C,)
    std:  np.ndarray  # (C,)

    @staticmethod
    def from_data(X: np.ndarray, eps: float = 1e-6) -> "NormStats":
        # X: (N, C, T) or (C, T) (we’ll handle both)
        if X.ndim == 2:
            C = X.shape[0]
            mean = X.mean(axis=1)
            std  = X.std(axis=1)
        else:
            C = X.shape[1]
            mean = X.mean(axis=(0, 2))
            std  = X.std(axis=(0, 2))
        std = np.where(std < eps, 1.0, std)
        return NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        # x: (C, T)
        return (x - self.mean[:, None]) / self.std[:, None]
