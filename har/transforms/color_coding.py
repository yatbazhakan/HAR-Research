from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple, Optional

def _minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + eps)

def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu, sd = x.mean(), x.std()
    sd = sd if sd > eps else 1.0
    return (x - mu) / sd

def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)

def color_triplet_rgb(x: np.ndarray,
                      triplet: Sequence[int] = (0,1,2),
                      normalize: str = "per_channel",
                      stretch: str = "minmax") -> np.ndarray:
    """
    Map 3 selected channels of shape (C,T) to an RGB image (H=Hc, W=T, 3).
    Common choice: acc_x, acc_y, acc_z OR [|acc|, |gyro|, ECG].
    H is 1 (row), so result is (1,T,3). You can later stack rows for multiple triplets.
    """
    C, T = x.shape
    assert len(triplet) == 3, "triplet must have length 3"
    sel = np.stack([x[i] for i in triplet], axis=0)  # (3,T)

    if normalize == "per_channel":
        for i in range(3):
            if stretch == "minmax":
                sel[i] = _minmax(sel[i])
            else:
                sel[i] = _minmax(_zscore(sel[i]))
    elif normalize == "global":
        if stretch == "minmax":
            sel = _minmax(sel)
        else:
            sel = _minmax(_zscore(sel))
    else:
        # no normalization
        pass

    img = sel.transpose(1,0)[None, ...]  # (1,T,3)
    return img  # float32 in [0,1]

def color_pca_rgb(x: np.ndarray,
                  k: int = 3,
                  center: bool = True,
                  whiten: bool = False) -> np.ndarray:
    """
    PCA along channel axis → 3 components mapped to RGB.
    x: (C,T). Returns (1,T,3) in [0,1].
    """
    C, T = x.shape
    X = x.copy().astype(np.float32)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    # Covariance over channels
    cov = np.cov(X, rowvar=True)  # (C,C)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:k]]  # (C,3)
    Z = W.T @ X              # (3,T)
    if whiten:
        lam = np.maximum(eigvals[idx[:k]], 1e-6)
        Z = Z / np.sqrt(lam[:, None])
    # Scale to [0,1] per component
    for i in range(3):
        Z[i] = _minmax(Z[i])
    return Z.transpose(1,0)[None, ...]

def color_hsv_energy(x: np.ndarray,
                     acc_idx: Tuple[int,int,int] = (0,1,2),
                     gyro_idx: Tuple[int,int,int] = (3,4,5),
                     ecg_idx: Optional[int] = None) -> np.ndarray:
    """
    Encode magnitude and temporal change into HSV; convert to RGB.
    H ~ relative contribution (e.g., acc vs gyro), S ~ dynamics, V ~ overall magnitude.
    Returns (1,T,3) in [0,1].
    """
    from colorsys import hsv_to_rgb

    C, T = x.shape
    # magnitudes
    acc_mag = np.sqrt((x[list(acc_idx)]**2).sum(axis=0))
    gyro_mag = np.sqrt((x[list(gyro_idx)]**2).sum(axis=0))
    ecg = x[ecg_idx] if ecg_idx is not None else None

    acc_n = _minmax(acc_mag)
    gyro_n = _minmax(gyro_mag)
    if ecg is not None:
        ecg_n = _minmax(np.abs(ecg))
        V = _minmax(0.5*acc_n + 0.3*gyro_n + 0.2*ecg_n)
    else:
        V = _minmax(0.6*acc_n + 0.4*gyro_n)

    # dynamics (temporal derivative magnitude)
    dyn = np.abs(np.gradient(V))
    S = _minmax(dyn)

    # hue: relative share acc vs gyro (0→acc, 2/3→gyro)
    # map ratio r = acc/(acc+gyro) to hue range [0, 0.66]
    denom = acc_n + gyro_n + 1e-6
    r = acc_n / denom
    H = 0.66 * (1.0 - r)

    # HSV → RGB
    rgb = np.zeros((T,3), dtype=np.float32)
    for t in range(T):
        rr, gg, bb = hsv_to_rgb(float(H[t]), float(S[t]), float(V[t]))
        rgb[t] = [rr, gg, bb]
    return rgb[None, ...]
