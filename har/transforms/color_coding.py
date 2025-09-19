"""Color Coding Transform Module (normalization optional)"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List

__all__ = ['compute_subject_minmax', 'ColorCodingTransform']


def _to_T3(a: np.ndarray, name: str) -> np.ndarray:
    """Return array shaped [T, 3] from [T,3] or [3,T]."""
    if a.ndim != 2:
        raise ValueError(f"{name}: data must be 2D, got {a.shape}")
    if a.shape[1] == 3:
        return a
    if a.shape[0] == 3:
        return a.T
    raise ValueError(f"{name}: expected 3 axes, got {a.shape}")


def compute_subject_minmax(
    streams_by_sensor: Dict[str, np.ndarray]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute per-subject min/max for each sensor on UNSEGMENTED streams.
    Returns dict[sensor] = (min[3], max[3]) as float32.
    Note: not required if normalize_mode!='minmax_to_255'.
    """
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sensor_name, data in streams_by_sensor.items():
        data = _to_T3(np.asarray(data), sensor_name).astype(np.float32, copy=False)
        min_vals = np.nanmin(data, axis=0).astype(np.float32)
        max_vals = np.nanmax(data, axis=0).astype(np.float32)
        if min_vals.shape != (3,) or max_vals.shape != (3,):
            raise ValueError(f"{sensor_name}: min/max must be shape (3,), got {min_vals.shape}/{max_vals.shape}")
        result[sensor_name] = (min_vals, max_vals)
    return result


class ColorCodingTransform:
    """
    Transform multi-sensor windows into color-coded RGB images.
    Vertical stacking order is given externally via `sensors_order`.

    normalize_mode:
      - 'minmax_to_255' (default): per-subject min–max → [0,255] (requires subject_minmax)
      - 'unit_to_255' : assume input already in [0,1]; scale to [0,255]
      - 'none'        : assume input already in [0,255]; just clip/cast to uint8
    """

    def __init__(
        self,
        sensors_order: List[str],
        time_steps_per_window: int = 50,
        sensor_band_height_px: int = 10,
        output_format: str = 'HWC',
        normalize_mode: str = 'minmax_to_255',
        eps: float = 1e-6
    ):
        if output_format not in ('HWC', 'CHW'):
            raise ValueError("output_format must be 'HWC' or 'CHW'")
        if len(sensors_order) != len(set(sensors_order)):
            dupes = [s for s in sensors_order if sensors_order.count(s) > 1]
            raise ValueError(f"Duplicate sensor names in sensors_order: {sorted(set(dupes))}")
        if normalize_mode not in ('minmax_to_255', 'unit_to_255', 'none'):
            raise ValueError("normalize_mode must be one of {'minmax_to_255','unit_to_255','none'}")

        self.sensors_order = list(sensors_order)
        self.time_steps_per_window = int(time_steps_per_window)
        self.sensor_band_height_px = int(sensor_band_height_px)
        self.output_format = output_format
        self.normalize_mode = normalize_mode
        self.eps = float(eps)

    def __call__(
        self,
        window_by_sensor: Dict[str, np.ndarray],
        subject_minmax: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None
    ) -> np.ndarray:
        # strict key check: no missing or extra sensors
        expected = set(self.sensors_order)
        got = set(window_by_sensor.keys())
        if expected != got:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(f"Sensor set mismatch. Missing={missing}, Extra={extra}")

        # require minmax only when needed
        if self.normalize_mode == 'minmax_to_255' and subject_minmax is None:
            raise ValueError("subject_minmax is required when normalize_mode='minmax_to_255'")

        num_sensors = len(self.sensors_order)
        H = self.sensor_band_height_px * num_sensors
        W = self.time_steps_per_window

        img = (np.zeros((H, W, 3), dtype=np.uint8)
               if self.output_format == 'HWC'
               else np.zeros((3, H, W), dtype=np.uint8))

        for i, sensor_name in enumerate(self.sensors_order):
            data = _to_T3(np.asarray(window_by_sensor[sensor_name]), sensor_name).astype(np.float32, copy=False)
            if data.shape[0] != self.time_steps_per_window:
                raise ValueError(
                    f"{sensor_name}: expected {self.time_steps_per_window} time steps, got {data.shape[0]}"
                )

            # --- Normalization / Scaling to 0..255 ---
            if self.normalize_mode == 'minmax_to_255':
                if sensor_name not in subject_minmax:
                    raise ValueError(f"Missing min/max for sensor: {sensor_name}")
                min_vals, max_vals = subject_minmax[sensor_name]
                min_vals = np.asarray(min_vals, dtype=np.float32)
                max_vals = np.asarray(max_vals, dtype=np.float32)
                denom = np.maximum(max_vals - min_vals, self.eps)
                scaled = 255.0 * (data - min_vals) / denom  # [T,3]
            elif self.normalize_mode == 'unit_to_255':
                # assume input already in [0,1]
                scaled = 255.0 * np.clip(data, 0.0, 1.0)
            else:  # 'none' → assume input already in [0,255]
                scaled = np.clip(data, 0.0, 255.0)

            scaled_u8 = np.clip(scaled, 0.0, 255.0).astype(np.uint8)  # [T,3]

            top = i * self.sensor_band_height_px
            bot = (i + 1) * self.sensor_band_height_px

            if self.output_format == 'HWC':
                band = np.broadcast_to(scaled_u8[np.newaxis, :, :], (self.sensor_band_height_px, W, 3))
                img[top:bot, :, :] = band
            else:  # CHW
                band = np.broadcast_to(scaled_u8.T[:, np.newaxis, :], (3, self.sensor_band_height_px, W))
                img[:, top:bot, :] = band

        return img
