from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    # probs: (N,C), labels: (N,)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (confidences > lo) & (confidences <= hi)
        if not np.any(m):
            continue
        conf_bin = confidences[m].mean()
        acc_bin = accuracies[m].mean()
        ece += np.abs(acc_bin - conf_bin) * (m.mean())
    return float(ece)


class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = torch.nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_temp)
        return logits / T

    @torch.no_grad()
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200, lr: float = 0.01):
        opt = torch.optim.LBFGS([self.log_temp], lr=lr, max_iter=max_iter)
        nll = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(torch.exp(self.log_temp).item())


