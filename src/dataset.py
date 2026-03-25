import torch
from torch.utils.data import Dataset
import numpy as np


class SP500Dataset(Dataset):
    """
    PyTorch Dataset for S&P 500 time-series modeling.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (samples, window_size, features).
    y : np.ndarray
        Target values.
    task : str
        One of:
            "regression"
            "classification"
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, task: str = "regression"):

        if task not in ["regression", "classification"]:
            raise ValueError("task must be 'regression' or 'classification'")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        self.task = task

        # Convert features to float tensor
        self.X = torch.tensor(X, dtype=torch.float32)

        # Convert targets
        if self.task == "classification":
            # Binary label: 1 if return > 0, else 0
            y_processed = (y > 0).astype(np.float32)
        else:
            y_processed = y.astype(np.float32)

        self.y = torch.tensor(y_processed, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int):
        """Returns single sample (X, y)."""
        return self.X[idx], self.y[idx]