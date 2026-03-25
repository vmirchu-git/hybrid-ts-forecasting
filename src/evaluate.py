import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def evaluate_model(
    model,
    dataset,
    task: str = "regression",
    device: str = "cpu",
    batch_size: int = 64
):
    """
    Evaluates trained model on given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataset : torch.utils.data.Dataset
        Evaluation dataset.
    task : str
        One of: "regression", "classification", "quantile".
    device : str
        Device for computation.
    batch_size : int
        DataLoader batch size.

    Returns
    -------
    tuple
        Predictions and targets.
    """

    if task not in ["regression", "classification", "quantile"]:
        raise ValueError("task must be 'regression', 'classification' or 'quantile'")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X)

            preds.append(outputs.cpu().numpy())
            targets.append(y.numpy())

    preds = np.vstack(preds)
    targets = np.hstack(targets)

    if task == "classification":
        return _evaluate_classification(preds, targets)

    if task == "regression":
        return _evaluate_regression(preds, targets)

    return _evaluate_quantile(preds, targets)


def _evaluate_classification(preds: np.ndarray, targets: np.ndarray):
    """Evaluates binary classification metrics."""

    probs = 1 / (1 + np.exp(-preds))
    preds_binary = (probs > 0.5).astype(int).flatten()

    acc = accuracy_score(targets, preds_binary)
    prec = precision_score(targets, preds_binary, zero_division=0)
    rec = recall_score(targets, preds_binary, zero_division=0)
    roc = roc_auc_score(targets, probs)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")

    return preds_binary, targets


def _evaluate_regression(preds: np.ndarray, targets: np.ndarray):
    """Evaluates regression metrics."""

    preds_flat = preds.flatten()

    mse = np.mean((preds_flat - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_flat - targets))

    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

    return preds_flat, targets


def _evaluate_quantile(preds: np.ndarray, targets: np.ndarray):
    """Returns raw quantile predictions."""

    print("Quantile predictions shape:", preds.shape)
    print("Targets shape:", targets.shape)

    return preds, targets