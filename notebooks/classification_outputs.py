from pathlib import Path
import numpy as np
import torch
import random

from src.preprocessing import SP500Preprocessor
from src.dataset import SP500Dataset
from src.models import CNNLSTMModel
from src.train import train_model
from src.evaluate import evaluate_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def main():

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "sp500_2005_2025.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data preparation
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(data_path)

    # Dataset objects
    train_dataset = SP500Dataset(X_train, y_train, task="classification")
    val_dataset = SP500Dataset(X_val, y_val, task="classification")
    test_dataset = SP500Dataset(X_test, y_test, task="classification")

    # Model initialization
    model = CNNLSTMModel(
        input_size=X_train.shape[2],
        task="classification"
    )

    # Training
    model = train_model(
        model,
        train_dataset,
        val_dataset,
        task="classification",
        epochs=20,
        device=device,
        save_path=project_root / "results/models/classification_model.pth"
    )

    # Evaluation
    preds, targets = evaluate_model(
        model,
        test_dataset,
        task="classification",
        device=device
    )

    # Baseline
    baseline_acc = compute_baseline(y_test)
    print(f"Baseline Accuracy: {baseline_acc:.4f}")


def prepare_data(data_path: Path):
    """Runs preprocessing pipeline and returns datasets."""

    preprocessor = SP500Preprocessor(
        data_path=data_path,
        window_size=30
    )

    X_train, y_train, X_val, y_val, X_test, y_test = (
        preprocessor
            .load_data()
            .create_features()
            .create_target()
            .split_data()
            .scale()
            .get_datasets()
    )

    print_shapes(X_train, y_train, X_val, y_val, X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def print_shapes(X_train, y_train, X_val, y_val, X_test, y_test):
    """Prints dataset shapes and NaN diagnostics."""

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("NaNs in X_train:", np.isnan(X_train).sum())
    print("NaNs in y_train:", np.isnan(y_train).sum())


def compute_baseline(y_test: np.ndarray) -> float:
    """Computes majority class baseline accuracy."""

    positive_ratio = np.mean(y_test > 0)
    negative_ratio = np.mean(y_test <= 0)

    return max(positive_ratio, negative_ratio)


if __name__ == "__main__":
    main()