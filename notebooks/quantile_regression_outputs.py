from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import scipy.stats as stats
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
    plots_path = project_root / "results" / "plots"
    model_path = project_root / "results" / "models" / "quantile_model.pth"

    plots_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantiles = [0.05, 0.5, 0.95]

    (
        X_train, y_train, train_dates,
        X_val, y_val, val_dates,
        X_test, y_test, test_dates
    ) = prepare_data(data_path)

    train_dataset = SP500Dataset(X_train, y_train, task="regression")
    val_dataset = SP500Dataset(X_val, y_val, task="regression")
    test_dataset = SP500Dataset(X_test, y_test, task="regression")

    model = CNNLSTMModel(
        input_size=X_train.shape[2],
        task="quantile",
        quantiles=quantiles
    )

    model = train_model(
        model,
        train_dataset,
        val_dataset,
        task="quantile",
        quantiles=quantiles,
        epochs=30,
        device=device,
        save_path=model_path
    )

    preds_test, targets_test = evaluate_model(
        model,
        test_dataset,
        task="quantile",
        device=device
    )

    preds_val, targets_val = evaluate_model(
        model,
        val_dataset,
        task="quantile",
        device=device
    )

    analyze_quantile_results(
        preds_test,
        targets_test,
        preds_val,
        targets_val,
        y_train,
        val_dates,
        plots_path
    )


def prepare_data(data_path: Path):
    """Runs preprocessing pipeline and returns datasets with dates."""

    preprocessor = SP500Preprocessor(
        data_path=data_path,
        window_size=30
    )

    return (
        preprocessor
            .load_data()
            .create_features()
            .create_target()
            .split_data()
            .scale()
            .get_datasets(return_dates=True)
    )


def analyze_quantile_results(
    preds_test,
    targets_test,
    preds_val,
    targets_val,
    y_train,
    val_dates,
    plots_path
):
    """Computes coverage, calibration and risk metrics."""

    lower = preds_test[:, 0]
    upper = preds_test[:, 2]

    coverage = np.mean((targets_test >= lower) & (targets_test <= upper))
    interval_width = np.mean(upper - lower)

    print(f"Coverage: {coverage:.4f}")
    print(f"Average interval width: {interval_width:.6f}")

    below_lower = np.mean(targets_test < lower)
    above_upper = np.mean(targets_test > upper)

    print(f"Below Q05: {below_lower:.4f}")
    print(f"Above Q95: {above_upper:.4f}")

    sigma = np.std(y_train)
    fixed_lower = -1.645 * sigma
    fixed_upper = 1.645 * sigma

    coverage_fixed = np.mean(
        (targets_test >= fixed_lower) &
        (targets_test <= fixed_upper)
    )

    print(f"Fixed interval coverage: {coverage_fixed:.4f}")

    LR_stat, p_val = kupiec_test(targets_test, lower)
    print(f"Kupiec LR: {LR_stat:.4f}")
    print(f"Kupiec p-value: {p_val:.6e}")

    VaR_5 = np.mean(lower)
    ES = np.mean(targets_test[targets_test < lower])

    print(f"Average VaR 5%: {VaR_5:.6f}")
    print(f"Expected Shortfall: {ES:.6f}")

    plot_intervals(targets_test, lower, upper, plots_path)

    crisis_analysis(preds_val, targets_val, val_dates)


def kupiec_test(actual, predicted_lower, alpha=0.05):
    """Performs Kupiec unconditional coverage test."""

    violations = np.sum(actual < predicted_lower)
    n = len(actual)

    pi_hat = violations / n
    pi_hat = np.clip(pi_hat, 1e-6, 1 - 1e-6)

    LR = -2 * (
        (n - violations) * np.log((1 - alpha) / (1 - pi_hat)) +
        violations * np.log(alpha / pi_hat)
    )

    p_value = 1 - stats.chi2.cdf(LR, df=1)
    return LR, p_value


def plot_intervals(targets, lower, upper, plots_path):
    """Plots first 200 predicted intervals."""

    plt.figure(figsize=(12, 6))
    plt.plot(targets[:200], label="True")
    plt.plot(lower[:200], label="Q05")
    plt.plot(upper[:200], label="Q95")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_path / "quantile_interval_example.png", dpi=300)
    plt.close()


def crisis_analysis(preds_val, targets_val, val_dates):
    """Evaluates interval coverage during 2020 crisis."""

    crisis_start = pd.Timestamp("2020-01-01")
    crisis_end = pd.Timestamp("2020-12-31")

    lower_val = preds_val[:, 0]
    upper_val = preds_val[:, 2]

    mask = (val_dates >= crisis_start) & (val_dates <= crisis_end)

    if mask.sum() == 0:
        print("No crisis observations in validation period.")
        return

    coverage_crisis = np.mean(
        (targets_val[mask] >= lower_val[mask]) &
        (targets_val[mask] <= upper_val[mask])
    )

    print(f"Crisis coverage (2020, validation): {coverage_crisis:.4f}")


if __name__ == "__main__":
    main()