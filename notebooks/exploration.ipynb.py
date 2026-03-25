import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model


plt.style.use("seaborn-v0_8")


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "sp500_2005_2025.csv"
    plots_path = project_root / "results" / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path)
    basic_diagnostics(data)

    data = add_log_returns(data)

    plot_close(data, plots_path)
    plot_returns(data, plots_path)
    plot_distribution(data, plots_path)

    compute_statistics(data)
    run_adf_test(data)

    plot_acf_analysis(data, plots_path)
    run_arch_test(data)
    run_garch_model(data)


def load_data(path: Path) -> pd.DataFrame:
    """Loads dataset and sets Date index."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df


def basic_diagnostics(df: pd.DataFrame):
    """Prints basic dataset diagnostics."""
    print("Shape:", df.shape)
    print(df.info())
    print("Missing values:\n", df.isna().sum())
    print("Duplicates:", df.duplicated().sum())


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds log returns column."""
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()


def plot_close(df: pd.DataFrame, save_path: Path):
    """Plots closing price."""
    plt.figure(figsize=(12, 5))
    plt.plot(df["Close"])
    plt.title("S&P 500 Close Price (2005–2025)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(save_path / "close_price.png", dpi=300)
    plt.close()


def plot_returns(df: pd.DataFrame, save_path: Path):
    """Plots log returns."""
    plt.figure(figsize=(12, 5))
    plt.plot(df["log_return"])
    plt.title("Log Returns")
    plt.tight_layout()
    plt.savefig(save_path / "log_returns.png", dpi=300)
    plt.close()


def plot_distribution(df: pd.DataFrame, save_path: Path):
    """Plots distribution of log returns."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["log_return"], bins=100, kde=True)
    plt.title("Distribution of Log Returns")
    plt.tight_layout()
    plt.savefig(save_path / "log_return_distribution.png", dpi=300)
    plt.close()


def compute_statistics(df: pd.DataFrame):
    """Prints descriptive statistics."""
    mean = df["log_return"].mean()
    std = df["log_return"].std()
    skew = df["log_return"].skew()
    kurt = df["log_return"].kurtosis()

    print("Mean:", mean)
    print("Std:", std)
    print("Skew:", skew)
    print("Kurtosis:", kurt)


def run_adf_test(df: pd.DataFrame):
    """Performs Augmented Dickey-Fuller test."""
    result = adfuller(df["log_return"])
    print("ADF Statistic:", result[0])
    print("ADF p-value:", result[1])


def plot_acf_analysis(df: pd.DataFrame, save_path: Path):
    """Plots ACF for returns and squared returns."""
    plt.figure(figsize=(10, 4))
    plot_acf(df["log_return"], lags=40)
    plt.title("ACF of Log Returns")
    plt.tight_layout()
    plt.savefig(save_path / "acf_log_returns.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plot_acf(df["log_return"] ** 2, lags=40)
    plt.title("ACF of Squared Log Returns")
    plt.tight_layout()
    plt.savefig(save_path / "acf_squared_log_returns.png", dpi=300)
    plt.close()


def run_arch_test(df: pd.DataFrame):
    """Performs ARCH test for volatility clustering."""
    arch_test = het_arch(df["log_return"])
    print("ARCH test p-value:", arch_test[1])


def run_garch_model(df: pd.DataFrame):
    """Fits GARCH(1,1) model."""
    model = arch_model(100 * df["log_return"], vol="Garch", p=1, q=1)
    res = model.fit(disp="off")
    print(res.summary())


if __name__ == "__main__":
    main()