from pathlib import Path
import yfinance as yf


def download_sp500(
    start: str = "2005-01-01",
    end: str = "2025-12-31",
    ticker: str = "^GSPC"
) -> Path:
    """
    Downloads historical S&P 500 data from Yahoo Finance
    and saves it to data/raw directory.

    Parameters
    ----------
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    ticker : str
        Yahoo Finance ticker symbol.

    Returns
    -------
    Path
        Path to saved CSV file.
    """

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "sp500_2005_2025.csv"

    data = yf.download(ticker, start=start, end=end)

    if data is None or data.empty:
        raise ValueError("Failed to download data from Yahoo Finance")

    # Remove potential multi-index columns
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    data.to_csv(data_path, index=False)

    return data_path


if __name__ == "__main__":
    path = download_sp500()
    print(f"Data saved to: {path}")