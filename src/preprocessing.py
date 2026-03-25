import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class SP500Preprocessor:
    """
    Preprocessing pipeline for S&P 500 time-series modeling.
    """

    def __init__(
        self,
        data_path: Path,
        window_size: int = 30,
        horizon: int = 1,
        train_end: str = "2018-12-31",
        val_end: str = "2021-12-31"
    ):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        self.data_path = data_path
        self.window_size = window_size
        self.horizon = horizon

        self.train_end = train_end
        self.val_end = val_end

        self.data: pd.DataFrame | None = None
        self.feature_columns: list[str] | None = None

    # 1. Load raw CSV data
    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")

        data = pd.read_csv(self.data_path, parse_dates=["Date"])
        data.set_index("Date", inplace=True)

        if data.isna().any().any():
            raise ValueError("Raw dataset contains missing values")

        self.data = data
        return self

    # 2. Feature engineering
    def create_features(self):
        if self.data is None:
            raise RuntimeError("Data must be loaded before feature creation")

        df = self.data.copy()

        # Log return
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # High-Low spread
        df["hl_spread"] = df["High"] - df["Low"]

        # Intraday return
        df["oc_return"] = np.log(df["Close"] / df["Open"])

        # Moving averages
        df["ma_5"] = df["Close"].rolling(5).mean()
        df["ma_10"] = df["Close"].rolling(10).mean()
        df["ma_20"] = df["Close"].rolling(20).mean()

        # Rolling volatility
        df["std_5"] = df["log_return"].rolling(5).std()
        df["std_10"] = df["log_return"].rolling(10).std()
        df["std_20"] = df["log_return"].rolling(20).std()

        # Log volume
        df["log_volume"] = np.log(df["Volume"] + 1)

        df = df.dropna()

        self.data = df

        self.feature_columns = [
            "log_return",
            "hl_spread",
            "oc_return",
            "ma_5",
            "ma_10",
            "ma_20",
            "std_5",
            "std_10",
            "std_20",
            "log_volume"
        ]

        return self

    # 3. Target construction
    def create_target(self):
        if self.data is None:
            raise RuntimeError("Features must be created before target construction")

        df = self.data.copy()

        if self.horizon == 1:
            df["target"] = df["log_return"].shift(-1)
        else:
            df["target"] = (
                df["log_return"]
                .rolling(self.horizon)
                .sum()
                .shift(-self.horizon)
            )

        df = df.dropna()

        self.data = df
        return self

    # 4. Time-based split
    def split_data(self):
        if self.data is None:
            raise RuntimeError("Target must be created before splitting")

        df = self.data.copy()

        train = df[df.index <= self.train_end]
        val = df[(df.index > self.train_end) & (df.index <= self.val_end)]
        test = df[df.index > self.val_end]

        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError("One of the splits is empty. Check date boundaries.")

        self.train_df = train
        self.val_df = val
        self.test_df = test

        return self

    # 5. Feature scaling using train statistics
    def scale(self):
        if not hasattr(self, "train_df"):
            raise RuntimeError("Data must be split before scaling")

        if self.feature_columns is None:
            raise RuntimeError("Feature columns are not defined")

        self.scaler = StandardScaler()

        # Fit only on train data
        self.scaler.fit(self.train_df[self.feature_columns])

        # Transform all splits
        self.train_df.loc[:, self.feature_columns] = self.scaler.transform(
            self.train_df[self.feature_columns]
        )

        self.val_df.loc[:, self.feature_columns] = self.scaler.transform(
            self.val_df[self.feature_columns]
        )

        self.test_df.loc[:, self.feature_columns] = self.scaler.transform(
            self.test_df[self.feature_columns]
        )

        return self

    # 6. Sequence generation
    def create_sequences(self, df: pd.DataFrame, return_dates: bool = False):
        if self.feature_columns is None:
            raise RuntimeError("Feature columns are not defined")

        X = []
        y = []
        dates = []

        values = df[self.feature_columns].values
        targets = df["target"].values
        index = df.index

        for i in range(self.window_size, len(df)):
            X.append(values[i - self.window_size:i])
            y.append(targets[i])

            if return_dates:
                dates.append(index[i])

        X = np.array(X)
        y = np.array(y)

        if return_dates:
            return X, y, np.array(dates)

        return X, y

    # 7. Dataset extraction
    def get_datasets(self, return_dates: bool = False):
        if not hasattr(self, "train_df"):
            raise RuntimeError("Data must be split before generating datasets")

        if return_dates:
            X_train, y_train, train_dates = self.create_sequences(
                self.train_df, return_dates=True
            )
            X_val, y_val, val_dates = self.create_sequences(
                self.val_df, return_dates=True
            )
            X_test, y_test, test_dates = self.create_sequences(
                self.test_df, return_dates=True
            )

            return (
                X_train, y_train, train_dates,
                X_val, y_val, val_dates,
                X_test, y_test, test_dates
            )

        X_train, y_train = self.create_sequences(self.train_df)
        X_val, y_val = self.create_sequences(self.val_df)
        X_test, y_test = self.create_sequences(self.test_df)

        return X_train, y_train, X_val, y_val, X_test, y_test