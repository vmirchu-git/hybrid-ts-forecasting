import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model for time-series forecasting.

    Supports:
        - regression
        - classification
        - quantile regression
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        task: str = "regression",
        quantiles: list | None = None
    ):
        super().__init__()

        if task not in ["regression", "classification", "quantile"]:
            raise ValueError("task must be 'regression', 'classification' or 'quantile'")

        if task == "quantile" and quantiles is None:
            raise ValueError("quantiles must be provided for quantile task")

        self.task = task
        self.quantiles = quantiles

        # Convolutional block for local pattern extraction
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        # LSTM for temporal dependency modeling
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output head
        output_dim = 1 if task in ["regression", "classification"] else len(quantiles)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Input shape:
            (batch, window_size, features)
        """

        # Conv1D expects (batch, channels, sequence_length)
        x = x.permute(0, 2, 1)

        x = self.conv(x)
        x = self.relu(x)

        # Back to (batch, sequence_length, features)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        # Use last time step
        out = lstm_out[:, -1, :]

        out = self.head(out)

        return out


def quantile_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: list
) -> torch.Tensor:
    """
    Pinball loss for multi-quantile regression.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted quantiles (batch, n_quantiles)
    target : torch.Tensor
        True values (batch,)
    quantiles : list
        List of quantile levels

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """

    losses = []

    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        loss_q = torch.max(
            (q - 1) * errors,
            q * errors
        )
        losses.append(loss_q.unsqueeze(1))

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

    return loss