import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def train_model(
    model,
    train_dataset,
    val_dataset,
    task="regression",
    quantiles=None,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    device="cpu",
    save_path="results/models/model.pth"
):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if task == "regression":
        criterion = nn.MSELoss()

    elif task == "classification":
        criterion = nn.BCEWithLogitsLoss()

    elif task == "quantile":
        from src.models import quantile_loss
        criterion = lambda preds, y: quantile_loss(preds, y, quantiles)

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(X)

            if task in ["regression", "classification", "quantile"]:
                y = y.unsqueeze(1)

            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:

                X = X.to(device)
                y = y.to(device)

                outputs = model(X)

                if task in ["regression", "classification", "quantile"]:
                    y = y.unsqueeze(1)

                loss = criterion(outputs, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print("Training complete. Best val loss:", best_val_loss)

    return model