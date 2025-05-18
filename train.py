import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import MLP

# -------------------- hiperparâmetros --------------------
EPOCHS      = 200
LR          = 1e-3
BATCH_SIZE  = 64
DATA_PATH   = "dataset.csv"
MODEL_PATH  = "model.pth"
# ---------------------------------------------------------

def load_data(csv_path: str):
    """Carrega o dataset e devolve DataLoader."""
    df = pd.read_csv(csv_path)

    X = df[['theta0', 'theta1', 'theta2', 'x_target', 'y_target']].values
    y = df[['dtheta0', 'dtheta1', 'dtheta2']].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train(model: nn.Module, loader: DataLoader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == 1 or epoch == EPOCHS:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch:3d}/{EPOCHS}, Loss: {avg_loss:.6f}")

    return model


def main():
    loader = load_data(DATA_PATH)
    model  = MLP()

    trained_model = train(model, loader)
    torch.save(trained_model.state_dict(), MODEL_PATH)
    print(f"Modelo salvo em '{MODEL_PATH}'.")


if __name__ == "__main__":
    main()
