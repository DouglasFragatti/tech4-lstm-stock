import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import json
import os

# ---------------------------
# Hiperparâmetros
# ---------------------------
SYMBOL = "DIS"           # pode trocar para "PETR4.SA", "VALE3.SA", etc.
START_DATE = "2015-01-01"
END_DATE = "2024-07-20"

TIME_STEPS = 60          # janela de dias
TEST_SIZE = 0.15
VAL_SIZE = 0.15

EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3

HIDDEN_SIZE = 64
NUM_LAYERS = 2

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando device:", DEVICE)

# ---------------------------
# 1) Coleta dos dados
# ---------------------------
print(f"Baixando dados de {SYMBOL}...")
df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)

if df.empty:
    raise ValueError("yfinance não retornou dados. Verifique ticker, datas ou conexão.")

data = df[["Close"]].copy().dropna()
print("Primeiras linhas:")
print(data.head())

# ---------------------------
# 2) Normalização
# ---------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

def create_sequences(dataset, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(dataset)):
        X.append(dataset[i - time_steps:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, TIME_STEPS)
# X shape: (samples, time_steps, 1)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print("Formato X:", X.shape, "Formato y:", y.shape)

# ---------------------------
# 3) Split train / val / test
# ---------------------------
n_samples = X.shape[0]
test_size = int(TEST_SIZE * n_samples)
val_size = int(VAL_SIZE * n_samples)
train_size = n_samples - test_size - val_size

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)

# Converte para tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# 4) Modelo LSTM em PyTorch
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, time_steps, features)
        out, _ = self.lstm(x)        # out: (batch, time_steps, hidden_size)
        out = out[:, -1, :]          # pega o último passo de tempo
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = LSTMModel(
    input_size=1,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(model)

# ---------------------------
# 5) Treino com early stopping simples
# ---------------------------
best_val_loss = float("inf")
patience = 5
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    # Treino
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Validação
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    print(f"Epoch {epoch}/{EPOCHS} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # salvar melhor modelo temporário
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_model_best.pt"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping ativado.")
            break

# Carrega melhor modelo salvo
best_model_path = os.path.join(MODEL_DIR, "lstm_model_best.pt")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

# ---------------------------
# 6) Avaliação: MAE, RMSE, MAPE
# ---------------------------
model.eval()
y_test_preds = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        preds = model(xb)
        y_test_preds.append(preds.cpu().numpy())

y_test_preds = np.vstack(y_test_preds).flatten()

# inverter escala
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_real = scaler.inverse_transform(y_test_preds.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = mean_squared_error(y_test_real, y_pred_real) ** 0.5
mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print(f"MAE  (test): {mae:.4f}")
print(f"RMSE (test): {rmse:.4f}")
print(f"MAPE (test): {mape:.2f}%")

metrics = {
    "symbol": SYMBOL,
    "time_steps": TIME_STEPS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "mae_test": float(mae),
    "rmse_test": float(rmse),
    "mape_test": float(mape)
}

# ---------------------------
# 7) Salvar modelo + scaler + config + métricas
# ---------------------------
model_path = os.path.join(MODEL_DIR, "lstm_model.pt")  # PyTorch
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
config_path = os.path.join(MODEL_DIR, "config.pkl")
metrics_path = os.path.join(MODEL_DIR, "metrics.json")

torch.save({
    "model_state_dict": model.state_dict(),
    "input_size": 1,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS
}, model_path)

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

config = {
    "symbol": SYMBOL,
    "time_steps": TIME_STEPS,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS
}
with open(config_path, "wb") as f:
    pickle.dump(config, f)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(f"\nModelo salvo em:   {model_path}")
print(f"Scaler salvo em:   {scaler_path}")
print(f"Config salva em:   {config_path}")
print(f"Métricas salvas em {metrics_path}")
print("Treino concluído.")
