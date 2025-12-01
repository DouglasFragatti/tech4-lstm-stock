from fastapi import FastAPI, Request
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import time
import json

# ---------------------------
# Caminhos dos artefatos
# ---------------------------
MODEL_DIR = "model"
model_path = os.path.join(MODEL_DIR, "lstm_model.pt")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
config_path = os.path.join(MODEL_DIR, "config.pkl")
metrics_path = os.path.join(MODEL_DIR, "metrics.json")

if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(config_path)):
    raise RuntimeError("Modelo/scaler/config não encontrados. Rode antes o script train_model.py")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(config_path, "rb") as f:
    config = pickle.load(f)

DEFAULT_SYMBOL = config["symbol"]
TIME_STEPS = config["time_steps"]
HIDDEN_SIZE = config["hidden_size"]
NUM_LAYERS = config["num_layers"]

metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

# ---------------------------
# Define o mesmo modelo LSTM usado no treino
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
        out, _ = self.lstm(x)       # (batch, time_steps, hidden)
        out = out[:, -1, :]         # pega o último passo de tempo
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = LSTMModel(
    input_size=1,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(DEVICE)

checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Stock LSTM API (PyTorch)",
    description="API REST para previsão de preço de fechamento de ações usando LSTM (PyTorch)",
    version="1.0.0"
)

# ---------------------------
# Middleware de monitoramento (tempo de resposta)
# ---------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # ms
    response.headers["X-Process-Time-ms"] = str(round(process_time, 2))
    print(f"{request.method} {request.url.path} - {process_time:.2f} ms")
    return response

# ---------------------------
# Schemas de request/response
# ---------------------------
class PredictBySymbolRequest(BaseModel):
    symbol: str | None = None

class PredictFromSeriesRequest(BaseModel):
    prices: list[float]
    n_steps_ahead: int = 1  # ainda 1 passo

class PredictResponse(BaseModel):
    symbol: str
    predicted_close: float

class MetricsResponse(BaseModel):
    symbol: str
    mae_test: float
    rmse_test: float
    mape_test: float

# ---------------------------
# Funções auxiliares
# ---------------------------
def build_sequence_from_symbol(symbol: str):
    data = yf.download(symbol, period="2y")

    if data.empty:
        raise ValueError("Não foi possível obter dados para o símbolo informado.")

    closes = data[["Close"]].dropna()
    if len(closes) < TIME_STEPS:
        raise ValueError(f"Dados insuficientes. Precisa de pelo menos {TIME_STEPS} pontos.")

    last_data = closes.values[-TIME_STEPS:]
    scaled = scaler.transform(last_data)
    X = np.array([scaled], dtype=np.float32)  # (1, time_steps, 1)
    return torch.tensor(X, dtype=torch.float32).to(DEVICE)


def build_sequence_from_series(prices: list[float]):
    if len(prices) < TIME_STEPS:
        raise ValueError(f"É necessário enviar pelo menos {TIME_STEPS} preços.")
    arr = np.array(prices, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(arr)
    last_scaled = scaled[-TIME_STEPS:]
    X = np.array([last_scaled], dtype=np.float32)  # (1, time_steps, 1)
    return torch.tensor(X, dtype=torch.float32).to(DEVICE)


def predict_from_X(X: torch.Tensor) -> float:
    with torch.no_grad():
        preds = model(X)  # (1, 1)
    pred_scaled = preds.cpu().numpy()[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    return float(pred)

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def root():
    return {
        "message": "API de previsão LSTM (PyTorch) está no ar.",
        "default_symbol": DEFAULT_SYMBOL,
        "time_steps": TIME_STEPS
    }


@app.post("/predict_by_symbol", response_model=PredictResponse)
def predict_by_symbol(request: PredictBySymbolRequest):
    symbol = request.symbol or DEFAULT_SYMBOL
    X = build_sequence_from_symbol(symbol)
    pred = predict_from_X(X)
    return PredictResponse(symbol=symbol, predicted_close=pred)


@app.post("/predict_from_series", response_model=PredictResponse)
def predict_from_series(request: PredictFromSeriesRequest):
    X = build_sequence_from_series(request.prices)
    pred = predict_from_X(X)
    return PredictResponse(symbol="CUSTOM_SERIES", predicted_close=pred)


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    if not metrics:
        raise RuntimeError("Métricas não encontradas. Rode o treinamento primeiro.")
    return MetricsResponse(
        symbol=metrics["symbol"],
        mae_test=metrics["mae_test"],
        rmse_test=metrics["rmse_test"],
        mape_test=metrics["mape_test"],
    )
