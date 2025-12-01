📈 Stock Price Prediction using LSTM (PyTorch) + FastAPI
Full ML Pipeline — Training, Evaluation, API Deployment

Este projeto implementa uma solução completa de Machine Learning para prever preços de fechamento de ações utilizando redes neurais LSTM com PyTorch, incluindo:

✔ Coleta automática de dados financeiros (Yahoo Finance)

✔ Treinamento do modelo LSTM

✔ Normalização com MinMaxScaler

✔ Cálculo das métricas (MAE, RMSE, MAPE)

✔ Salvamento seguro do modelo e do scaler

✔ Deploy completo via API REST construída com FastAPI

✔ Swagger UI com endpoints para inferência em tempo real

🚀 1. Objetivo do Projeto

Criar uma pipeline profissional de previsão de preços de ações capaz de:

Aprender padrões temporais via LSTM

Prever o próximo preço de fechamento

Fornecer previsões via API REST pública

Facilitar treinamento, reuso e deploy do modelo

Ideal para:

Portfólios

Trabalhos acadêmicos

Sistemas reais de análise financeira

Estudo prático de Deep Learning

📦 2. Estrutura do Projeto
Tech4/
│── api.py               # API FastAPI (deploy)
│── train_model.py       # Treinamento do modelo LSTM
│── main.py              # Script simples de coleta/salvar no SQLite (opcional)
│── market_data.db       # Banco local (opcional)
│── requirements.txt     # Dependências
│── README.md            # Este arquivo
│── model/
│     ├── lstm_model.pt     # Modelo PyTorch treinado
│     ├── scaler.pkl        # Scaler para normalização
│     ├── config.pkl        # Configurações (time_steps, ticker)
│     └── metrics.json      # Métricas de validação
└── .venv/               # Ambiente virtual Python

🔧 3. Tecnologias Utilizadas

Linguagem:

Python 3.10

Machine Learning / Deep Learning:

PyTorch

NumPy

Scikit-Learn

MinMaxScaler

Dados Financeiros:

yfinance

API & Deploy:

FastAPI

Uvicorn

📥 4. Instalação do ambiente
1️⃣ Criar ambiente virtual
python -m venv .venv
.\.venv\Scripts\activate

2️⃣ Instalar dependências
pip install -r requirements.txt

📊 5. Treinamento do Modelo

Para treinar o modelo LSTM:

python train_model.py


Ao final, os arquivos serão gerados dentro da pasta model/:

lstm_model.pt

scaler.pkl

config.pkl

metrics.json

Exemplo de saída:
MAE: 1.51
RMSE: 2.07
MAPE: 1.64%

⚙️ 6. Inicializando a API

Com o ambiente ativo, execute:

uvicorn api:app --reload


Acesse no navegador:
👉 http://127.0.0.1:8000/docs

Você verá a interface Swagger (documentação interativa).

🌐 7. Endpoints
✔ GET /

Retorna status da API e configuração padrão.

✔ POST /predict_by_symbol

Faz o download dos últimos 2 anos de dados do ticker informado, gera a série e retorna a previsão.

Body:

{
  "symbol": "DIS"
}


Retorno:

{
  "symbol": "DIS",
  "predicted_close": 98.24
}

✔ POST /predict_from_series

Inferência usando uma série customizada de preços.

Body:

{
  "prices": [100,101,102... 60 valores],
  "n_steps_ahead": 1
}

✔ GET /metrics

Retorna métricas do último treinamento.

{
  "symbol": "DIS",
  "mae_test": 1.51,
  "rmse_test": 2.07,
  "mape_test": 1.64
}

🧠 8. Arquitetura do Modelo LSTM (PyTorch)
Input → LSTM(64 units, 2 layers) → ReLU → Dense(32) → Dense(1)


time_steps = 60

Aprendizado temporal usando duas camadas LSTM

Normalização MinMax

Saída: próximo preço de fechamento

🔬 9. Métricas Utilizadas

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

MAPE – Mean Absolute Percentage Error

🚧 10. Melhorias futuras

Deploy com Docker

Deploy em nuvem (Railway, Render, AWS)

Frontend React para exibir gráficos

Suporte a múltiplos modelos

Previsão multi-step (7 dias, 30 dias)

🏁 11. Licença

Código livre para uso acadêmico e profissional.