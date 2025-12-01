import yfinance as yf
import pandas as pd
import sqlite3

# 1) Parâmetros
symbol = 'DIS'
start_date = '2018-01-01'
end_date = '2025-11-26'

# 2) Baixar dados do yfinance
df = yf.download(symbol, start=start_date, end=end_date)

print("Linhas baixadas do yfinance:", df.shape)  # (linhas, colunas)

# Transformar índice em coluna
df = df.reset_index()

# 3) Conectar (ou criar) o banco SQLite
conn = sqlite3.connect('market_data.db')

# 4) Salvar os dados na tabela
tabela = 'disney_prices'
df.to_sql(tabela, conn, if_exists='replace', index=False)

# 5) Conferir quantas linhas ficaram na tabela
cursor = conn.cursor()
cursor.execute(f"SELECT COUNT(*) FROM {tabela}")
qtd = cursor.fetchone()[0]
print(f"Linhas na tabela {tabela}: {qtd}")

conn.close()
print("Fim do script.")
