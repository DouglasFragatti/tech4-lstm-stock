import sqlite3
import pandas as pd

conn = sqlite3.connect('market_data.db')

df_db = pd.read_sql("delete * FROM disney_prices limit 10 ", conn)
print(df_db)

conn.close()
