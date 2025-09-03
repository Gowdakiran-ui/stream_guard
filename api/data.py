import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
engine = create_engine("postgresql+psycopg2://postgres:GOwda%407924@localhost:5432/detection")

# Correct absolute path
file_path = r"C:\Users\Kiran gowda.A\Downloads\ieee-fraud-detection\train_transaction.csv"

# Load the IEEE fraud CSV in chunks (since it's huge)
chunksize = 100000
for chunk in pd.read_csv(file_path, chunksize=chunksize):
    chunk.to_sql("transactions", engine, if_exists="append", index=False)
