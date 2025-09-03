import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection
engine = create_engine("postgresql+psycopg2://postgres:add_password@localhost:5432/detection")

# Load all data from transactions table
query = "SELECT * FROM transactions LIMIT 1000;"  # use LIMIT to avoid loading full dataset
df = pd.read_sql(query, engine)

print(df.head())  # preview the first rows
print(f"Shape of data loaded: {df.shape}")

