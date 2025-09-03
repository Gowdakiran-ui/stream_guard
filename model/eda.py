# src/eda.py
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv("/content/train_transaction.csv")

# Quick overview
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Distribution of numerical columns
df.hist(figsize=(12, 10))
plt.show()

# conn.close() # This line seems out of place as 'conn' is not defined in this snippet.