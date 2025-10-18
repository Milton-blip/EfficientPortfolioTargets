import pandas as pd

df = pd.read_csv("holdings.csv")
print(df.head())
print("Total portfolio value: $", df["Value"].sum())