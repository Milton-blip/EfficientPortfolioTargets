# test_env.py
import sys, pandas as pd, numpy as np, matplotlib.pyplot as plt
print("Python", sys.version)
print("pandas", pd.__version__)
print("Total Value Check:", pd.read_csv("holdings.csv")["Value"].sum())
plt.plot([0,1],[0,1]); plt.title("Matplotlib OK"); plt.savefig("ok.png", dpi=120)
print("Plot saved: ok.png")