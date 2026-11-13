import pandas as pd
import yfinance as yf

df = pd.read_csv("data/yahoo_adj_close_1980_2019.csv", index_col=0, parse_dates=True)
print("JNJ first non-null index:", df["JNJ"].first_valid_index())
print("JNJ nulls count:", df["JNJ"].isna().sum())
print(df[df["JNJ"].isna()].tail(10))
print(df[df["JNJ"].notna()].head(10))

# Also check JNJ daily data and Close vs Adj Close in 1980-1985
jnj_mo = yf.download("JNJ", start="1979-12-01", end="1985-06-01", interval="1mo", auto_adjust=False, progress=False)
print("\nJNJ monthly 1979-12 to 1985-06:")
print(jnj_mo.head(10))
print(jnj_mo.tail(10))

jnj_d = yf.download("JNJ", start="1980-01-01", end="1980-02-15", interval="1d", auto_adjust=False, progress=False)
print("\nJNJ daily Jan-Feb 1980:")
print(jnj_d.head(10))
