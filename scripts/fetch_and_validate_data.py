"""Download Yahoo Finance monthly prices and validate both datasets."""

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
FF_PATH = DATA_DIR / "F-F_Research_Data_Factors.csv"

# 1. Inspect Yahoo Finance download with auto_adjust=False to get explicit Adj Close
tickers = ["IBM", "GE", "JNJ"]
raw = yf.download(
    tickers,
    start="1980-01-01",
    end="2020-01-01",
    interval="1mo",
    auto_adjust=False,
    progress=False,
)

print("Raw columns:", raw.columns)
adj_close = raw["Adj Close"][tickers].copy()
print("Adj Close shape:", adj_close.shape)
print("Head:\n", adj_close.head(5))
print("Tail:\n", adj_close.tail(5))
print("Missing values:\n", adj_close.isna().sum())

# Save raw downloaded adj_close to data/
adj_close.to_csv(DATA_DIR / "yahoo_adj_close_1980_2019.csv")

# 2. Parse Fama-French CSV (monthly block only)
# Read lines until the blank line / Annual Factors header
rows = []
with open(FF_PATH, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s:
            if rows:  # blank line after monthly block
                break
            continue
        if s.startswith("Annual Factors"):
            break
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 5 and (parts[0] == "" or (parts[0].isdigit() and len(parts[0]) == 6)):
            rows.append(parts)

ff_df = pd.DataFrame(rows[1:], columns=["YYYYMM", "Mkt-RF", "SMB", "HML", "RF"])
ff_df["YYYYMM"] = ff_df["YYYYMM"].astype(int)
for col in ["Mkt-RF", "SMB", "HML", "RF"]:
    ff_df[col] = ff_df[col].astype(float)

print("\nFF monthly parsed shape:", ff_df.shape)
print("FF date range:", ff_df["YYYYMM"].min(), "to", ff_df["YYYYMM"].max())
print("FF NaNs:\n", ff_df.isna().sum())

# Slice FF to 198001..201912 and 198002..201912
ff_80_19 = ff_df[(ff_df["YYYYMM"] >= 198001) & (ff_df["YYYYMM"] <= 201912)].copy()
print("FF 198001-201912 shape:", ff_80_19.shape)

# Compute returns from adj_close
returns = adj_close.pct_change().dropna()
print("\nStock returns shape:", returns.shape)
print("Return dates head:", returns.index[:3].tolist())
print("Return dates tail:", returns.index[-3:].tolist())

# Align dates using YYYYMM
ret_yyyymm = returns.index.year * 100 + returns.index.month
ff_ret_slice = ff_df[(ff_df["YYYYMM"] >= 198002) & (ff_df["YYYYMM"] <= 201912)].copy()
print("FF 198002-201912 shape:", ff_ret_slice.shape)
assert len(returns) == len(ff_ret_slice) == 479, f"Length mismatch: {len(returns)} vs {len(ff_ret_slice)}"
assert (ret_yyyymm.values == ff_ret_slice["YYYYMM"].values).all(), "Date alignment mismatch!"

# Unit check: FF is in percent, stock returns are in decimals
print("\n--- Unit Sanity Check (1980-02 to 2019-12) ---")
print("Stock returns mean (decimals):\n", returns.mean())
print("Stock returns std (decimals):\n", returns.std())
print("FF raw mean (percent):\n", ff_ret_slice[["Mkt-RF", "SMB", "HML", "RF"]].mean())
print("FF divided by 100 mean (decimals):\n", (ff_ret_slice[["Mkt-RF", "SMB", "HML", "RF"]] / 100.0).mean())
