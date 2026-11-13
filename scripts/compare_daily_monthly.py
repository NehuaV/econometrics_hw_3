import pandas as pd
import yfinance as yf

tickers = ["IBM", "GE", "JNJ"]
daily = yf.download(
    tickers,
    start="1980-01-01",
    end="2020-01-01",
    interval="1d",
    auto_adjust=False,
    progress=False,
)
daily_adj = daily["Adj Close"][tickers]
print("Daily Adj Close shape:", daily_adj.shape)
print("Daily missing values:\n", daily_adj.isna().sum())
print("Daily head:\n", daily_adj.head(5))
print("Daily tail:\n", daily_adj.tail(5))

# Resample daily to monthly last observation (month-end)
monthly_from_daily = daily_adj.resample("ME").last()
print("\nMonthly from daily shape:", monthly_from_daily.shape)
print("Monthly from daily missing:\n", monthly_from_daily.isna().sum())
print("Monthly from daily head:\n", monthly_from_daily.head(5))

# Compare with 1mo download
mo = pd.read_csv("data/yahoo_adj_close_1980_2019.csv", index_col=0, parse_dates=True)
# Align by YYYY-MM
m1 = monthly_from_daily.copy()
m1.index = m1.index.to_period("M")
m2 = mo.copy()
m2.index = m2.index.to_period("M")

diff = (m1 - m2).abs()
print("\nMax abs diff between daily-resampled-ME and 1mo download:")
print(diff.max())
print("Mean abs diff:")
print(diff.mean())
print("\nSample comparison (2019):")
print("Daily-resampled:\n", m1.tail(5))
print("1mo download:\n", m2.tail(5))
