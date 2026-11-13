"""Verify manual OLS implementation against statsmodels and inspect all numerical results."""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

DATA_DIR = Path("data")
FF_PATH = DATA_DIR / "F-F_Research_Data_Factors.csv"
YAHOO_CSV = DATA_DIR / "yahoo_adj_close_1980_2019.csv"

tickers = ["IBM", "GE", "JNJ"]

# Download daily prices and resample to month-end to avoid Yahoo 1mo missing JNJ 1980-1984
daily = yf.download(
    tickers,
    start="1980-01-01",
    end="2020-01-01",
    interval="1d",
    auto_adjust=False,
    progress=False,
)
adj_close = daily["Adj Close"][tickers].resample("ME").last()
assert adj_close.shape == (480, 3)
assert adj_close.isna().sum().sum() == 0
adj_close.to_csv(YAHOO_CSV)
print(f"Saved {YAHOO_CSV} with shape {adj_close.shape}")

# Load Fama-French monthly block
rows = []
with open(FF_PATH, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s:
            if rows:
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
    ff_df[col] = ff_df[col].astype(float) / 100.0  # percent -> decimal

# Compute simple monthly stock returns
returns = adj_close.pct_change().dropna()
ret_yyyymm = returns.index.year * 100 + returns.index.month

ff_slice = ff_df[(ff_df["YYYYMM"] >= 198002) & (ff_df["YYYYMM"] <= 201912)].reset_index(drop=True)
assert len(returns) == len(ff_slice) == 479
assert np.array_equal(ret_yyyymm.values, ff_slice["YYYYMM"].values)

# Excess returns y_i = R_i - RF
excess_returns = returns.copy()
for t in tickers:
    excess_returns[t] = returns[t].values - ff_slice["RF"].values

# Design matrix: X (T x K), S (T x 1), X_tilde = [S, X] (T x (K+1))
T = len(returns)
K = 3
S = np.ones((T, 1))
X = ff_slice[["Mkt-RF", "SMB", "HML"]].to_numpy()
X_tilde = np.hstack([S, X])

residuals_dict = {}

for ticker in tickers:
    y_i = excess_returns[ticker].to_numpy().reshape(T, 1)

    # Manual OLS: theta_hat = (X_tilde' X_tilde)^{-1} X_tilde' y_i
    XtX = X_tilde.T @ X_tilde
    XtX_inv = np.linalg.inv(XtX)
    theta_hat = XtX_inv @ (X_tilde.T @ y_i)
    eps_hat = y_i - X_tilde @ theta_hat

    sigma2_hat = float((eps_hat.T @ eps_hat)[0, 0] / (T - K - 1))
    cov_theta = sigma2_hat * XtX_inv
    se_theta = np.sqrt(np.diag(cov_theta))
    t_stats = theta_hat.flatten() / se_theta

    # R^2 and adj R^2
    y_demean = y_i - y_i.mean()
    tss = float((y_demean.T @ y_demean)[0, 0])
    rss = float((eps_hat.T @ eps_hat)[0, 0])
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * (T - 1) / (T - K - 1)

    residuals_dict[ticker] = eps_hat.flatten()

    # Cross-check with statsmodels
    sm_model = sm.OLS(y_i.flatten(), X_tilde).fit()

    assert np.allclose(theta_hat.flatten(), sm_model.params, atol=1e-12)
    assert np.allclose(se_theta, sm_model.bse, atol=1e-12)
    assert np.allclose(t_stats, sm_model.tvalues, atol=1e-12)
    assert np.isclose(sigma2_hat, sm_model.mse_resid, atol=1e-12)
    assert np.isclose(r2, sm_model.rsquared, atol=1e-12)
    assert np.isclose(adj_r2, sm_model.rsquared_adj, atol=1e-12)

    print(f"\n=== {ticker} (verified against statsmodels) ===")
    names = ["alpha", "beta_MKT", "beta_SMB", "beta_HML"]
    for name, est, se, tval, pval in zip(names, theta_hat.flatten(), se_theta, t_stats, sm_model.pvalues):
        print(f"  {name:10s}: est={est: .6f}  se={se: .6f}  t={tval: .4f}  p={pval: .6f}")
    print(f"  sigma^2_hat = {sigma2_hat:.8f} (sigma_hat = {np.sqrt(sigma2_hat):.6f})")
    print(f"  R^2 = {r2:.6f}, Adj R^2 = {adj_r2:.6f}")

E_hat = np.column_stack([residuals_dict[t] for t in tickers])
Sigma_hat = (E_hat.T @ E_hat) / (T - K - 1)
print("\n=== Cross-Firm Residual Covariance Matrix (df = T - K - 1) ===")
print(pd.DataFrame(Sigma_hat, index=tickers, columns=tickers))
print("\n=== Cross-Firm Residual Correlation Matrix ===")
print(pd.DataFrame(E_hat, columns=tickers).corr())
