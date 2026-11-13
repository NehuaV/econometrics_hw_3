"""
Financial Econometrics — Homework 3
Estimating and Testing Multifactor Pricing Models (Fama-French 3-Factor Model)

This standalone script reproduces all data preparation, descriptive statistics,
histograms, manual OLS estimations, error variance-covariance matrices, and
summary outputs used in the final PDF report.

Notes on data formatting:
- Fama-French CSV (`F-F_Research_Data_Factors.csv`) contains 3 description lines
  at the top, followed by the monthly data block (`192607`–`202609`), a blank
  line, an annual data block starting at line 1210 (`Annual Factors: ...`),
  and a copyright footer. Only the monthly block is parsed, and values (reported
  in percent) are divided by 100 to convert to decimals.
- Yahoo Finance `1mo` endpoint omits `JNJ` prices prior to Jan 1985. To obtain
  the complete Jan 1980 – Dec 2019 (480 monthly observations) Adjusted Close
  series for all three tickers (`IBM`, `GE`, `JNJ`), daily Adjusted Close prices
  are downloaded and resampled to month-end (`ME`) closing prices (which match
  Yahoo's `1mo` Adjusted Close where both exist to within < 1e-4).
"""

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKERS = ["IBM", "GE", "JNJ"]
START_DATE = "1980-01-01"
END_DATE = "2020-01-01"  # exclusive upper bound -> ends Dec 31, 2019
START_YYYYMM = 198002  # first return observation (Jan 1980 lost to pct_change)
END_YYYYMM = 201912

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
FF_ZIP_PATH = DATA_DIR / "F-F_Research_Data_Factors_CSV.zip"
FF_CSV_PATH = DATA_DIR / "F-F_Research_Data_Factors.csv"
YAHOO_CSV_PATH = DATA_DIR / "yahoo_adj_close_1980_2019.csv"


# ---------------------------------------------------------------------------
# 1. Data Ingestion & Cleaning
# ---------------------------------------------------------------------------
def ensure_fama_french_csv(csv_path: Path, zip_path: Path) -> Path:
    """Extract Fama-French CSV from zip into data/ when missing or outdated."""
    if zip_path.exists() and (
        not csv_path.exists() or zip_path.stat().st_mtime > csv_path.stat().st_mtime
    ):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if Path(m).name == csv_path.name]
            if not members:
                raise FileNotFoundError(
                    f"{csv_path.name} not found inside archive {zip_path}"
                )
            with zf.open(members[0]) as src, open(csv_path, "wb") as dst:
                dst.write(src.read())
    return csv_path


def load_fama_french_monthly(filepath: Path) -> pd.DataFrame:
    """Parse only the monthly block of F-F_Research_Data_Factors.csv and convert % to decimals."""
    filepath = ensure_fama_french_csv(filepath, FF_ZIP_PATH)
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find Fama-French CSV at {filepath}")

    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                # Stop at the blank line separating monthly and annual blocks
                if rows:
                    break
                continue
            if stripped.startswith("Annual Factors"):
                break
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) == 5 and (
                parts[0] == "" or (parts[0].isdigit() and len(parts[0]) == 6)
            ):
                rows.append(parts)

    if len(rows) < 2:
        raise ValueError("Failed to parse monthly data rows from Fama-French CSV.")

    ff_df = pd.DataFrame(rows[1:], columns=["YYYYMM", "Mkt-RF", "SMB", "HML", "RF"])
    ff_df["YYYYMM"] = ff_df["YYYYMM"].astype(int)

    # Convert percentage points (e.g., 1.25) to decimal returns (0.0125)
    factor_cols = ["Mkt-RF", "SMB", "HML", "RF"]
    for col in factor_cols:
        ff_df[col] = ff_df[col].astype(float) / 100.0

    if ff_df.isna().any().any():
        raise ValueError("Unexpected missing values in Fama-French monthly data.")

    return ff_df


def load_or_download_stock_prices(csv_path: Path, tickers: list[str]) -> pd.DataFrame:
    """Load cached monthly Adjusted Close prices or download daily prices from Yahoo Finance."""
    if csv_path.exists():
        adj_close = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        adj_close = adj_close[tickers]
    else:
        import yfinance as yf

        raw = yf.download(
            tickers,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        adj_close = raw["Adj Close"][tickers].resample("ME").last()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        adj_close.to_csv(csv_path)

    assert adj_close.shape == (480, len(tickers)), (
        f"Expected 480 monthly price observations, got {adj_close.shape}"
    )
    assert not adj_close.isna().any().any(), (
        "Missing values detected in Adjusted Close prices."
    )
    return adj_close


def prepare_aligned_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute monthly returns, align sample period (Feb 1980 - Dec 2019), and build excess returns."""
    adj_close = load_or_download_stock_prices(YAHOO_CSV_PATH, TICKERS)
    ff_all = load_fama_french_monthly(FF_CSV_PATH)

    # Simple monthly returns: R_{i,t} = P_{i,t} / P_{i,t-1} - 1 (drops Jan 1980)
    stock_returns = adj_close.pct_change().dropna()
    stock_yyyymm = (
        stock_returns.index.year * 100 + stock_returns.index.month
    ).to_numpy()

    # Slice Fama-French factors to match exact return dates (198002 .. 201912)
    ff_sample = ff_all[
        (ff_all["YYYYMM"] >= START_YYYYMM) & (ff_all["YYYYMM"] <= END_YYYYMM)
    ].reset_index(drop=True)

    # Strict dimension and date alignment assertions
    assert len(stock_returns) == 479, (
        f"Expected T=479 return observations, got {len(stock_returns)}"
    )
    assert len(ff_sample) == 479, (
        f"Expected T=479 factor observations, got {len(ff_sample)}"
    )
    assert np.array_equal(stock_yyyymm, ff_sample["YYYYMM"].to_numpy()), (
        "Date alignment mismatch between stock returns and Fama-French factors."
    )

    # Monthly excess returns: y_{i,t} = R_{i,t} - RF_t (both in decimals)
    excess_returns = pd.DataFrame(index=stock_returns.index)
    for ticker in TICKERS:
        excess_returns[ticker] = (
            stock_returns[ticker].to_numpy() - ff_sample["RF"].to_numpy()
        )

    # Save unified cleaned dataset
    cleaned = pd.DataFrame({"YYYYMM": ff_sample["YYYYMM"].to_numpy()})
    for ticker in TICKERS:
        cleaned[f"R_{ticker}"] = stock_returns[ticker].to_numpy()
        cleaned[f"y_{ticker}"] = excess_returns[ticker].to_numpy()
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        cleaned[col] = ff_sample[col].to_numpy()
    cleaned.to_csv(DATA_DIR / "cleaned_dataset_1980_2019.csv", index=False)

    return stock_returns, excess_returns, ff_sample


# ---------------------------------------------------------------------------
# 2. Descriptive Statistics & Plots
# ---------------------------------------------------------------------------
def compute_descriptive_stats(
    stock_returns: pd.DataFrame,
    excess_returns: pd.DataFrame,
    ff_sample: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly and annualized summary statistics for returns, excess returns, and factors."""
    series_map = {
        "IBM (Raw Return)": stock_returns["IBM"].to_numpy(),
        "GE (Raw Return)": stock_returns["GE"].to_numpy(),
        "JNJ (Raw Return)": stock_returns["JNJ"].to_numpy(),
        "IBM (Excess Return)": excess_returns["IBM"].to_numpy(),
        "GE (Excess Return)": excess_returns["GE"].to_numpy(),
        "JNJ (Excess Return)": excess_returns["JNJ"].to_numpy(),
        "Mkt-RF": ff_sample["Mkt-RF"].to_numpy(),
        "SMB": ff_sample["SMB"].to_numpy(),
        "HML": ff_sample["HML"].to_numpy(),
        "RF": ff_sample["RF"].to_numpy(),
    }

    records = []
    for name, arr in series_map.items():
        mean_m = float(np.mean(arr))
        std_m = float(np.std(arr, ddof=1))
        records.append(
            {
                "Series": name,
                "N": len(arr),
                "Mean": mean_m,
                "Std": std_m,
                "Min": float(np.min(arr)),
                "Median": float(np.median(arr)),
                "Max": float(np.max(arr)),
                "Skewness": float(stats.skew(arr, bias=False)),
                "Excess_Kurtosis": float(stats.kurtosis(arr, fisher=True, bias=False)),
                "Ann_Mean": 12.0 * mean_m,
                "Ann_Std": np.sqrt(12.0) * std_m,
            }
        )

    desc_df = pd.DataFrame(records)
    desc_df.to_csv(OUTPUT_DIR / "descriptive_stats.csv", index=False)
    return desc_df


def plot_histograms(
    stock_returns: pd.DataFrame,
    excess_returns: pd.DataFrame,
    ff_sample: pd.DataFrame,
) -> None:
    """Generate vector PDF histograms for excess stock returns, raw stock returns, and FF3 factors."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # 1. Excess stock returns histogram
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    colors = ["#1f4e79", "#b03a2e", "#1e8449"]
    for ax, ticker, color in zip(axes, TICKERS, colors):
        vals = excess_returns[ticker].to_numpy() * 100.0  # plot in % for readability
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1)
        ax.hist(
            vals,
            bins=30,
            density=True,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.6,
        )
        x_grid = np.linspace(vals.min() - 2, vals.max() + 2, 300)
        ax.plot(
            x_grid,
            stats.norm.pdf(x_grid, mean_v, std_v),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Normal fit",
        )
        ax.axvline(
            mean_v,
            color="#111111",
            linestyle="-",
            linewidth=1.1,
            label=f"Mean = {mean_v:.2f}%",
        )
        ax.set_title(f"{ticker} Excess Return ($y_{{i,t}}$)")
        ax.set_xlabel("Monthly Excess Return (%)")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.text(
            0.97,
            0.93,
            f"SD = {std_v:.2f}%\nMin = {vals.min():.1f}%\nMax = {vals.max():.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="#cccccc",
            ),
        )
    axes[0].set_ylabel("Density")
    fig.suptitle(
        "Monthly Excess Returns Distribution (Feb 1980 – Dec 2019, T = 479)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "hist_stock_excess_returns.pdf", bbox_inches="tight"
    )
    plt.close(fig)

    # 2. Raw stock returns histogram
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, ticker, color in zip(axes, TICKERS, colors):
        vals = stock_returns[ticker].to_numpy() * 100.0
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1)
        ax.hist(
            vals,
            bins=30,
            density=True,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.6,
        )
        x_grid = np.linspace(vals.min() - 2, vals.max() + 2, 300)
        ax.plot(
            x_grid,
            stats.norm.pdf(x_grid, mean_v, std_v),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Normal fit",
        )
        ax.axvline(
            mean_v,
            color="#111111",
            linestyle="-",
            linewidth=1.1,
            label=f"Mean = {mean_v:.2f}%",
        )
        ax.set_title(f"{ticker} Raw Return ($R_{{i,t}}$)")
        ax.set_xlabel("Monthly Return (%)")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.text(
            0.97,
            0.93,
            f"SD = {std_v:.2f}%\nMin = {vals.min():.1f}%\nMax = {vals.max():.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="#cccccc",
            ),
        )
    axes[0].set_ylabel("Density")
    fig.suptitle(
        "Monthly Raw Stock Returns Distribution (Feb 1980 – Dec 2019, T = 479)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hist_stock_raw_returns.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. Fama-French 3 factors histogram
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    factor_names = ["Mkt-RF", "SMB", "HML"]
    f_colors = ["#2c3e50", "#8e44ad", "#d35400"]
    for ax, fname, color in zip(axes, factor_names, f_colors):
        vals = ff_sample[fname].to_numpy() * 100.0
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1)
        ax.hist(
            vals,
            bins=30,
            density=True,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.6,
        )
        x_grid = np.linspace(vals.min() - 2, vals.max() + 2, 300)
        ax.plot(
            x_grid,
            stats.norm.pdf(x_grid, mean_v, std_v),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Normal fit",
        )
        ax.axvline(
            mean_v,
            color="#111111",
            linestyle="-",
            linewidth=1.1,
            label=f"Mean = {mean_v:.2f}%",
        )
        ax.set_title(f"Factor: {fname}")
        ax.set_xlabel("Monthly Factor Return (%)")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.text(
            0.97,
            0.93,
            f"SD = {std_v:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="#cccccc",
            ),
        )
    axes[0].set_ylabel("Density")
    fig.suptitle(
        "Fama-French Three Factors Distribution (Feb 1980 – Dec 2019, T = 479)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hist_ff_factors.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Manual OLS Estimation (Section 3.1)
# ---------------------------------------------------------------------------
def manual_ols_ff3(
    excess_returns: pd.DataFrame,
    ff_sample: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """
    Manually estimate y_i = S * alpha_i + X * beta_i + eps_i via OLS:
        theta_hat = (X_tilde' X_tilde)^{-1} X_tilde' y_i
    where X_tilde = [S, X], S = 1_{T x 1}, X = [Mkt-RF, SMB, HML]_{T x 3}.
    """
    T = len(excess_returns)
    K = 3
    assert T == 479

    # Construct S (T x 1), X (T x K), and augmented design matrix X_tilde (T x (K+1))
    S = np.ones((T, 1), dtype=float)
    X = ff_sample[["Mkt-RF", "SMB", "HML"]].to_numpy(dtype=float)
    assert X.shape == (T, K)

    X_tilde = np.hstack([S, X])
    assert X_tilde.shape == (T, K + 1)

    XtX = X_tilde.T @ X_tilde
    XtX_inv = np.linalg.inv(XtX)

    param_names = ["alpha", "beta_MKT", "beta_SMB", "beta_HML"]
    ols_rows = []
    firm_details = {}
    residuals_mat = np.zeros((T, len(TICKERS)), dtype=float)

    for idx, ticker in enumerate(TICKERS):
        y_i = excess_returns[ticker].to_numpy(dtype=float).reshape(T, 1)
        assert y_i.shape == (T, 1)

        # Manual OLS point estimates: theta_hat = (X_tilde' X_tilde)^{-1} X_tilde' y_i
        theta_hat = XtX_inv @ (X_tilde.T @ y_i)
        alpha_hat = float(theta_hat[0, 0])
        beta_hat = theta_hat[1:, 0]

        # Fitted values and residuals
        y_hat = X_tilde @ theta_hat
        eps_hat = y_i - y_hat
        residuals_mat[:, idx] = eps_hat[:, 0]

        # Unbiased residual variance estimate: sigma_hat^2 = (eps_hat' eps_hat) / (T - K - 1)
        # Under i.i.d. errors, V_hat(eps_i) = sigma_hat^2 * I_T
        dof = T - K - 1
        rss = float((eps_hat.T @ eps_hat)[0, 0])
        sigma2_hat = rss / dof
        sigma_hat = float(np.sqrt(sigma2_hat))

        # Parameter covariance matrix: Var_hat(theta_hat) = sigma_hat^2 * (X_tilde' X_tilde)^{-1}
        cov_theta_hat = sigma2_hat * XtX_inv
        se_theta_hat = np.sqrt(np.diag(cov_theta_hat))

        # t-statistics and two-sided p-values (Student's t with T - K - 1 = 475 df)
        t_stats = theta_hat[:, 0] / se_theta_hat
        p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)

        # R-squared and Adjusted R-squared
        y_centered = y_i - np.mean(y_i)
        tss = float((y_centered.T @ y_centered)[0, 0])
        r2 = 1.0 - rss / tss
        adj_r2 = 1.0 - (1.0 - r2) * (T - 1) / dof

        firm_details[ticker] = {
            "theta_hat": theta_hat[:, 0],
            "alpha_hat": alpha_hat,
            "beta_hat": beta_hat,
            "se": se_theta_hat,
            "t_stats": t_stats,
            "p_vals": p_vals,
            "sigma2_hat": sigma2_hat,
            "sigma_hat": sigma_hat,
            "cov_theta_hat": cov_theta_hat,
            "R2": r2,
            "Adj_R2": adj_r2,
            "RSS": rss,
            "TSS": tss,
            "dof": dof,
        }

        for pname, est, se, tval, pval in zip(
            param_names, theta_hat[:, 0], se_theta_hat, t_stats, p_vals
        ):
            ols_rows.append(
                {
                    "Firm": ticker,
                    "Parameter": pname,
                    "Estimate": est,
                    "Std_Error": se,
                    "t_stat": tval,
                    "p_value": pval,
                    "CI_95_Low": est - stats.t.ppf(0.975, df=dof) * se,
                    "CI_95_High": est + stats.t.ppf(0.975, df=dof) * se,
                    "sigma2_hat": sigma2_hat,
                    "sigma_hat": sigma_hat,
                    "R2": r2,
                    "Adj_R2": adj_r2,
                }
            )

    ols_df = pd.DataFrame(ols_rows)
    ols_df.to_csv(OUTPUT_DIR / "ols_results.csv", index=False)

    # Cross-firm residual variance-covariance matrix (3 x 3) and correlation matrix
    Sigma_eps_hat = pd.DataFrame(
        (residuals_mat.T @ residuals_mat) / (T - K - 1),
        index=TICKERS,
        columns=TICKERS,
    )
    Corr_eps_hat = pd.DataFrame(residuals_mat, columns=TICKERS).corr()
    Sigma_eps_hat.to_csv(OUTPUT_DIR / "error_cov_matrix.csv")
    Corr_eps_hat.to_csv(OUTPUT_DIR / "error_corr_matrix.csv")

    return ols_df, firm_details, Sigma_eps_hat, Corr_eps_hat


def plot_factor_loadings(ols_df: pd.DataFrame) -> None:
    """Create a comparison plot of estimated FF3 betas and 95% confidence intervals across firms."""
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [1, 2.4]}
    )

    # Panel A: Monthly alpha (in %)
    ax_a = axes[0]
    alpha_sub = ols_df[ols_df["Parameter"] == "alpha"].set_index("Firm").loc[TICKERS]
    x_pos = np.arange(len(TICKERS))
    vals_a = alpha_sub["Estimate"].to_numpy() * 100.0
    err_a = (
        alpha_sub["CI_95_High"].to_numpy() - alpha_sub["Estimate"].to_numpy()
    ) * 100.0
    colors = ["#1f4e79", "#b03a2e", "#1e8449"]
    ax_a.bar(
        x_pos,
        vals_a,
        yerr=err_a,
        capsize=5,
        color=colors,
        alpha=0.82,
        edgecolor="black",
        linewidth=0.7,
    )
    ax_a.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(TICKERS)
    ax_a.set_ylabel("Monthly Alpha (%)")
    ax_a.set_title("Jensen's Alpha ($\\hat{\\alpha}_i$) with 95% CI")

    # Panel B: Factor betas
    ax_b = axes[1]
    beta_params = ["beta_MKT", "beta_SMB", "beta_HML"]
    beta_labels = [
        "$\\hat{\\beta}_{MKT}$",
        "$\\hat{\\beta}_{SMB}$",
        "$\\hat{\\beta}_{HML}$",
    ]
    width = 0.24
    x_base = np.arange(len(beta_params))
    for i, (ticker, color) in enumerate(zip(TICKERS, colors)):
        sub = ols_df[
            (ols_df["Firm"] == ticker) & (ols_df["Parameter"].isin(beta_params))
        ]
        sub = sub.set_index("Parameter").loc[beta_params]
        vals = sub["Estimate"].to_numpy()
        errs = sub["CI_95_High"].to_numpy() - vals
        ax_b.bar(
            x_base + (i - 1) * width,
            vals,
            width=width,
            yerr=errs,
            capsize=4,
            color=color,
            alpha=0.82,
            edgecolor="black",
            linewidth=0.7,
            label=ticker,
        )
    ax_b.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax_b.set_xticks(x_base)
    ax_b.set_xticklabels(beta_labels)
    ax_b.set_ylabel("Factor Loading Estimate")
    ax_b.set_title("Fama-French Factor Loadings ($\\hat{\\beta}_i$) with 95% CI")
    ax_b.legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "regression_betas_comparison.pdf", bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_returns, excess_returns, ff_sample = prepare_aligned_dataset()
    desc_df = compute_descriptive_stats(stock_returns, excess_returns, ff_sample)
    plot_histograms(stock_returns, excess_returns, ff_sample)
    ols_df, firm_details, Sigma_eps_hat, Corr_eps_hat = manual_ols_ff3(
        excess_returns, ff_sample
    )
    plot_factor_loadings(ols_df)

    # Print formatted summary
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(
        "=== DESCRIPTIVE STATISTICS (Monthly Decimals, Feb 1980 - Dec 2019, T = 479) ==="
    )
    print(desc_df.round(4).to_string(index=False))

    print("\n=== MANUAL OLS ESTIMATES: FAMA-FRENCH 3-FACTOR MODEL ===")
    print(ols_df.round(6).to_string(index=False))

    print(
        "\n=== FIRM-LEVEL ERROR VARIANCE ESTIMATES V_hat(eps_i) = sigma_hat^2 * I_T ==="
    )
    for ticker in TICKERS:
        d = firm_details[ticker]
        print(
            f"  {ticker:3s}: sigma_hat^2 = {d['sigma2_hat']:.6f}, "
            f"sigma_hat = {d['sigma_hat']:.4f} ({d['sigma_hat'] * 100:.2f}%/mo), "
            f"R^2 = {d['R2']:.4f}, Adj R^2 = {d['Adj_R2']:.4f}"
        )

    print("\n=== CROSS-FIRM RESIDUAL COVARIANCE MATRIX (Sigma_hat) ===")
    print(Sigma_eps_hat.round(6))

    print("\n=== CROSS-FIRM RESIDUAL CORRELATION MATRIX ===")
    print(Corr_eps_hat.round(4))


if __name__ == "__main__":
    main()
