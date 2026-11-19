# Project Documentation & Agent Handoff Guide

**Project:** Financial Econometrics — Homework 3 (Estimating and Testing Multifactor Pricing Models: Fama–French 3-Factor Model)  
**Target Audience:** Another AI agent or developer taking over, auditing, or reproducing this workspace.  
**Governing Rules:** [AGENTS.md](file:///Users/dobri/git/homework/econometrics_hw_3/AGENTS.md) and assignment specification [econometric_homework_3___4 (1).pdf](file:///Users/dobri/git/homework/econometrics_hw_3/econometric_homework_3___4%20(1).pdf).

---

## 1. Repository Architecture & File Placement Rationale

Per [AGENTS.md](file:///Users/dobri/git/homework/econometrics_hw_3/AGENTS.md), the workspace separates exploratory verification, clean production code, data, generated outputs, report sources, and documentation:

```text
├── AGENTS.md                        # Mandatory domain, directory, and econometric rules
├── main_analysis.ipynb              # Executed step-by-step submission notebook (tables + inline plots)
├── main_analysis.py                 # Standalone, reproducible submission script (no scripts/ imports)
├── pyproject.toml / uv.lock         # Python >= 3.15 environment managed via uv (hatchling backend)
├── data/                            # data/*.csv is gitignored; zip archive is tracked
│   ├── F-F_Research_Data_Factors_CSV.zip # Tracked source archive (auto-unzipped on run)
│   ├── F-F_Research_Data_Factors.csv     # Extracted on demand by main_analysis.py
│   ├── yahoo_adj_close_1980_2019.csv     # Cached 480 x 3 monthly Adj Close panel (or downloaded)
│   └── cleaned_dataset_1980_2019.csv     # Final aligned 479-row dataset (198002 to 201912, decimals)
├── scripts/                         # Exploratory & cross-check scripts ONLY
│   ├── fetch_and_validate_data.py
│   ├── check_jnj.py
│   ├── compare_daily_monthly.py
│   └── verify_ols.py
├── output/                          # Generated CSV tables and vector PDF figures
│   ├── descriptive_stats.csv
│   ├── ols_results.csv
│   ├── error_cov_matrix.csv
│   ├── error_corr_matrix.csv
│   ├── hist_stock_excess_returns.pdf
│   ├── hist_stock_raw_returns.pdf
│   ├── hist_ff_factors.pdf
│   └── regression_betas_comparison.pdf
├── report/                          # LaTeX source, build artifacts, and compiled PDF
│   ├── homework3_report.tex
│   └── homework3_report.pdf
└── doc/
    └── documentation.md             # This file
```

**Rationale:**
- `data/*.csv` and `output/*` are listed in `.gitignore`. Tracking `data/F-F_Research_Data_Factors_CSV.zip` keeps the repository lightweight while `ensure_fama_french_csv()` in `main_analysis.py` automatically extracts `data/F-F_Research_Data_Factors.csv` before parsing.
- `main_analysis.py` has zero dependencies on `scripts/` so it serves as a self-contained deliverable.
- `statsmodels` was installed solely for independent cross-checking inside `scripts/verify_ols.py`; `main_analysis.py` never imports `statsmodels`, `sklearn`, or built-in regression wrappers.

---

## 2. Environment & Packaging Setup

Dependencies were installed using `uv`:
```bash
uv add numpy pandas yfinance matplotlib scipy statsmodels
```
Python version is `3.15.2` (`.python-version` = `3.15`). PDF compilation uses `pdflatex` (TeX Live 2026).

In [pyproject.toml](file:///Users/dobri/git/homework/econometrics_hw_3/pyproject.toml), the default `src/econometrics_hw_3/__init__.py` scaffold was removed and the build backend was switched from `uv_build` (which requires a package directory with `__init__.py`) to `hatchling` (`only-include = ["main_analysis.py"]`), with `[project.scripts]` set to:
```toml
econometrics-hw-3 = "main_analysis:main"
```
For portable notebook submission (zip export), a standard [requirements.txt](file:///Users/dobri/git/homework/econometrics_hw_3/requirements.txt) is provided at the project root so the grader can install dependencies with `pip install -r requirements.txt` and run [main_analysis.ipynb](file:///Users/dobri/git/homework/econometrics_hw_3/main_analysis.ipynb).

---

## 3. Data Challenges Found & How They Were Resolved

### 3.1 Fama–French CSV Two-Block Structure (`F-F_Research_Data_Factors.csv`)
**Problem:** The raw CSV (`1312` lines) is not a clean rectangular table:
- **Lines 1–3:** Text description (`CRSP database`, `1-month TBill rate` notes).
- **Line 4:** Blank line.
- **Line 5:** Header `,Mkt-RF,SMB,HML,RF`.
- **Lines 6–1208:** Monthly factor observations (`192607` through `202609`, `1203` rows, `YYYYMM` 6-digit integers).
- **Line 1209:** Blank line.
- **Line 1210:** ` Annual Factors: January-December ` (start of second block).
- **Lines 1211–1310:** Annual factor observations (`1927`–`2025`, 4-digit years).
- **Lines 1311–1312:** Blank line + `Copyright 2026 Eugene F. Fama and Kenneth R. French`.

**Solution (`ensure_fama_french_csv` & `load_fama_french_monthly` in [main_analysis.py](file:///Users/dobri/git/homework/econometrics_hw_3/main_analysis.py)):**
- Before loading the CSV, `ensure_fama_french_csv(csv_path, FF_ZIP_PATH)` checks whether `data/F-F_Research_Data_Factors_CSV.zip` exists and extracts `F-F_Research_Data_Factors.csv` into `data/` safely (writing directly via `zipfile.ZipFile.open`) if the CSV is missing or older than the archive.
- Streams lines from the top, skips leading blank lines and non-table metadata, collects the header + 6-digit `YYYYMM` monthly rows, and **terminates immediately** on the blank line (or `Annual Factors` header) after monthly rows begin.
- Converts `Mkt-RF`, `SMB`, `HML`, and `RF` from percentage points to **decimals** by dividing by `100.0` immediately on load.
- Asserts zero `NaN` values across all `1203` parsed monthly rows.

### 3.2 Yahoo Finance `1mo` Missing History for `JNJ` (1980–1984)
**Problem Discovered (`scripts/fetch_and_validate_data.py` & `scripts/check_jnj.py`):**
When downloading `yf.download(["IBM", "GE", "JNJ"], start="1980-01-01", end="2020-01-01", interval="1mo", auto_adjust=False)`:
- `IBM` and `GE` returned all `480` monthly observations (`1980-01-01` to `2019-12-01`).
- **`JNJ` returned `NaN` for the first 60 months (`1980-01` to `1984-12`)**, starting only at `1985-01-01` (`419` returns instead of `479`).

**Investigation (`scripts/compare_daily_monthly.py`):**
- Querying Yahoo Finance with `interval="1d"` returned `10,087` daily trading observations from `1980-01-02` to `2019-12-31` with **0 missing values** for all three tickers (`IBM`, `GE`, `JNJ`).
- Resampling daily `Adj Close` to month-end (`resample("ME").last()`) produced the exact `480 x 3` monthly panel (`1980-01-31` to `2019-12-31`).
- Comparing daily-resampled month-end prices against Yahoo's `1mo` `Adj Close` (where `1mo` exists) showed max absolute differences of `0.000076` (`IBM`), `0.000092` (`GE`), and `0.000046` (`JNJ`) — i.e., identical up to float rounding.

**Solution (`load_or_download_stock_prices` in [main_analysis.py](file:///Users/dobri/git/homework/econometrics_hw_3/main_analysis.py)):**
- Daily `Adj Close` prices are downloaded with `auto_adjust=False` and resampled to month-end (`ME`), then cached to [yahoo_adj_close_1980_2019.csv](file:///Users/dobri/git/homework/econometrics_hw_3/data/yahoo_adj_close_1980_2019.csv).
- `main_analysis.py` loads the cached CSV if present (allowing offline/sandboxed reproduction) and downloads + caches it automatically if absent.
- Asserts `adj_close.shape == (480, 3)` and zero `NaN`s.

### 3.3 Return Construction, Alignment & Units
**Implementation (`prepare_aligned_dataset` in [main_analysis.py](file:///Users/dobri/git/homework/econometrics_hw_3/main_analysis.py)):**
1. Simple monthly returns: $R_{i,t} = P_{i,t} / P_{i,t-1} - 1$ via `adj_close.pct_change().dropna()`. Dropping `Jan 1980` yields $T = 479$ observations (`1980-02` to `2019-12`).
2. Fama–French monthly data is sliced to `198002 <= YYYYMM <= 201912`.
3. Strict assertions verify `len(stock_returns) == len(ff_sample) == 479` and exact element-wise equality of `YYYYMM` integer dates (`stock_yyyymm == ff_sample["YYYYMM"]`).
4. Excess returns are computed in decimals: $y_{i,t} = R_{i,t} - RF_t$.
5. Combined cleaned dataset is exported to [cleaned_dataset_1980_2019.csv](file:///Users/dobri/git/homework/econometrics_hw_3/data/cleaned_dataset_1980_2019.csv).

---

## 4. Econometric Implementation & Verification

### 4.1 Manual OLS Implementation (`manual_ols_ff3` in [main_analysis.py](file:///Users/dobri/git/homework/econometrics_hw_3/main_analysis.py))
Following Section 3.1 of the assignment and [AGENTS.md](file:///Users/dobri/git/homework/econometrics_hw_3/AGENTS.md):
- $T = 479$, $K = 3$
- $S = \mathbf{1}_{479 \times 1}$, $X = [\text{Mkt-RF}, \text{SMB}, \text{HML}]_{479 \times 3}$, augmented matrix $\tilde{X} = [S, X]_{479 \times 4}$
- For each firm $i \in \{\text{IBM}, \text{GE}, \text{JNJ}\}$ with $y_i \in \mathbb{R}^{479 \times 1}$:
  - Parameter vector: $\hat{\theta}_i = (\tilde{X}'\tilde{X})^{-1}\tilde{X}'y_i = [\hat{\alpha}_i, \hat{\beta}_{1,i}, \hat{\beta}_{2,i}, \hat{\beta}_{3,i}]'$
  - Residuals: $\hat{\varepsilon}_i = y_i - \tilde{X}\hat{\theta}_i$
  - Degrees of freedom: $\text{df} = T - K - 1 = 475$
  - Residual variance: $\hat{\sigma}_i^2 = (\hat{\varepsilon}_i'\hat{\varepsilon}_i) / 475$, yielding $\hat{V}(\varepsilon_i) = \hat{\sigma}_i^2 I_{479}$
  - Parameter covariance: $\widehat{\text{Var}}(\hat{\theta}_i) = \hat{\sigma}_i^2 (\tilde{X}'\tilde{X})^{-1}$
  - Standard errors, $t$-stats, two-sided $p$-values (`2 * scipy.stats.t.sf(|t|, df=475)`), 95% CIs, $R^2$, and adjusted $R^2$.
- Also computes the $3 \times 3$ cross-firm residual variance–covariance matrix $\hat{\Sigma}_{\varepsilon} = \frac{1}{475}\hat{E}'\hat{E}$ and correlation matrix $\widehat{\text{Corr}}(\varepsilon)$.

### 4.2 Independent Verification (`scripts/verify_ols.py`)
`scripts/verify_ols.py` ran `statsmodels.OLS(y_i, X_tilde).fit()` for all three firms and asserted `np.allclose(..., atol=1e-12)` against the manual NumPy formulas for:
- `params` ($\hat{\alpha}_i, \hat{\beta}_i$)
- `bse` (standard errors)
- `tvalues` ($t$-statistics)
- `mse_resid` ($\hat{\sigma}_i^2$)
- `rsquared` and `rsquared_adj`

All assertions passed with exit code `0`.

---

## 5. Visualization & Report Generation

### 5.1 Vector PDF Plots (`output/*.pdf`)
Originally saved at 300 DPI PNG, all four plots were switched to **vector PDF** (`fig.savefig(..., bbox_inches="tight")`) per user request so LaTeX embeds crisp, resolution-independent graphics:
1. `output/hist_stock_excess_returns.pdf` — 3-panel histogram of $y_{i,t}$ (%) with normal fit, mean line, and summary box.
2. `output/hist_stock_raw_returns.pdf` — 3-panel histogram of $R_{i,t}$ (%).
3. `output/hist_ff_factors.pdf` — 3-panel histogram of `Mkt-RF`, `SMB`, and `HML` (%).
4. `output/regression_betas_comparison.pdf` — 2-panel bar chart of $\hat{\alpha}_i$ (%) and $\hat{\beta}_i$ with 95% confidence intervals.

### 5.2 LaTeX Report (`report/homework3_report.tex` $\to$ `report/homework3_report.pdf`)
- Compiled with two passes of `pdflatex -interaction=nonstopmode homework3_report.tex` inside `report/`.
- Contains full data cleaning explanations, descriptive stats table, all histograms, exact algebraic answers to 3.1.a–3.1.d, OLS results table, $\hat{V}(\varepsilon_i) = \hat{\sigma}_i^2 I_{479}$, $3 \times 3$ cross-firm error covariance/correlation matrices, and statistical + economic interpretation for IBM, GE, and JNJ.
- Verified zero LaTeX errors, zero overfull hboxes, and visually inspected all 6 rendered pages.

---

## 6. How to Reproduce End-to-End

1. **Refresh Yahoo Finance data (optional, requires internet):** Delete `data/yahoo_adj_close_1980_2019.csv` and run `uv run python main_analysis.py`.
2. **Re-run full analysis (works offline using cached CSVs):**
   ```bash
   uv run econometrics-hw-3
   # or: uv run python main_analysis.py
   ```
3. **Cross-check manual OLS against `statsmodels`:**
   ```bash
   uv run python scripts/verify_ols.py
   ```
4. **Recompile PDF report:**
   ```bash
   cd report && pdflatex -interaction=nonstopmode homework3_report.tex && pdflatex -interaction=nonstopmode homework3_report.tex
   ```
