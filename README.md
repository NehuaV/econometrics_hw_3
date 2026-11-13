# Financial Econometrics — Homework 3

Empirical estimation and analysis of the **Fama–French 3-Factor Model** for **IBM**, **General Electric (`GE`)**, and **Johnson & Johnson (`JNJ`)** over **January 1980 – December 2019** ($T = 479$ monthly excess returns).

## Project Structure

- `main_analysis.py` — Standalone end-to-end pipeline (data cleaning, alignment, descriptive statistics, vector PDF histograms, manual OLS estimation, and residual covariance analysis).
- `data/` — Raw and cleaned datasets (`F-F_Research_Data_Factors.csv`, Yahoo Finance Adjusted Close prices, and aligned monthly returns).
- `output/` — Generated summary CSV tables and vector PDF plots.
- `report/` — LaTeX source (`homework3_report.tex`) and compiled PDF report (`homework3_report.pdf`).
- `scripts/` — Exploratory validation and `statsmodels` cross-check scripts.
- `doc/documentation.md` — Detailed implementation and handoff notes.

## Running the Project

```bash
# Run via the configured script entrypoint
uv run econometrics-hw-3

# Or run the standalone script directly
uv run python main_analysis.py
```
