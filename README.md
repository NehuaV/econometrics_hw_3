# Financial Econometrics — Homework 3

Empirical estimation and analysis of the **Fama–French 3-Factor Model** for **IBM**, **General Electric (`GE`)**, and **Johnson & Johnson (`JNJ`)** over **January 1980 – December 2019** ($T = 479$ monthly excess returns).

## Project Structure

- `main_analysis.ipynb` — Executed step-by-step Jupyter notebook (with inline tables and graphs) ready for submission.
- `main_analysis.py` — Standalone end-to-end Python script reproducing the same pipeline.
- `data/` — Raw and cleaned datasets (`F-F_Research_Data_Factors.csv`, Yahoo Finance Adjusted Close prices, and aligned monthly returns).
- `output/` — Generated summary CSV tables and vector PDF plots.
- `report/` — LaTeX source (`homework3_report.tex`) and compiled PDF report (`homework3_report.pdf`).
- `scripts/` — Exploratory validation and `statsmodels` cross-check scripts.
- `doc/documentation.md` — Detailed implementation and handoff notes.

## Running the Project

```bash
# Install dependencies (for pip / Jupyter)
pip install -r requirements.txt

# Or run the standalone script via uv
uv run econometrics-hw-3
```
