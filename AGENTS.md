
# Project Overview & Context

**Course:** Financial Econometrics (Fall Semester) — Università della Svizzera italiana (USI)  
**Instructor:** Prof. Loriano Mancini | **TA:** Dardan Gashi  
**Assignment:** Homework 3 — Estimating and Testing Multifactor Pricing Models (Fama-French 3-Factor Model)  
**Deliverables:**
1. A concise, well-formatted PDF report (LaTeX or Word) containing all mathematical derivations, numerical results, descriptive statistics, plots, and economic/statistical interpretations.
2. A clean, reproducible code file (`.py`, `.R`, or `.m`) containing the final pipeline used to generate all results.

---

# Workspace & Directory Rules

To keep the repository clean and ensure the final deliverable script is easily identifiable, adhere strictly to the following directory structure:

```text
├── data/               # Raw and cleaned datasets (Fama-French CSV, Yahoo Finance prices)
├── scripts/            # Temporary, exploratory, and data-validation scripts ONLY
├── output/             # Generated figures (histograms), LaTeX tables, and intermediate logs
├── report/             # LaTeX (.tex) source files and compiled final PDF report
├── doc/                # Project documentation and agent handoff guide (doc/documentation.md)
├── main_analysis.py    # (or .R / .m) The clean, standalone final code submission
└── Agents.md           # This instruction file
```

### Mandatory File Placement Rules
* **Documentation:** Detailed implementation notes, data anomaly resolutions, verification checks, and reproduction instructions are documented in `doc/documentation.md`.
* **Temporary & Exploratory Scripts:** Any scratchpad code, quick sanity checks, API download tests, or data inspection scripts **MUST** be placed inside the `scripts/` directory (e.g., `scripts/check_ff_dates.py`, `scripts/verify_units.py`). Never leave temporary scripts in the root directory.
* **Clean Main Code:** The root directory must only contain the final, well-commented submission script (`main_analysis.*`) that runs end-to-end without relying on anything inside `scripts/`.

---

# Econometric & Data Guardrails

When writing code or drafting interpretations for this project, agents must strictly enforce the following domain-specific rules:

### 1. Data Alignment & Sample Period
* **Assets:** Monthly Adjusted Close prices from Yahoo Finance for **IBM** (`IBM`), **General Electric** (`GE`), and **Johnson & Johnson** (`JNJ`).
* **Factors:** Monthly Fama/French 3 Factors (`Mkt-RF`, `SMB`, `HML`, and `RF`) from the Kenneth R. French Data Library.
* **Timeframe:** January 1, 1980 – December 31, 2019 ($480$ price observations $\rightarrow 479$ monthly return observations).
* **Dimension Check:** Calculating returns from prices drops the first observation (`Jan 1980`). Ensure the Fama-French factor series are sliced to match the exact return dates (`Feb 1980` – `Dec 2019`, $T = 479$). Always assert matching shapes before matrix operations.

### 2. Units of Measurement (Critical Pitfall)
* Kenneth R. French data is reported in **percentages** (e.g., `1.25` means $1.25\%$, or $0.0125$).
* Stock returns calculated from Yahoo Finance prices $(P_t / P_{t-1} - 1)$ are in **decimals** (e.g., `0.0125`).
* **Rule:** Convert both series to the exact same unit (preferably decimals by dividing Fama-French columns by $100$) *before* computing excess returns ($y_{i,t} = R_{i,t} - RF_t$) or running regressions. Document this explicitly in `scripts/` checks and the main script.

### 3. Manual OLS Implementation Mandate
* **DO NOT** use built-in regression functions (e.g., `statsmodels.OLS`, `lm()`, `fitlm()`, `sklearn`) for the final estimation in Section 3.1.c.
* You **MUST** manually construct the design matrix $\tilde{X} = [\mathbf{1}_{T \times 1}, X_{T \times 3}]$ and response vector $y_i$ ($T \times 1$), and compute parameters using linear algebra:
  $$\hat{\theta}_i = \begin{bmatrix} \hat{\alpha}_i \\ \hat{\beta}_i \end{bmatrix} = (\tilde{X}'\tilde{X})^{-1}\tilde{X}'y_i$$
* You may use `statsmodels` or `lm()` inside a temporary script in `scripts/verify_ols.py` solely to cross-check your manual matrix math, standard errors, and residual variance $\hat{\sigma}_i^2 = \frac{\hat{\epsilon}_i'\hat{\epsilon}_i}{T - K - 1}$.

---

# Agent Workflow & Coding Standards

1. **Verify Before Modeling:** Whenever ingesting new data, first write a validation script in `scripts/` to verify:
   * Missing values (`NaN`s) or footer text in the Kenneth French CSV.
   * Date alignment between Yahoo Finance timestamps (often 1st of the month) and French YYYYMM integers.
   * Scale consistency (mean and standard deviation sanity check to confirm percentages vs. decimals).
2. **Matrix Notation Consistency:** Keep variable names in code aligned with the assignment's notation:
   * `T`: Number of time-series return observations ($479$)
   * `K`: Number of factors ($3$)
   * `y_i`: Excess return vector ($T \times 1$)
   * `X`: Factor matrix ($T \times K$)
   * `S` or `iota`: Vector of ones ($T \times 1$)
3. **Output Formatting:** Save high-resolution plots (300 DPI) for return histograms into `output/` and format numerical tables cleanly (4 decimal places for decimals, or clearly labeled percentages) for easy inclusion in the LaTeX/Word report.
4. **Economic Interpretation:** When assisting with the report text, ensure every statistical finding (e.g., significance of $\hat{\alpha}_i$, size of $\hat{\beta}_{MKT}$, positive/negative loadings on `SMB` and `HML`) is paired with economic intuition regarding the specific companies (IBM, GE, JNJ as large-cap blue-chip firms).
