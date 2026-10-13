# Quantitative Trading with Machine Learning: Predicting Volkswagen Returns from Supply Chain Dynamics

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course](https://img.shields.io/badge/Course-UE24CS352A%20Machine%20Learning-indigo.svg)](https://pes.edu)
[![Institution](https://img.shields.io/badge/Institution-PES%20University-crimson.svg)](https://pes.edu)

A machine learning framework for predicting asset returns using inter-firm supply chain spillover effects. Based on the foundational methodology of **René M. Glawion (Stanford University, CS229)**, this project predicts future percentage returns of **Volkswagen AG (`VOW3.DE`)** across horizons $h \in \{1, 5, 20\}$ days using the lagged returns of its Tier-1 automotive suppliers and macroeconomic bellwethers.

---

## 👥 Team Information

| Name | SRN | Contribution |
|---|---|---|
| **Rottela Haritej** | `PES1UG24AM231` | Pipeline engineering, walk-forward validation, backtest simulation, web demo |
| **Pradyun P Urs** | `PES1UG24AM195` | Data collection & hygiene, model architectures, hyperparameter tuning, write-up |

**Course:** UE24CS352A – Machine Learning Mini-Project  
**Department:** Department of Computer Science & Engineering, PES University  
**Evaluation:** Final Review & Demonstration  

---

## 📌 Problem Statement & Research Foundation

Traditional asset pricing models assume that financial markets instantaneously digest all public information under the **Efficient Market Hypothesis (EMH)**. However, modern corporate value chains are intricate economic webs where shocks propagate with structural delay.

In the automotive sector, original equipment manufacturers (OEMs) like Volkswagen depend heavily on Tier-1 suppliers for critical components (semiconductors, braking systems, transmission bearings, coatings). Price shocks hitting suppliers serve as a **lead-lag signal** for the OEM's subsequent stock performance.

We formalize this as a supervised multi-horizon regression problem:
$$\mathbb{E}[y_{t, t+h} \mid \mathcal{F}_t] \approx g(X_t, \theta)$$
where:
- $y_{t, t+h} = \frac{P_{t+h}}{P_t} - 1$ is the realized return of Volkswagen AG over forecast horizon $h \in \{1, 5, 20\}$ days.
- $\mathcal{F}_t$ represents the filtration (information set) strictly available at the close of day $t$.
- $X_t \in \mathbb{R}^p$ denotes the lagged feature vector across the supplier ecosystem.
- $g(\cdot, \theta)$ is an optimized machine learning estimator evaluated under a leak-free walk-forward protocol.

---

## 📊 Dataset & Data Hygiene Protocol

All market series are ingested via Yahoo Finance (`auto_adjust=True`) covering daily trading from **January 2015 to October 2026** (2,797 synchronized market observations).

### Asset Universe
1. **Target Asset:** Volkswagen AG (`VOW3.DE`, Frankfurt Xetra)
2. **Tier-1 Suppliers:**
   - **Continental AG (`CON.DE`):** Tires, braking, and cockpit electronics
   - **Infineon Technologies (`IFX.DE`):** Automotive microcontrollers and power chips
   - **Schaeffler AG (`SHA0.DE`):** Transmission bearings and engine components
   - **Valeo SE (`FR.PA`):** ADAS sensors, LiDAR, and thermal systems
   - **BASF SE (`BAS.DE`):** Automotive OEM coatings and engineering plastics
   - **Michelin (`ML.PA`):** OEM tire fitments
   - **ElringKlinger AG (`ZIL2.DE`):** Cylinder gaskets and EV battery housing
3. **Macroeconomic & Market Proxies:**
   - **DAX 40 Index (`^GDAXI`):** Broad German equity beta
   - **EUR/USD Exchange Rate (`EURUSD=X`):** Export competitiveness proxy

### Strict Zero-Fabrication Missing Data Protocol
- **IPO Thresholding (`MAX_START_DATE = 2016-01-04`):** Companies listing after this date (e.g. Knorr-Bremse `KBX.DE`, IPO late 2018) are excluded to ensure a common, authentic sample period. **We strictly never synthesize pre-listing zero returns.**
- **Calendar Alignment:** Series are aligned to the Volkswagen Frankfurt trading calendar. Exchange holiday mismatches are forward-filled up to a strict 5-day ceiling.
- **Timezone Asynchrony Protection:** Because EUR/USD daily settlement occurs after the Frankfurt equity close, EUR/USD enters features only from **lag $\ge 1$** (`lag0` is omitted to eliminate lookahead bias).

---

## 🛠 Feature Engineering & Model Suite

### Feature Matrix ($p = 50$)
For each retained series, we extract $N_{\text{lags}} = 5$ lagged returns:
$$X_t = \Big[ r_{k, t - l} \Big]_{k \in \mathcal{K}, \, l \in \{0, \dots, 4\}}$$
where `lag0` is the return on prediction date $t$ and `lag1..lag4` represent historical trailing days.

### Evaluated Model Architectures
1. **Zero-Return Baseline:** Martingale random walk benchmark ($\hat{y} = 0$).
2. **Ordinary Least Squares (OLS):** Linear regression with `StandardScaler`.
3. **Elastic Net:** L1/L2 regularized linear model:
   $$\min_{\theta} \frac{1}{2n}\|y - X\theta\|_2^2 + \alpha \Big( \rho \|\theta\|_1 + \frac{1-\rho}{2} \|\theta\|_2^2 \Big)$$
   Groups collinear supplier returns and prevents noise overfitting.
4. **Decision Tree Regressor (CART):** Non-linear binary recursive partitioning.
5. **XGBoost Regressor:** 2nd-order gradient-boosted decision trees with shrinkage and column subsampling.
6. **LightGBM Regressor:** Leaf-wise histogram-based gradient boosting.

---

## 🔬 Walk-Forward Expanding Cross-Validation

```
[Expanding Train Window: >= 5 Years] --> [Validation: 1 Year] --(Gap h)--> [Test Block: 1 Month]
========================================= Walk-Forward Step ========================================>
[==================== Development Phase (Hyperparameter & Threshold Selection) ====================] | [Holdout: ~1 Year]
```

To model real-world deployment without lookahead bias:
1. **Expanding Window:** Initial training starts at 1,260 days (~5 years); validation spans 252 days (~1 year); test block spans 21 days (~1 month) rolled sequentially.
2. **Label Embargo Gap ($h$ Days):** Because an $h$-day return spans until $t+h$, an embargo gap of $h$ days separates train, validation, and test blocks.
3. **Perturbation Audit:** Corrupting all future features and labels with Gaussian noise leaves model predictions bit-identical ($\text{atol} = 10^{-12}$), proving **zero lookahead leakage**.
4. **Quarantined Holdout:** The final ~1 year (252–271 trading days) is quarantined as an untouched holdout. All model choices and trading thresholds are calibrated solely on earlier development data.

---

## 📈 Empirical Results & Performance

### 1. Predictive Accuracy Horse-Race (Holdout Period)

| Model Architecture | Horizon ($h$) | RMSE (Dec) | MAE (Dec) | OOS $R^2$ | Directional Acc. | Inference Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zero Baseline** | 1 | **0.0189** | **0.0141** | **0.000** | – | 0.0 ms |
| **Elastic Net** | 1 | 0.0189 | 0.0142 | -0.003 | 46.5% | 191.1 ms |
| **LightGBM** | 1 | 0.0190 | 0.0142 | -0.007 | **56.8%** | 411.2 ms |
| **Decision Tree** | 1 | 0.0191 | 0.0144 | -0.023 | 46.5% | 385.8 ms |
| **XGBoost** | 1 | 0.0192 | 0.0144 | -0.027 | 50.2% | 786.0 ms |
| **Linear Regression** | 1 | 0.0196 | 0.0149 | -0.077 | 45.8% | 39.9 ms |
| **Zero Baseline** | 5 | **0.0393** | **0.0310** | **0.000** | – | 0.0 ms |
| **Elastic Net** | 5 | 0.0396 | 0.0312 | -0.016 | 44.9% | 189.2 ms |
| **Zero Baseline** | 20 | **0.0756** | **0.0598** | **0.000** | – | 0.0 ms |
| **Elastic Net** | 20 | 0.0770 | 0.0607 | -0.037 | 44.8% | 193.3 ms |

> **Key Finding:** In financial time series with high noise and low signal-to-noise ratio, **out-of-sample $R^2$ is negative across all horizons, and no ML model beats the zero-return baseline in holdout RMSE**. While LightGBM achieves 56.8% directional accuracy at $h=1$, directional accuracy alone does not imply lower squared prediction error.

### 2. Systematic Trading Performance (Untouched Holdout, €1,000 Capital, 0.1% Friction)

| Strategy / Model | Horizon ($h$) | Threshold ($c$) | Net Profit % | Trades | Sharpe Ratio | Max Drawdown |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Buy & Hold (VW)** | 1 | – | **-28.61%** | 1 | -0.89 | -34.58% |
| **Buy & Hold (VW)** | 5 | – | **-28.04%** | 1 | -0.89 | -33.72% |
| **Buy & Hold (VW)** | 20 | – | **-14.48%** | 1 | -0.38 | -31.30% |
| **Cash Benchmark** | – | – | **0.00%** | 0 | 0.00 | 0.00% |
| **Linear Regression** | 1 | 1.00% | **+21.30%** | 8 | **0.77** | -24.73% |
| **Decision Tree** | 5 | 0.50% | **+23.97%** | 12 | **0.86** | -19.69% |
| **LightGBM** | 5 | 0.50% | **+14.56%** | 23 | 0.59 | -17.49% |
| **Elastic Net** | 1 | 0.50% | **0.00% (Cash)** | 0 | 0.00 | 0.00% |
| **Elastic Net** | 20 | 0.50% | **-7.88%** | 6 | -0.13 | -26.29% |

> **Methodology & Key Finding:** The model forecasts multi-day forward returns $E[y_{t, t+h}]$, while the daily backtest applies the resulting position to next-day returns ($t+1$) and updates daily. During Volkswagen's severe holdout bear market (-28.61% Buy & Hold), threshold filtering successfully preserved capital. Some threshold-based strategies achieved positive net returns (+21.30% for Linear Regression and +23.97% for Decision Tree), but these profits derive from a small number of trades (8 and 12 trades) over a single 1-year holdout window. They represent regime-specific exploratory alpha rather than proof of a generally profitable trading system.

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Internet access for initial Yahoo Finance cache download

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd ML-miniproject

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

#### 1. Command-Line Automated Pipeline
Run the full ML training, walk-forward validation, and backtest pipeline:
```bash
python run_pipeline.py
```

#### 2. Interactive Jupyter Notebook
Open and run all cells in Jupyter Notebook:
```bash
jupyter notebook Quantitative_Trading_Final.ipynb
```
*(Select `Kernel → Restart & Run All`)*

#### 3. Launch the Web Demonstration Application
Start the interactive quantitative trading simulation UI:
```bash
python web/server.py
```
Then navigate to `http://localhost:8000` in your browser.

#### 4. Compile the 2-Page PDF Write-up
Generate the formal academic write-up:
```bash
python reports/generate_pdf.py
```
*(Outputs to `reports/UE24CS352A_MiniProject_Writeup.pdf`)*

---

## 📂 Repository Structure

```
ML-miniproject/
├── Quantitative_Trading_Final.ipynb  # Core Jupyter notebook with full ML workflow
├── run_pipeline.py                   # Automated CLI pipeline execution script
├── requirements.txt                  # Pinned Python package dependencies
├── .gitignore                        # Git ignore patterns
│
├── data/
│   ├── raw/                          # Cached raw daily prices from Yahoo Finance
│   └── processed/                    # Aligned price matrices and return series
│
├── models/
│   ├── final_model.joblib            # Production-fitted Elastic Net model
│   ├── feature_columns.json          # Strict feature column ordering
│   └── model_info.json               # Full model metadata and hyperparameter config
│
├── results/
│   ├── prediction_metrics.csv        # Out-of-sample RMSE, MAE, R², Dir Acc
│   ├── trading_development_grid.csv  # Grid search trading returns on dev set
│   ├── trading_holdout.csv           # Final holdout trading performance
│   ├── data_policy_report.csv        # Missing data handling and alignment audit
│   ├── walk_forward_windows.csv      # Walk-forward date ranges and embargo verification
│   ├── latest_prediction.csv         # Latest live market prediction
│   ├── vw_price_and_returns.png      # Price trajectory and return distribution plot
│   ├── rmse_vs_zero.png              # Model RMSE comparison bar chart
│   └── holdout_equity.png            # Holdout portfolio equity curves
│
├── reports/
│   ├── writeup.md                    # Academic project write-up (Markdown source)
│   ├── generate_pdf.py               # ReportLab PDF compiler script
│   └── UE24CS352A_MiniProject_Writeup.pdf  # Final compiled 2-Page submission PDF
│
├── presentation/
│   └── index.html                    # Interactive HTML5 slide deck for oral review
│
└── web/
    ├── index.html                    # Modern interactive web demonstration UI
    ├── style.css                     # Premium dark-mode glassmorphic styling
    ├── app.js                        # Real-time inference engine and canvas charts
    └── server.py                     # Lightweight demonstration web server
```

---

## 📋 Submission Deliverables Checklist

- [x] **A. Source Code in GitHub Repository:** Complete, modular, reproducible codebase with unit tests and pipeline runner.
- [x] **B. One-Page Write-up (2-Page PDF):** Formatted academic PDF (`reports/UE24CS352A_MiniProject_Writeup.pdf`) adhering to all criteria.
- [x] **C. Review & Demonstration Slide Deck:** Interactive presentation deck (`presentation/index.html`) ready for the review session.
- [x] **D. Live Interactive Web Demonstration:** Fully responsive web application (`web/index.html`) for real-time model inference and backtest visualization.

---

## 📜 Disclaimer & Acknowledgements

*This project is conducted for educational and academic research purposes under the **UE24CS352A Machine Learning** course at PES University. All quantitative trading signals and results are exploratory and do not constitute financial advice.*

*Based on research by René M. Glawion, "Quantitative Trading with Machine Learning", Department of Computer Science, Stanford University (CS229).*
