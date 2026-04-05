# CGECD: Correlation Graph Eigenvalue Crisis Detector

**Detecting Financial Market Crises via Spectral Analysis of Dynamic Correlation Networks**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Abstract.** We propose CGECD, a framework that exploits spectral properties of dynamic asset correlation networks to detect extreme tail events in equity markets. By extracting eigenvalue-based features from rolling correlation matrices — including absorption ratios, spectral entropy, effective rank, and Marchenko-Pastur excess — we capture structural regime shifts that precede crises and rallies at a 10-day horizon. On 15 years of daily US market data with walk-forward validation, CGECD achieves a Balanced Crisis Detection AUC (BCD-AUC) of **0.741**, ranking first against five competitive baselines including HAR-RV, GARCH, and turbulence-based methods. Ablation studies confirm that spectral features contribute +5.2% AUC for rally detection and +10.3% AUC for crash detection over traditional volatility features alone. A backtested ensemble walk-forward strategy translates these signals into a **Sharpe ratio of 1.13** with 15.6% CAGR and only -7.5% maximum drawdown.

---

## 1. Key Results

### Classification Performance (Walk-Forward Out-of-Sample)

| Model | Rally AUC | Crash AUC | BCD-AUC | # Features |
|:------|:---------:|:---------:|:-------:|:----------:|
| **CGECD Combined (Ours)** | **0.772** | 0.711 | **0.741** | 206 |
| HAR-RV (Corsi, 2009) | 0.696 | **0.763** | 0.729 | 4 |
| SMA Volatility | 0.682 | 0.671 | 0.676 | 9 |
| Traditional RF | 0.736 | 0.608 | 0.669 | 27 |
| Spectral Only RF | 0.620 | 0.666 | 0.643 | 179 |
| Turbulence (Kritzman & Li, 2010) | 0.647 | 0.565 | 0.605 | 5 |

**BCD-AUC** = sqrt(Rally AUC x Crash AUC) rewards balanced detection in both directions.

### Backtesting Performance

| Strategy | Sharpe | CAGR | Max DD | Calmar | Avg Leverage |
|:---------|:------:|:----:|:------:|:------:|:------------:|
| **Ensemble WFO (Ours)** | **1.13** | **15.6%** | **-7.5%** | **2.09** | 0.39x |
| Full Stack | 0.44 | 8.0% | -8.8% | 0.91 | 0.47x |
| RORO Crash | 0.43 | 9.9% | -15.0% | 0.66 | 0.60x |
| Buy & Hold (Benchmark) | 0.09 | 3.7% | -33.7% | 0.11 | 1.00x |

---

## 2. Motivation

During market crises, normally uncorrelated assets become highly correlated, causing measurable changes in the correlation matrix eigenvalue spectrum **before** the full crisis materializes. Standard volatility indicators (GARCH, realized variance) react *after* the event; spectral features detect the structural shifts *during formation*.

<p align="center"><b>Stable regime:</b> eigenvalues spread &rarr; diversification intact<br><b>Pre-crisis:</b> dominant eigenvalue absorbs variance &rarr; herding begins<br><b>Crisis:</b> spectral gap collapses &rarr; contagion underway</p>

---

## 3. Methodology

### 3.1 Correlation Graph Construction

We compute rolling correlation matrices across a universe of **24 diversified assets** (equities, sectors, international, fixed income, commodities, currencies) using three parallel estimators:

- **60-day rolling window** (short-term regime)
- **120-day rolling window** (medium-term regime)
- **Exponentially weighted** (halflife=30 days, adaptive)

### 3.2 Spectral Feature Extraction (127 features)

From each correlation matrix, we extract:

| Feature Group | Examples | Intuition |
|:-------------|:---------|:----------|
| **Eigenvalues** | lambda_1, lambda_2, lambda_3 | Variance concentration across principal components |
| **Spectral Gap** | lambda_1 / lambda_2 | Single-factor dominance (herding indicator) |
| **Absorption Ratios** | AR_k = sum(lambda_1..k) / trace, k in {1,3,5} | Fraction of total variance explained by top factors |
| **Eigenvalue Entropy** | -sum(p_i log p_i), p_i = lambda_i / trace | Information-theoretic spectral diversity |
| **Effective Rank** | exp(entropy) | True degrees of freedom in correlation structure |
| **Marchenko-Pastur Excess** | Distance from RMT null | Signal-to-noise separation (real correlations vs noise) |
| **Condition Number** | lambda_max / lambda_min | Matrix ill-conditioning (systemic fragility) |
| **Eigenvector Concentration** | Entropy of v_1 loadings | Whether dominant factor is broad market or sector-specific |
| **Graph Topology** | Edge density, clustering coefficient, centralization | Network connectivity at multiple correlation thresholds |
| **Dynamics** | 5/10/20-day ROC, z-scores, acceleration | Rate of change in spectral properties |

### 3.3 Traditional Features (79 features)

Multi-horizon returns, realized and GARCH volatility, drawdowns, momentum, RSI, higher moments (skewness, kurtosis), cross-asset dispersion.

### 3.4 Model

- **Algorithm:** Random Forest (200 trees, max_depth=6, class_weight='balanced_subsample')
- **Preprocessing:** RobustScaler
- **Combined feature set:** 206 features (127 spectral + 79 traditional)

### 3.5 Walk-Forward Validation

Temporal cross-validation with strict anti-leakage guarantees:

```
|--- 3yr train ---|-- 10d gap --|--- 6mo test ---|
                               |--- 3yr train ---|-- 10d gap --|--- 6mo test ---|
                                                               |--- 3yr train ---|...
```

- **8 folds** across 15 years of data
- **10-day gap** between train and test to prevent look-ahead bias
- All metrics reported on out-of-sample test folds only

---

## 4. Prediction Tasks

We evaluate on two complementary tasks to ensure balanced crisis detection:

| Task | Target Definition | Base Rate | Horizon |
|:-----|:-----------------|:---------:|:-------:|
| **Rally Detection** | S&P 500 up > 3% within 10 days | ~7.7% | 10 days |
| **Crash Detection** | S&P 500 max drawdown > 7% within 10 days | ~0.9% | 10 days |

Both are rare event detection problems requiring careful handling of class imbalance.

---

## 5. Ablation Study

Spectral features provide substantial gains, especially for crash detection:

| Feature Set | Rally AUC | Crash AUC | Delta vs Traditional |
|:-----------|:---------:|:---------:|:--------------------:|
| Traditional only | 0.736 | 0.608 | baseline |
| Spectral only | 0.620 | 0.666 | -0.116 / +0.058 |
| **Combined (CGECD)** | **0.772** | **0.711** | **+0.036 / +0.103** |

The spectral contribution for crash detection (+10.3% AUC) indicates that correlation network structure contains crisis precursors absent from price-based volatility features.

---

## 6. Asset Universe

| Category | Assets |
|:---------|:-------|
| **Broad Equity** | SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000) |
| **Sectors** | XLF, XLE, XLK, XLV, XLU, XLP, XLY, XLI, XLB, XLRE |
| **International** | EFA (Developed), EEM (Emerging), VGK (Europe), EWJ (Japan) |
| **Fixed Income** | TLT (Long Treasury), IEF (Intermediate), LQD (IG Corp), HYG (HY) |
| **Alternatives** | GLD (Gold), USO (Oil), UUP (USD), VNQ (REITs) |

---

## 7. Getting Started

### Installation

```bash
git clone https://github.com/BorisKriuk/RALEC-GNN.git
cd RALEC-GNN
pip install -r requirements.txt
```

### Configuration

Set your EODHD API key (required for data download):

```bash
export EODHD_API_KEY="your_api_key"
```

### Run Classification Experiment

```bash
python run.py
```

Outputs:
- `results/experiment_results.csv` — per-task metrics for all models
- `results/bcd_auc_results.csv` — combined BCD-AUC ranking

### Run Backtesting

```bash
python backtesting.py
```

Outputs:
- `results/backtest_results.csv` — strategy performance (Sharpe, CAGR, drawdown)
- `results/equity_curves.csv` — daily equity curves
- `results/backtest_positions.csv` — position history

### Interactive Dashboard

```bash
python visualization_server.py
# Open http://localhost:5000
```

---

## 8. Project Structure

```
RALEC-GNN/
├── algorithm.py            # Core CGECD: spectral feature extraction, model, walk-forward
├── run.py                  # Two-task classification experiment runner
├── backtesting.py          # 10 trading strategies with ensemble WFO
├── benchmarks.py           # Baseline models (HAR-RV, GARCH, Turbulence, SMA Vol)
├── config.py               # Hyperparameters, asset universe, walk-forward settings
├── data_loader.py          # EODHD API client with local caching
├── metrics.py              # AUC, precision/recall, Brier score, bootstrap CI
├── visualization_server.py # Flask dashboard for interactive result exploration
├── requirements.txt
├── results/                # Experiment outputs (CSVs, figures)
└── cache/                  # Cached price data (auto-generated)
```

---

## 9. Requirements

- Python >= 3.9
- numpy, pandas, scikit-learn, scipy, statsmodels
- matplotlib, seaborn (visualization)
- flask, flask-cors (dashboard)
- requests, python-dotenv (data retrieval)
- ta (technical analysis indicators)

Full pinned versions in `requirements.txt`.

---

## 10. Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kriuk2026cgecd,
  title={CGECD: Correlation Graph Eigenvalue Crisis Detector},
  author={Kriuk, Boris and Kriuk, Fedor},
  year={2026}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
