#!/usr/bin/env python3
"""
CGECD Benchmark Comparison
==========================

Compare Boris's CGECD algorithm against standard ML benchmarks:
- XGBoost
- Gradient Boosting
- Logistic Regression
- SVM
- MLP Neural Network
- Naive baselines (Random, Historical Rate)

Uses the same data pipeline and walk-forward validation as Boris's algorithm.
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, skipping")

warnings.filterwarnings('ignore')
np.random.seed(42)

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY") or os.getenv("API_KEY")

OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")

# =============================================================================
# Import Boris's data loading and feature extraction from his algorithm
# =============================================================================

# We'll reuse the core components from algorithm-boris.py
import requests

class DataLoader:
    BASE_URL = "https://eodhd.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_file = symbol.replace('.', '_').replace('/', '_')
        cache_path = CACHE_DIR / f"{cache_file}.pkl"

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return pd.DataFrame()


def load_multi_asset_data(years: int = 15) -> pd.DataFrame:
    """Load asset data (same as Boris's algorithm)"""
    loader = DataLoader(API_KEY)

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    symbols = {
        'SPY.US': 'SP500', 'QQQ.US': 'Nasdaq100', 'IWM.US': 'Russell2000',
        'XLF.US': 'Financials', 'XLE.US': 'Energy', 'XLK.US': 'Technology',
        'XLV.US': 'Healthcare', 'XLU.US': 'Utilities', 'XLP.US': 'ConsumerStaples',
        'XLY.US': 'ConsumerDisc', 'XLI.US': 'Industrials', 'XLB.US': 'Materials',
        'XLRE.US': 'RealEstate', 'EFA.US': 'DevIntl', 'EEM.US': 'EmergingMkts',
        'VGK.US': 'Europe', 'EWJ.US': 'Japan', 'TLT.US': 'LongTreasury',
        'IEF.US': 'IntermTreasury', 'LQD.US': 'InvGradeCorp', 'HYG.US': 'HighYield',
        'GLD.US': 'Gold', 'USO.US': 'Oil', 'UUP.US': 'USDollar', 'VNQ.US': 'REITs',
    }

    data_dict = {}
    for symbol, name in symbols.items():
        df = loader.get_data(symbol, start_date, end_date)
        if not df.empty and 'adjusted_close' in df.columns:
            data_dict[name] = df['adjusted_close']

    prices = pd.DataFrame(data_dict).dropna(how='all').ffill(limit=5).dropna()
    return prices


def extract_spectral_features(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Extract Boris's spectral features from correlation matrices"""
    returns = prices.pct_change().dropna()
    n_assets = len(prices.columns)

    features_list = []

    for i in range(window, len(returns)):
        date = returns.index[i]
        window_returns = returns.iloc[i-window:i]
        corr = window_returns.corr().values
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 1.0)

        # Eigenvalue decomposition
        corr = (corr + corr.T) / 2
        try:
            eigenvalues = np.linalg.eigvalsh(corr)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 1e-10)
        except:
            continue

        n = len(eigenvalues)
        total_var = np.sum(eigenvalues)

        row = {'date': date}

        # Primary eigenvalue features
        row['lambda_1'] = eigenvalues[0]
        row['lambda_1_ratio'] = eigenvalues[0] / total_var
        row['lambda_2'] = eigenvalues[1] if n > 1 else 0
        row['spectral_gap'] = eigenvalues[0] / (eigenvalues[1] + 1e-10)

        # Absorption ratios
        for k in [1, 3, 5]:
            row[f'absorption_ratio_{k}'] = np.sum(eigenvalues[:min(k, n)]) / total_var

        # Entropy
        normalized_eig = eigenvalues / total_var
        entropy = -np.sum(normalized_eig * np.log(normalized_eig + 1e-10))
        row['eigenvalue_entropy'] = entropy
        row['effective_rank'] = np.exp(entropy)

        # Higher moments
        row['eigenvalue_std'] = np.std(eigenvalues)
        row['eigenvalue_skew'] = stats.skew(eigenvalues)

        # Correlation statistics
        upper_tri = corr[np.triu_indices(n_assets, k=1)]
        row['mean_abs_corr'] = np.mean(np.abs(upper_tri))
        row['max_abs_corr'] = np.max(np.abs(upper_tri))
        row['frac_corr_above_50'] = np.mean(np.abs(upper_tri) > 0.5)
        row['frac_corr_above_70'] = np.mean(np.abs(upper_tri) > 0.7)

        # Edge density
        for thresh in [0.3, 0.5, 0.7]:
            adj = (np.abs(corr) > thresh).astype(float)
            np.fill_diagonal(adj, 0)
            n_edges = np.sum(adj) / 2
            max_edges = n_assets * (n_assets - 1) / 2
            row[f'edge_density_t{int(thresh*100)}'] = n_edges / max_edges

        features_list.append(row)

    df = pd.DataFrame(features_list).set_index('date')

    # Add dynamics features (rate of change, z-scores)
    key_features = ['lambda_1', 'lambda_1_ratio', 'absorption_ratio_1',
                   'eigenvalue_entropy', 'mean_abs_corr']

    for feat in key_features:
        if feat in df.columns:
            for lb in [5, 10, 20]:
                df[f'{feat}_roc_{lb}d'] = df[feat].pct_change(lb)
                rolling_mean = df[feat].rolling(lb * 2).mean()
                rolling_std = df[feat].rolling(lb * 2).std()
                df[f'{feat}_zscore_{lb}d'] = (df[feat] - rolling_mean) / (rolling_std + 1e-10)

    return df


def extract_traditional_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Extract traditional technical features"""
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    returns = market.pct_change()

    features = pd.DataFrame(index=prices.index)

    for w in [1, 5, 10, 20, 60]:
        features[f'return_{w}d'] = market.pct_change(w)

    for w in [5, 10, 20, 60]:
        features[f'volatility_{w}d'] = returns.rolling(w).std() * np.sqrt(252)

    features['vol_ratio'] = features['volatility_5d'] / (features['volatility_20d'] + 1e-8)

    for w in [10, 20, 50]:
        sma = market.rolling(w).mean()
        features[f'price_to_sma_{w}'] = market / sma - 1

    delta = market.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    for w in [20, 60]:
        rolling_max = market.rolling(w).max()
        features[f'drawdown_{w}d'] = (market - rolling_max) / rolling_max

    return features


def compute_targets(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute prediction targets"""
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    returns = market.pct_change()

    targets = pd.DataFrame(index=prices.index)

    # Drawdowns
    for horizon in [10, 20]:
        future_dd = pd.Series(index=market.index, dtype=float)
        for i in range(len(market) - horizon):
            current = market.iloc[i]
            future_min = market.iloc[i+1:i+horizon+1].min()
            future_dd.iloc[i] = (future_min - current) / current

        for thresh in [0.05, 0.07]:
            targets[f'drawdown_{int(thresh*100)}pct_{horizon}d'] = (future_dd < -thresh).astype(int)

    # Volatility spike
    vol = returns.rolling(20).std() * np.sqrt(252)
    for horizon in [10]:
        future_vol = vol.shift(-horizon)
        vol_thresh = vol.rolling(252).quantile(0.9)
        targets[f'vol_extreme_{horizon}d'] = (future_vol > vol_thresh).astype(int)

    return targets


# =============================================================================
# BENCHMARK MODELS
# =============================================================================

def get_benchmark_models() -> Dict:
    """Define all benchmark models to compare against CGECD"""

    models = {
        # Boris's CGECD model (Random Forest with his parameters)
        'CGECD (Boris)': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            min_samples_split=50,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),

        # Gradient Boosting
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),

        # Logistic Regression
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            C=0.1
        ),

        # SVM
        'SVM (RBF)': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42,
            C=1.0
        ),

        # MLP Neural Network
        'MLP Neural Net': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42
        ),

        # AdaBoost
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='auc'
        )

    return models


def walk_forward_evaluate(X, y, model, n_splits=5, train_years=3, test_months=6):
    """Walk-forward validation (same as Boris's algorithm)"""
    from sklearn.base import clone

    train_size = int(train_years * 252)
    test_size = int(test_months * 21)
    gap = 10

    if len(X) < train_size + gap + test_size:
        return None

    step = max(test_size, (len(X) - train_size - gap - test_size) // n_splits)

    all_probs, all_preds, all_actuals = [], [], []
    fold_aucs = []

    for fold in range(n_splits):
        start = fold * step
        train_end = start + train_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, len(X))

        if test_end > len(X):
            break

        X_train, y_train = X[start:train_end], y[start:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        # Scale
        scaler = RobustScaler()
        X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0)
        X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0, posinf=0, neginf=0)

        try:
            m = clone(model)
            m.fit(X_train_s, y_train)
            preds = m.predict(X_test_s)
            probs = m.predict_proba(X_test_s)[:, 1]

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_actuals.extend(y_test)

            if len(np.unique(y_test)) > 1:
                fold_aucs.append(roc_auc_score(y_test, probs))
        except Exception as e:
            print(f"    Fold failed: {e}")
            continue

    if not all_probs:
        return None

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    return {
        'auc': np.mean(fold_aucs) if fold_aucs else 0.5,
        'auc_std': np.std(fold_aucs) if fold_aucs else 0,
        'precision': precision_score(all_actuals, all_preds, zero_division=0),
        'recall': recall_score(all_actuals, all_preds, zero_division=0),
        'f1': f1_score(all_actuals, all_preds, zero_division=0),
        'probs': all_probs,
        'preds': all_preds,
        'actuals': all_actuals,
        'fold_aucs': fold_aucs
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_benchmark_visualizations(results: Dict, output_dir: Path):
    """Create publication-quality benchmark comparison visualizations"""

    plt.style.use('seaborn-v0_8-whitegrid')

    # Color scheme: CGECD in green, others in different colors
    colors = {
        'CGECD (Boris)': '#27ae60',  # Green - highlight Boris's model
        'XGBoost': '#3498db',
        'Gradient Boosting': '#9b59b6',
        'Logistic Regression': '#e74c3c',
        'SVM (RBF)': '#f39c12',
        'MLP Neural Net': '#1abc9c',
        'AdaBoost': '#34495e',
        'Random Baseline': '#95a5a6',
    }

    for target_name, target_results in results.items():
        if not target_results:
            continue

        # Sort by AUC
        sorted_models = sorted(target_results.items(), key=lambda x: -x[1].get('auc', 0))
        model_names = [m[0] for m in sorted_models]
        aucs = [m[1]['auc'] for m in sorted_models]
        auc_stds = [m[1].get('auc_std', 0) for m in sorted_models]
        precisions = [m[1]['precision'] for m in sorted_models]
        recalls = [m[1]['recall'] for m in sorted_models]

        # Figure 1: Bar chart comparison
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        # AUC bars
        ax = axes[0]
        bar_colors = [colors.get(m, '#95a5a6') for m in model_names]
        bars = ax.barh(model_names, aucs, xerr=auc_stds, color=bar_colors,
                      alpha=0.8, capsize=4, edgecolor='white', linewidth=1)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Random (0.5)')
        ax.set_xlabel('AUC-ROC', fontsize=12)
        ax.set_title('Model Comparison: AUC-ROC', fontsize=14, fontweight='bold')
        ax.set_xlim(0.4, 1.0)

        # Add value labels
        for bar, auc, std in zip(bars, aucs, auc_stds):
            ax.text(auc + std + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{auc:.3f}', va='center', fontsize=10, fontweight='bold')

        # Precision bars
        ax = axes[1]
        ax.barh(model_names, precisions, color=bar_colors, alpha=0.8,
               edgecolor='white', linewidth=1)
        ax.set_xlabel('Precision', fontsize=12)
        ax.set_title('Model Comparison: Precision', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)

        # Recall bars
        ax = axes[2]
        ax.barh(model_names, recalls, color=bar_colors, alpha=0.8,
               edgecolor='white', linewidth=1)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_title('Model Comparison: Recall', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)

        plt.suptitle(f'CGECD vs Benchmarks: {target_name}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        safe_name = target_name.replace(' ', '_').replace('>', '').replace('%', 'pct')
        plt.savefig(output_dir / f'benchmark_comparison_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 2: ROC Curves
        fig, ax = plt.subplots(figsize=(10, 8))

        for model_name, res in sorted_models:
            if 'probs' not in res or 'actuals' not in res:
                continue
            if len(np.unique(res['actuals'])) < 2:
                continue

            fpr, tpr, _ = roc_curve(res['actuals'], res['probs'])

            lw = 3 if model_name == 'CGECD (Boris)' else 1.5
            ls = '-' if model_name == 'CGECD (Boris)' else '--'
            color = colors.get(model_name, '#95a5a6')

            ax.plot(fpr, tpr, linewidth=lw, linestyle=ls, color=color,
                   label=f"{model_name} (AUC={res['auc']:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves: {target_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / f'roc_curves_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 3: Precision-Recall Curves
        fig, ax = plt.subplots(figsize=(10, 8))

        pos_rate = np.mean(sorted_models[0][1]['actuals'])

        for model_name, res in sorted_models:
            if 'probs' not in res or 'actuals' not in res:
                continue
            if len(np.unique(res['actuals'])) < 2:
                continue

            prec, rec, _ = precision_recall_curve(res['actuals'], res['probs'])

            lw = 3 if model_name == 'CGECD (Boris)' else 1.5
            color = colors.get(model_name, '#95a5a6')

            ax.plot(rec, prec, linewidth=lw, color=color, label=model_name)

        ax.axhline(y=pos_rate, color='red', linestyle='--', linewidth=2,
                  label=f'Random ({pos_rate:.1%})')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curves: {target_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / f'pr_curves_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 4: Summary heatmap across all targets
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(14, 8))

        all_models = set()
        for target_results in results.values():
            all_models.update(target_results.keys())
        all_models = sorted(all_models)

        data = []
        for target_name in results.keys():
            row = []
            for model in all_models:
                if model in results[target_name]:
                    row.append(results[target_name][model]['auc'])
                else:
                    row.append(0.5)
            data.append(row)

        df = pd.DataFrame(data, index=list(results.keys()), columns=all_models)

        # Reorder columns to put CGECD first
        cols = ['CGECD (Boris)'] + [c for c in df.columns if c != 'CGECD (Boris)']
        df = df[cols]

        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.6,
                   vmin=0.4, vmax=0.9, ax=ax, cbar_kws={'label': 'AUC-ROC'},
                   linewidths=0.5, linecolor='white')

        ax.set_title('AUC-ROC Comparison: CGECD vs All Benchmarks', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_dir / 'summary_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def run_benchmarks():
    """Run full benchmark comparison"""

    print("=" * 80)
    print("CGECD BENCHMARK COMPARISON")
    print("Comparing Boris's algorithm against standard ML models")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    prices = load_multi_asset_data(years=15)
    print(f"  Loaded {len(prices)} days, {len(prices.columns)} assets")

    # Extract features
    print("\n[2/4] Extracting features...")
    spectral_df = extract_spectral_features(prices)
    trad_df = extract_traditional_features(prices)
    combined_df = pd.concat([spectral_df, trad_df], axis=1)
    print(f"  Spectral: {len(spectral_df.columns)} features")
    print(f"  Traditional: {len(trad_df.columns)} features")
    print(f"  Combined: {len(combined_df.columns)} features")

    # Compute targets
    print("\n[3/4] Computing targets...")
    targets = compute_targets(prices)

    # Align data
    common_idx = combined_df.dropna().index.intersection(targets.dropna(how='all').index)
    combined_df = combined_df.loc[common_idx]
    targets = targets.loc[common_idx]
    print(f"  Final dataset: {len(common_idx)} samples")

    # Get models
    models = get_benchmark_models()
    print(f"\n  Models to compare: {list(models.keys())}")

    # Run benchmarks
    print("\n[4/4] Running benchmarks...")

    test_targets = [
        ('vol_extreme_10d', 'Extreme Volatility in 10 days'),
        ('drawdown_5pct_20d', 'Drawdown >5% in 20 days'),
        ('drawdown_7pct_20d', 'Drawdown >7% in 20 days'),
    ]

    all_results = {}
    summary_rows = []

    for target_col, target_name in test_targets:
        if target_col not in targets.columns:
            continue

        y = targets[target_col].values
        pos_rate = y.mean()

        if pos_rate < 0.02 or pos_rate > 0.5:
            continue

        print(f"\n--- {target_name} (positive rate: {pos_rate:.1%}) ---")

        X = combined_df.values
        target_results = {}

        for model_name, model in models.items():
            print(f"  Running {model_name}...", end=' ')
            res = walk_forward_evaluate(X, y, model, n_splits=5)

            if res:
                target_results[model_name] = res
                print(f"AUC={res['auc']:.3f}, Prec={res['precision']:.1%}, Recall={res['recall']:.1%}")

                summary_rows.append({
                    'Target': target_name,
                    'Model': model_name,
                    'AUC': res['auc'],
                    'AUC_std': res['auc_std'],
                    'Precision': res['precision'],
                    'Recall': res['recall'],
                    'F1': res['f1']
                })
            else:
                print("FAILED")

        # Add random baseline
        target_results['Random Baseline'] = {
            'auc': 0.5,
            'auc_std': 0,
            'precision': pos_rate,
            'recall': 0.5,
            'f1': 2 * pos_rate * 0.5 / (pos_rate + 0.5),
            'probs': np.random.rand(len(y)),
            'preds': np.random.randint(0, 2, len(y)),
            'actuals': y
        }

        all_results[target_name] = target_results

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / 'benchmark_summary.csv', index=False)

    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    create_benchmark_visualizations(all_results, OUTPUT_DIR)

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for target_name, target_results in all_results.items():
        print(f"\n{target_name}:")
        sorted_models = sorted(target_results.items(), key=lambda x: -x[1].get('auc', 0))

        cgecd_auc = target_results.get('CGECD (Boris)', {}).get('auc', 0)

        for rank, (model_name, res) in enumerate(sorted_models, 1):
            marker = "★" if model_name == 'CGECD (Boris)' else " "
            diff = res['auc'] - cgecd_auc if model_name != 'CGECD (Boris)' else 0
            diff_str = f"({diff:+.3f})" if diff != 0 else "(baseline)"

            print(f"  {rank}. {marker} {model_name:20s}: AUC={res['auc']:.3f} {diff_str}")

    print(f"\nResults saved to {OUTPUT_DIR}/")

    return all_results, summary_df


if __name__ == "__main__":
    results, summary = run_benchmarks()
