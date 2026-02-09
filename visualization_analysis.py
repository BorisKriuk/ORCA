#!/usr/bin/env python3
"""
CGECD Comprehensive Visualization Analysis
==========================================

1. SHAP Analysis (Individual + Group features)
2. Feature Importance Rankings
3. Performance by Module (Spectral vs Traditional vs Combined)
4. Model Comparison Visualizations
"""

import os
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Try SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

warnings.filterwarnings('ignore')
np.random.seed(42)

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY") or os.getenv("API_KEY")

OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")


# =============================================================================
# DATA LOADING (same as before)
# =============================================================================
class DataLoader:
    BASE_URL = "https://eodhd.com/api"

    def __init__(self, api_key: str):
        self.api_key = api_key
        import requests
        self.session = requests.Session()

    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_path = CACHE_DIR / f"{symbol.replace('.', '_').replace('/', '_')}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return pd.DataFrame()


def load_data(years: int = 15) -> pd.DataFrame:
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

    return pd.DataFrame(data_dict).dropna(how='all').ffill(limit=5).dropna()


# =============================================================================
# FEATURE EXTRACTION WITH CATEGORIES
# =============================================================================
def extract_spectral_features(prices: pd.DataFrame, window: int = 60):
    """Extract spectral features with category labels."""
    returns = prices.pct_change().dropna()
    n_assets = len(prices.columns)

    features_list = []
    feature_categories = {}

    for i in range(window, len(returns)):
        date = returns.index[i]
        window_returns = returns.iloc[i-window:i]
        corr = np.nan_to_num(window_returns.corr().values, nan=0)
        np.fill_diagonal(corr, 1.0)
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

        # === EIGENVALUE FEATURES ===
        row['lambda_1'] = eigenvalues[0]
        row['lambda_1_ratio'] = eigenvalues[0] / total_var
        row['lambda_2'] = eigenvalues[1] if n > 1 else 0
        row['spectral_gap'] = eigenvalues[0] / (eigenvalues[1] + 1e-10)

        for k in [1, 3, 5]:
            row[f'absorption_ratio_{k}'] = np.sum(eigenvalues[:min(k, n)]) / total_var

        normalized_eig = eigenvalues / total_var
        entropy = -np.sum(normalized_eig * np.log(normalized_eig + 1e-10))
        row['eigenvalue_entropy'] = entropy
        row['effective_rank'] = np.exp(entropy)
        row['eigenvalue_std'] = np.std(eigenvalues)
        row['eigenvalue_skew'] = stats.skew(eigenvalues)

        # === CORRELATION FEATURES ===
        upper_tri = corr[np.triu_indices(n_assets, k=1)]
        row['mean_abs_corr'] = np.mean(np.abs(upper_tri))
        row['max_abs_corr'] = np.max(np.abs(upper_tri))
        row['corr_std'] = np.std(upper_tri)
        row['frac_corr_above_50'] = np.mean(np.abs(upper_tri) > 0.5)
        row['frac_corr_above_70'] = np.mean(np.abs(upper_tri) > 0.7)

        # === GRAPH TOPOLOGY FEATURES ===
        for thresh in [0.3, 0.5, 0.7]:
            adj = (np.abs(corr) > thresh).astype(float)
            np.fill_diagonal(adj, 0)
            n_edges = np.sum(adj) / 2
            max_edges = n_assets * (n_assets - 1) / 2
            row[f'edge_density_t{int(thresh*100)}'] = n_edges / max_edges

        features_list.append(row)

    df = pd.DataFrame(features_list).set_index('date')

    # Define categories
    eigenvalue_features = ['lambda_1', 'lambda_1_ratio', 'lambda_2', 'spectral_gap',
                          'absorption_ratio_1', 'absorption_ratio_3', 'absorption_ratio_5',
                          'eigenvalue_entropy', 'effective_rank', 'eigenvalue_std', 'eigenvalue_skew']
    correlation_features = ['mean_abs_corr', 'max_abs_corr', 'corr_std',
                           'frac_corr_above_50', 'frac_corr_above_70']
    topology_features = ['edge_density_t30', 'edge_density_t50', 'edge_density_t70']

    for f in eigenvalue_features:
        if f in df.columns:
            feature_categories[f] = 'Eigenvalue'
    for f in correlation_features:
        if f in df.columns:
            feature_categories[f] = 'Correlation'
    for f in topology_features:
        if f in df.columns:
            feature_categories[f] = 'Topology'

    # Add dynamics features
    key_features = ['lambda_1', 'absorption_ratio_1', 'mean_abs_corr']
    for feat in key_features:
        if feat in df.columns:
            for lb in [5, 10, 20]:
                col_roc = f'{feat}_roc_{lb}d'
                col_zscore = f'{feat}_zscore_{lb}d'
                df[col_roc] = df[feat].pct_change(lb)
                rolling_mean = df[feat].rolling(lb * 2).mean()
                rolling_std = df[feat].rolling(lb * 2).std()
                df[col_zscore] = (df[feat] - rolling_mean) / (rolling_std + 1e-10)
                feature_categories[col_roc] = 'Dynamics'
                feature_categories[col_zscore] = 'Dynamics'

    return df, feature_categories


def extract_traditional_features(prices: pd.DataFrame):
    """Extract traditional features with categories."""
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    returns = market.pct_change()

    features = pd.DataFrame(index=prices.index)
    feature_categories = {}

    # Returns
    for w in [1, 5, 10, 20]:
        col = f'return_{w}d'
        features[col] = market.pct_change(w)
        feature_categories[col] = 'Returns'

    # Volatility
    for w in [5, 10, 20]:
        col = f'volatility_{w}d'
        features[col] = returns.rolling(w).std() * np.sqrt(252)
        feature_categories[col] = 'Volatility'

    features['vol_ratio'] = features['volatility_5d'] / (features['volatility_20d'] + 1e-8)
    feature_categories['vol_ratio'] = 'Volatility'

    # Momentum
    for w in [10, 20]:
        col = f'price_to_sma_{w}'
        sma = market.rolling(w).mean()
        features[col] = market / sma - 1
        feature_categories[col] = 'Momentum'

    # RSI
    delta = market.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    feature_categories['rsi_14'] = 'Momentum'

    # Drawdown
    for w in [20, 60]:
        col = f'drawdown_{w}d'
        rolling_max = market.rolling(w).max()
        features[col] = (market - rolling_max) / rolling_max
        feature_categories[col] = 'Drawdown'

    return features, feature_categories


def compute_target(prices: pd.DataFrame, target_type: str = 'vol_extreme_10d'):
    """Compute target variable."""
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    returns = market.pct_change()

    if target_type == 'vol_extreme_10d':
        vol = returns.rolling(20).std() * np.sqrt(252)
        future_vol = vol.shift(-10)
        vol_thresh = vol.rolling(252).quantile(0.9)
        return (future_vol > vol_thresh).astype(int)
    elif target_type == 'drawdown_5pct_10d':
        future_dd = pd.Series(index=market.index, dtype=float)
        for i in range(len(market) - 10):
            current = market.iloc[i]
            future_min = market.iloc[i+1:i+11].min()
            future_dd.iloc[i] = (future_min - current) / current
        return (future_dd < -0.05).astype(int)

    return None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_shap_analysis(model, X_train, X_test, feature_names, feature_categories, output_dir):
    """Create SHAP analysis visualizations."""
    if not SHAP_AVAILABLE:
        print("SHAP not available, skipping SHAP analysis")
        return

    print("  Computing SHAP values...")

    # Use TreeExplainer for RF
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])  # Limit for speed

    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (positive)

    # 1. SHAP Summary Plot (Individual Features)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names,
                     show=False, max_display=20)
    plt.title('SHAP Feature Importance (Individual)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_individual.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. SHAP Bar Plot (Mean |SHAP|)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names,
                     plot_type='bar', show=False, max_display=20)
    plt.title('Mean |SHAP| Values by Feature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. SHAP by Feature Group
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    })

    # Map to categories
    feature_importance['category'] = feature_importance['feature'].map(
        lambda x: feature_categories.get(x, 'Other')
    )

    # Group importance
    group_importance = feature_importance.groupby('category')['importance'].sum().sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    colors = {'Eigenvalue': '#27ae60', 'Correlation': '#3498db', 'Topology': '#9b59b6',
              'Dynamics': '#e74c3c', 'Returns': '#f39c12', 'Volatility': '#1abc9c',
              'Momentum': '#34495e', 'Drawdown': '#e67e22', 'Other': '#95a5a6'}
    bar_colors = [colors.get(cat, '#95a5a6') for cat in group_importance.index]

    bars = plt.barh(group_importance.index, group_importance.values, color=bar_colors, alpha=0.8)
    plt.xlabel('Sum of |SHAP| Values', fontsize=12)
    plt.title('SHAP Importance by Feature Group', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, group_importance.values):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'shap_group.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  SHAP analysis complete")
    return feature_importance


def plot_feature_importance_rf(model, feature_names, feature_categories, output_dir):
    """Plot Random Forest feature importance."""
    importance = model.feature_importances_

    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    feat_imp['category'] = feat_imp['feature'].map(
        lambda x: feature_categories.get(x, 'Other')
    )

    # Top 20 features
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Individual features
    ax = axes[0]
    top20 = feat_imp.head(20)
    colors = {'Eigenvalue': '#27ae60', 'Correlation': '#3498db', 'Topology': '#9b59b6',
              'Dynamics': '#e74c3c', 'Returns': '#f39c12', 'Volatility': '#1abc9c',
              'Momentum': '#34495e', 'Drawdown': '#e67e22', 'Other': '#95a5a6'}
    bar_colors = [colors.get(cat, '#95a5a6') for cat in top20['category']]

    ax.barh(range(len(top20)), top20['importance'].values, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Features (Random Forest)', fontsize=12, fontweight='bold')

    # Add legend
    legend_elements = [mpatches.Patch(color=c, label=l) for l, c in colors.items()
                      if l in top20['category'].values]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # Group importance
    ax = axes[1]
    group_imp = feat_imp.groupby('category')['importance'].sum().sort_values(ascending=True)
    bar_colors = [colors.get(cat, '#95a5a6') for cat in group_imp.index]

    ax.barh(group_imp.index, group_imp.values, color=bar_colors, alpha=0.8)
    ax.set_xlabel('Sum of Importance')
    ax.set_title('Importance by Feature Group', fontsize=12, fontweight='bold')

    plt.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    return feat_imp


def plot_module_performance(results_by_module, output_dir):
    """Plot performance comparison across different feature modules."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    modules = list(results_by_module.keys())
    aucs = [results_by_module[m]['auc'] for m in modules]
    precisions = [results_by_module[m]['precision'] for m in modules]
    recalls = [results_by_module[m]['recall'] for m in modules]
    f1s = [results_by_module[m]['f1'] for m in modules]

    colors = {'Spectral Only': '#27ae60', 'Traditional Only': '#3498db',
              'Combined': '#9b59b6', 'Eigenvalue Only': '#2ecc71',
              'Correlation Only': '#1abc9c', 'Dynamics Only': '#e74c3c'}
    bar_colors = [colors.get(m, '#95a5a6') for m in modules]

    # AUC
    ax = axes[0, 0]
    bars = ax.bar(modules, aucs, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('AUC-ROC by Feature Module', fontsize=12, fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
               ha='center', fontsize=10, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Precision
    ax = axes[0, 1]
    ax.bar(modules, precisions, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_ylabel('Precision')
    ax.set_title('Precision by Feature Module', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Recall
    ax = axes[1, 0]
    ax.bar(modules, recalls, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_ylabel('Recall')
    ax.set_title('Recall by Feature Module', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # F1
    ax = axes[1, 1]
    ax.bar(modules, f1s, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Feature Module', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Performance Comparison: Feature Modules', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'module_performance.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_by_model, output_dir):
    """Plot performance comparison across different models."""

    models = list(results_by_model.keys())
    aucs = [results_by_model[m]['auc'] for m in models]

    # Sort by AUC
    sorted_idx = np.argsort(aucs)[::-1]
    models = [models[i] for i in sorted_idx]
    aucs = [aucs[i] for i in sorted_idx]

    colors = {'CGECD (RF)': '#27ae60', 'Gradient Boosting': '#9b59b6',
              'Logistic Regression': '#e74c3c', 'SVM': '#f39c12',
              'Random Baseline': '#95a5a6'}

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_colors = [colors.get(m, '#3498db') for m in models]
    bars = ax.barh(models, aucs, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Random (0.5)')

    # Add value labels
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
               va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('AUC-ROC', fontsize=12)
    ax.set_title('Model Comparison: AUC-ROC', fontsize=14, fontweight='bold')
    ax.set_xlim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_module_experiments(X_spectral, X_traditional, y, feature_names_spectral,
                           feature_names_traditional, feature_categories):
    """Run experiments with different feature modules."""

    from sklearn.model_selection import train_test_split

    results = {}

    # Split data
    train_size = int(len(y) * 0.7)
    X_spec_train, X_spec_test = X_spectral[:train_size], X_spectral[train_size:]
    X_trad_train, X_trad_test = X_traditional[:train_size], X_traditional[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scale
    scaler_spec = RobustScaler()
    scaler_trad = RobustScaler()

    X_spec_train_s = np.nan_to_num(scaler_spec.fit_transform(X_spec_train), nan=0)
    X_spec_test_s = np.nan_to_num(scaler_spec.transform(X_spec_test), nan=0)
    X_trad_train_s = np.nan_to_num(scaler_trad.fit_transform(X_trad_train), nan=0)
    X_trad_test_s = np.nan_to_num(scaler_trad.transform(X_trad_test), nan=0)

    # Combined
    X_comb_train = np.hstack([X_spec_train_s, X_trad_train_s])
    X_comb_test = np.hstack([X_spec_test_s, X_trad_test_s])

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                class_weight='balanced_subsample', random_state=42, n_jobs=-1)

    # Test each module
    modules = {
        'Spectral Only': (X_spec_train_s, X_spec_test_s),
        'Traditional Only': (X_trad_train_s, X_trad_test_s),
        'Combined': (X_comb_train, X_comb_test),
    }

    for name, (X_train, X_test) in modules.items():
        print(f"  Testing {name}...")
        model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                       class_weight='balanced_subsample', random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        results[name] = {
            'auc': roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5,
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0),
            'model': model if name == 'Combined' else None
        }

    return results, X_comb_train, X_comb_test, y_train, y_test


def run_model_experiments(X_train, X_test, y_train, y_test):
    """Run experiments with different models."""

    models = {
        'CGECD (RF)': RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                             class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"  Testing {name}...")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        results[name] = {
            'auc': roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5,
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0),
            'model': model if name == 'CGECD (RF)' else None
        }

    # Add random baseline
    results['Random Baseline'] = {
        'auc': 0.5,
        'precision': y_test.mean(),
        'recall': 0.5,
        'f1': 0.0
    }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("CGECD COMPREHENSIVE VISUALIZATION ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading data...")
    prices = load_data(years=15)
    print(f"  Loaded {len(prices)} days, {len(prices.columns)} assets")

    # Extract features
    print("\n[2/6] Extracting features...")
    spectral_df, spectral_categories = extract_spectral_features(prices)
    traditional_df, traditional_categories = extract_traditional_features(prices)

    # Merge categories
    all_categories = {**spectral_categories, **traditional_categories}

    print(f"  Spectral: {len(spectral_df.columns)} features")
    print(f"  Traditional: {len(traditional_df.columns)} features")

    # Compute target
    print("\n[3/6] Computing target...")
    target = compute_target(prices, 'vol_extreme_10d')

    # Align data
    common_idx = spectral_df.dropna().index.intersection(traditional_df.dropna().index)
    common_idx = common_idx.intersection(target.dropna().index)

    X_spectral = spectral_df.loc[common_idx].values
    X_traditional = traditional_df.loc[common_idx].values
    y = target.loc[common_idx].values

    feature_names_spectral = list(spectral_df.columns)
    feature_names_traditional = list(traditional_df.columns)
    all_feature_names = feature_names_spectral + feature_names_traditional

    print(f"  Final dataset: {len(y)} samples, {y.mean():.1%} positive rate")

    # Run module experiments
    print("\n[4/6] Running module experiments...")
    module_results, X_train, X_test, y_train, y_test = run_module_experiments(
        X_spectral, X_traditional, y,
        feature_names_spectral, feature_names_traditional, all_categories
    )

    # Run model experiments
    print("\n[5/6] Running model experiments...")
    model_results = run_model_experiments(X_train, X_test, y_train, y_test)

    # Create visualizations
    print("\n[6/6] Creating visualizations...")

    # Module performance
    plot_module_performance(module_results, OUTPUT_DIR)
    print("  ✓ Module performance plot")

    # Model comparison
    plot_model_comparison(model_results, OUTPUT_DIR)
    print("  ✓ Model comparison plot")

    # Feature importance (RF)
    combined_model = module_results['Combined']['model']
    feat_imp = plot_feature_importance_rf(combined_model, all_feature_names, all_categories, OUTPUT_DIR)
    print("  ✓ Feature importance plot")

    # SHAP analysis
    if SHAP_AVAILABLE:
        plot_shap_analysis(combined_model, X_train, X_test, all_feature_names, all_categories, OUTPUT_DIR)
        print("  ✓ SHAP analysis plots")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\nModule Performance:")
    for name, res in module_results.items():
        print(f"  {name:20s}: AUC={res['auc']:.3f}, Prec={res['precision']:.1%}, Recall={res['recall']:.1%}")

    print("\nModel Performance:")
    for name, res in sorted(model_results.items(), key=lambda x: -x[1]['auc']):
        print(f"  {name:20s}: AUC={res['auc']:.3f}")

    print(f"\nVisualizations saved to {OUTPUT_DIR}/")
    print("\nFiles:")
    for f in OUTPUT_DIR.glob('*.png'):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
